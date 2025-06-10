import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from mf_sgd_base import MatrixFactorizationSGD
from eval_mf import evaluate_recommendations

def combine_score_with_climate(df, climate_cols, alpha=0.5):
    df = df.copy()

    df["score"] = df["score"].fillna(1)
    df[climate_cols] = df[climate_cols].fillna(0)

    climate_scaler = MinMaxScaler()
    df[climate_cols] = climate_scaler.fit_transform(df[climate_cols])

    climate_component = df[climate_cols].mean(axis=1)
    combined = alpha * df["score"] + (1 - alpha) * climate_component
    combined = np.clip(combined, 0, 1)

    df["score"] = combined
    return df[["user_id", "poi_id", "score"]]


if __name__ == "__main__":

    CITY = "New York"
    ALPHA = 0.5
    VARS = ["temp", "precip", "snow", "windspeed", "visibility", "uvindex"]

    agg_path = f"../../data/FoursquareProcessed/{CITY}_checkins_agg.txt"
    weather_path = f"../../data/FoursquareProcessed/{CITY}_checkins_weather.txt"
    checkins_agg = pd.read_csv(agg_path, sep="\t", names=["user_id", "poi_id", "score"])
    checkins_weather = pd.read_csv(weather_path, sep="\t",
                                names=["user_id", "poi_id", "temp", "precip", "snow", "windspeed", "visibility", "uvindex"])
    
    df_weather = checkins_weather.merge(checkins_agg, on=["user_id", "poi_id"], how="left")
    df_weather["score"] = df_weather["score"].fillna(0)
    df_combined = combine_score_with_climate(df_weather, VARS, alpha=ALPHA)

    train_df, test_df = train_test_split(df_combined, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(test_df, test_size=2/3, random_state=42)

    # train_df["score"] = np.log1p(train_df["score"])

    mf = MatrixFactorizationSGD(n_factors=30, n_epochs=25, learning_rate=0.01, reg=0.1, verbose=False)
    mf.fit(train_df)

    def mf_wrapper(user_id, model, train_data):
        return model.recommend(user_id, train_data, n=10)

    results = evaluate_recommendations(test_df, train_df, mf_wrapper, mf, train_df)
    print(results)