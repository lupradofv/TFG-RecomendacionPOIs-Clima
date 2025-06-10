import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from mf_sgd_base import MatrixFactorizationSGD
from eval_mf import evaluate_recommendations

def re_rank_with_climate(recommendations, user_id, climate_df, weights):
    if not recommendations:
        return []

    climate_cols = list(weights.keys())
    user_context = climate_df[climate_df["user_id"] == user_id]
    rec_df = user_context[user_context["poi_id"].isin(recommendations)].copy()

    if rec_df.empty:
        return recommendations

    scaler = MinMaxScaler()
    rec_df[climate_cols] = scaler.fit_transform(rec_df[climate_cols])

    rec_df["climate_score"] = sum(rec_df[col] * weights[col] for col in climate_cols)
    reranked = rec_df.sort_values("climate_score", ascending=False)["poi_id"].tolist()

    return reranked


if __name__ == "__main__":
    CITY = "London" # Modifica según la ciudad deseada
    VARS = ['temp', 'precip', 'windspeed']  # Variables climáticas a usar
    weights = {var: 1 / len(VARS) for var in VARS}  

    df_agg = pd.read_csv(f"../../data/FoursquareProcessed/{CITY}_checkins_agg.txt", sep="\t",
                         names=["user_id", "poi_id", "score"])
    df_weather = pd.read_csv(f"../../data/FoursquareProcessed/{CITY}_checkins_weather.txt", sep="\t",
                             names=["user_id", "poi_id", "temp", "precip", "snow", "windspeed", "visibility", "uvindex"])

    df = df_weather.merge(df_agg, on=["user_id", "poi_id"], how="inner")
    df["score"] = np.log1p(df["score"])

    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(test_df, test_size=2/3, random_state=42)

    mf = MatrixFactorizationSGD(n_factors=20, n_epochs=25, learning_rate=0.01, reg=1e-4)
    mf.fit(train_df)

    # Wrapper con re-rankeo climático
    def mf_with_rerank(user_id, model, train_data, weather_df):
        recs = model.recommend(user_id, train_data, n=10)
        return re_rank_with_climate(recs, user_id, weather_df, weights)

    results = evaluate_recommendations(test_df, train_df, mf_with_rerank, mf, train_df, df)
    print("Resultados:")
    print(results)
