import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import csr_matrix
from eval_knn import evaluate_recommendations

def knn_matrix(checkins):
    user_mapping = {u: i for i, u in enumerate(checkins["user_id"].unique())}
    poi_mapping = {p: i for i, p in enumerate(checkins["poi_id"].unique())}
    checkins["user_idx"] = checkins["user_id"].map(user_mapping)
    checkins["poi_idx"] = checkins["poi_id"].map(poi_mapping)
    matrix = csr_matrix((checkins["score"], (checkins["user_idx"], checkins["poi_idx"])),
                        shape=(len(user_mapping), len(poi_mapping)))
    return matrix, user_mapping, poi_mapping

def get_user_climate_profiles(df, climate_cols):
    profiles = df.groupby("user_id")[climate_cols].mean().fillna(0)
    scaled = MinMaxScaler().fit_transform(profiles)
    return pd.DataFrame(scaled, index=profiles.index, columns=climate_cols)

def jaquard_weather_knn_recommendations(user_id, interaction_matrix, user_mapping, poi_mapping,
                                        user_climate_profiles, k=10, n=10, alpha=0.5):
    if user_id not in user_mapping or user_id not in user_climate_profiles.index:
        return []
    user_idx = user_mapping[user_id]
    bin_matrix = interaction_matrix.copy()
    bin_matrix.data = np.ones_like(bin_matrix.data)
    user_vector = bin_matrix[user_idx]
    intersections = user_vector.dot(bin_matrix.T).toarray().flatten()
    user_sum = user_vector.sum()
    other_sums = bin_matrix.sum(axis=1).A1
    unions = user_sum + other_sums - intersections
    jaccard_scores = np.divide(intersections, unions, out=np.zeros_like(intersections), where=unions != 0)
    jaccard_scores[user_idx] = 0

    common_users = user_climate_profiles.index.intersection(user_mapping.keys())
    idx_map = {u: i for i, u in enumerate(common_users)}
    climate_matrix = user_climate_profiles.loc[common_users].values
    sim_matrix = cosine_similarity(climate_matrix)
    sim_scores = sim_matrix[idx_map[user_id]] if user_id in idx_map else np.zeros_like(jaccard_scores)
    combined_scores = alpha * jaccard_scores + (1 - alpha) * sim_scores
    top_k = np.argsort(-combined_scores)[:k]

    user_pois = set(interaction_matrix[user_idx].nonzero()[1])
    recommended_pois = {}
    for neighbor_idx in top_k:
        neighbor_pois = interaction_matrix[neighbor_idx].nonzero()[1]
        for poi_idx in neighbor_pois:
            if poi_idx not in user_pois:
                poi_id = list(poi_mapping.keys())[poi_idx]
                recommended_pois[poi_id] = recommended_pois.get(poi_id, 0) + combined_scores[neighbor_idx]
    return [poi for poi, _ in sorted(recommended_pois.items(), key=lambda x: x[1], reverse=True)[:n]]

if __name__ == "__main__":
    CITY = 'New York' # Modifica la ciudad según sea necesario
    VARS = ["temp", "precip"]  # Variables climáticas a considerar
    agg = pd.read_csv(f"../../data/FoursquareProcessed/{CITY}_checkins_agg.txt", sep="\t", names=["user_id", "poi_id", "score"])
    weather = pd.read_csv(f"../../data/FoursquareProcessed/{CITY}_checkins_weather.txt", sep="\t",
                          names=["user_id", "poi_id", "temp", "precip", "snow", "windspeed", "visibility", "uvindex"])
    weather = weather.merge(agg, on=["user_id", "poi_id"], how="left").fillna(0)
    train, test = train_test_split(agg, test_size=0.3, random_state=42)
    _, test = train_test_split(test, test_size=2/3, random_state=42)
    matrix, user_map, poi_map = knn_matrix(train)
    user_profiles = get_user_climate_profiles(weather, VARS)
    results = evaluate_recommendations(test, train, jaquard_weather_knn_recommendations,
                                       matrix, user_map, poi_map, user_profiles, 10, 10, 0.5)
    print("KNN Jaccard + Clima:", results)
