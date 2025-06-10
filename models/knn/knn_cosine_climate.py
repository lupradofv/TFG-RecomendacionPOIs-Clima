import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from eval_knn import evaluate_recommendations

def knn_weather_masked_matrix(checkins_weather, climate_cols):
    agg_dict = {col: 'mean' for col in climate_cols}
    agg_dict['score'] = 'sum'
    grouped = checkins_weather.groupby(['user_id', 'poi_id']).agg(agg_dict).reset_index()

    user_mapping = {u: i for i, u in enumerate(grouped["user_id"].unique())}
    poi_mapping = {p: i for i, p in enumerate(grouped["poi_id"].unique())}
    grouped["user_idx"] = grouped["user_id"].map(user_mapping)
    grouped["poi_idx"] = grouped["poi_id"].map(poi_mapping)

    interaction_matrix = csr_matrix((grouped["score"], (grouped["user_idx"], grouped["poi_idx"])),
                                    shape=(len(user_mapping), len(poi_mapping)))

    user_profiles = grouped.groupby("user_id")[climate_cols].mean().fillna(0)
    user_profiles = pd.DataFrame(MinMaxScaler().fit_transform(user_profiles),
                                 index=user_profiles.index, columns=climate_cols)

    sim_matrix = cosine_similarity(user_profiles.values)
    np.fill_diagonal(sim_matrix, 0)

    weighted_matrix = sim_matrix @ interaction_matrix.toarray()
    mask = (interaction_matrix.toarray() > 0).astype(float)
    final_matrix = weighted_matrix * mask
    return csr_matrix(final_matrix), user_mapping, poi_mapping

def cosine_climate_knn_recommendations(user_id, interaction_matrix, user_mapping, poi_mapping, k=10, n=10):
    if user_id not in user_mapping:
        return []

    user_idx = user_mapping[user_id]
    sim_matrix = cosine_similarity(interaction_matrix)
    sim_matrix[user_idx, user_idx] = 0

    top_k = np.argsort(-sim_matrix[user_idx])[:k]
    user_pois = set(interaction_matrix[user_idx].nonzero()[1])
    recommended_pois = {}

    for neighbor_idx in top_k:
        sim = sim_matrix[user_idx, neighbor_idx]
        for poi_idx in interaction_matrix[neighbor_idx].nonzero()[1]:
            if poi_idx not in user_pois:
                poi_id = list(poi_mapping.keys())[poi_idx]
                score = interaction_matrix[neighbor_idx, poi_idx]
                recommended_pois[poi_id] = recommended_pois.get(poi_id, 0) + sim * score

    return [poi for poi, _ in sorted(recommended_pois.items(), key=lambda x: x[1], reverse=True)[:n]]

if __name__ == "__main__":
    CITY = "London" # Modifica la ciudad según sea necesario
    VARS = ["temp", "precip", "snow", "windspeed", "visibility", "uvindex"]  # Variables climáticas a considerar
    file_path_weather = f"../../data/FoursquareProcessed/{CITY}_checkins_weather.txt"
    checkins_weather = pd.read_csv(file_path_weather, sep="\t",
                                   names=["user_id", "poi_id", "temp", "precip", "snow", "windspeed", "visibility", "uvindex"])

    interaction_matrix, user_map, poi_map = knn_weather_masked_matrix(checkins_weather, VARS)

    checkins_agg = pd.read_csv(f"../../data/FoursquareProcessed/{CITY}_checkins_agg.txt", sep="\t",
                               names=["user_id", "poi_id", "score"])
    train_data, test_data = train_test_split(checkins_agg, test_size=0.3, random_state=42)
    _, test_data = train_test_split(test_data, test_size=2/3, random_state=42)

    results = evaluate_recommendations(test_data, train_data, cosine_climate_knn_recommendations,
                                       interaction_matrix, user_map, poi_map)
    print("KNN Coseno + Clima:", results)
