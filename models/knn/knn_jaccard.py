import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
from eval_knn import evaluate_recommendations

def knn_matrix(checkins):
    user_mapping = {user: idx for idx, user in enumerate(checkins["user_id"].unique())}
    poi_mapping = {poi: idx for idx, poi in enumerate(checkins["poi_id"].unique())}
    checkins["user_idx"] = checkins["user_id"].map(user_mapping)
    checkins["poi_idx"] = checkins["poi_id"].map(poi_mapping)
    interaction_matrix = csr_matrix((checkins["score"], (checkins["user_idx"], checkins["poi_idx"])),
                                    shape=(len(user_mapping), len(poi_mapping)))
    return interaction_matrix, user_mapping, poi_mapping

def jaquard_knn_recommendations(user_id, interaction_matrix, user_mapping, poi_mapping, k=10, n=10):
    if user_id not in user_mapping:
        return []
    user_idx = user_mapping[user_id]
    bin_matrix = interaction_matrix.copy()
    bin_matrix.data = np.ones_like(bin_matrix.data)
    user_vector = bin_matrix[user_idx]
    intersections = user_vector.dot(bin_matrix.T).toarray().flatten()
    user_sum = user_vector.sum()
    other_sums = bin_matrix.sum(axis=1).A1
    unions = user_sum + other_sums - intersections
    jaccard_scores = np.divide(intersections, unions, out=np.zeros_like(intersections, dtype=float), where=unions != 0)
    jaccard_scores[user_idx] = 0
    top_k = np.argsort(-jaccard_scores)[:k]
    user_pois = set(interaction_matrix[user_idx].nonzero()[1])
    recommended_pois = {}
    for neighbor_idx in top_k:
        neighbor_pois = interaction_matrix[neighbor_idx].nonzero()[1]
        for poi_idx in neighbor_pois:
            if poi_idx not in user_pois:
                poi_id = list(poi_mapping.keys())[poi_idx]
                recommended_pois[poi_id] = recommended_pois.get(poi_id, 0) + jaccard_scores[neighbor_idx]
    sorted_pois = sorted(recommended_pois.items(), key=lambda x: x[1], reverse=True)
    return [poi for poi, _ in sorted_pois[:n]]

if __name__ == "__main__":
    CITY = 'New York' # Modifica la ciudad según sea necesario
    FILE_PATH = f'../../data/FoursquareProcessed/{CITY}_checkins_agg.txt'
    checkins = pd.read_csv(FILE_PATH, sep='\t', names=['user_id', 'poi_id', 'score'])

    train_data, test_data = train_test_split(checkins, test_size=0.3, random_state=42)
    val_data, test_data = train_test_split(test_data, test_size=2/3, random_state=42)

    interaction_matrix, user_mapping, poi_mapping = knn_matrix(train_data)
    results = evaluate_recommendations(test_data, train_data, jaquard_knn_recommendations,
                                       interaction_matrix, user_mapping, poi_mapping, 10, 10)
    print('Resultados KNN Jaccard:', results)