import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict
from eval_knn import evaluate_recommendations

def contextual_similarity_by_poi(df, climate_cols, climate_threshold=0.3):
    scaler = MinMaxScaler()
    df[climate_cols] = scaler.fit_transform(df[climate_cols])

    poi_groups = df.groupby('poi_id')
    similarity_matrix = defaultdict(float)
    count_matrix = defaultdict(int)

    for poi, group in poi_groups:
        if group['user_id'].nunique() < 2:
            continue
        users = group['user_id'].values
        climates = group[climate_cols].values

        for i, j in combinations(range(len(users)), 2):
            u1, u2 = users[i], users[j]
            dist = np.linalg.norm(climates[i] - climates[j])
            if dist < climate_threshold:
                sim = 1 - dist
                similarity_matrix[(u1, u2)] += sim
                similarity_matrix[(u2, u1)] += sim
                count_matrix[(u1, u2)] += 1
                count_matrix[(u2, u1)] += 1

    user_ids = df['user_id'].unique()
    sim_df = pd.DataFrame(0, index=user_ids, columns=user_ids, dtype=float)

    for (u1, u2), sim_sum in similarity_matrix.items():
        count = count_matrix[(u1, u2)]
        sim_df.loc[u1, u2] = sim_sum / count

    return sim_df

def knn_contextual_recommendations(user_id, df, user_similarity_matrix, n_neighbors=10, n_recommendations=10):
    if user_id not in user_similarity_matrix.index:
        return []

    user_pois = set(df[df['user_id'] == user_id]['poi_id'])
    similar_users = user_similarity_matrix.loc[user_id].sort_values(ascending=False).head(n_neighbors).index

    poi_scores = {}
    for neighbor in similar_users:
        sim = user_similarity_matrix.loc[user_id, neighbor]
        for _, row in df[df['user_id'] == neighbor].iterrows():
            poi = row['poi_id']
            if poi in user_pois:
                continue
            score = row.get('score', 1)
            poi_scores[poi] = poi_scores.get(poi, 0) + sim * score

    sorted_pois = sorted(poi_scores.items(), key=lambda x: x[1], reverse=True)
    return [poi for poi, _ in sorted_pois[:n_recommendations]]

def contextual_recommender_eval(user_id, df_checkins, sim_matrix, n=10):
    return knn_contextual_recommendations(user_id, df_checkins, sim_matrix, n_neighbors=10, n_recommendations=n)

if __name__ == "__main__":
    CITY = "London" # Modifica la ciudad según sea necesario
    VARS = ['temp', 'precip', 'snow', 'windspeed'] # Variables climáticas a considerar
    
    checkins_agg = pd.read_csv(f"../../data/FoursquareProcessed/{CITY}_checkins_agg.txt", sep="\t", names=["user_id", "poi_id", "score"])
    checkins_weather = pd.read_csv(f"../../data/FoursquareProcessed/{CITY}_checkins_weather.txt", sep="\t",
                                   names=['user_id', 'poi_id', 'temp','precip', 'snow', 'windspeed', 'visibility', 'uvindex'])
    checkins_weather = checkins_weather.merge(checkins_agg, on=['user_id', 'poi_id'], how='left').fillna(0)

    train_data, test_data = train_test_split(checkins_agg, test_size=0.3, random_state=42)
    _, test_data = train_test_split(test_data, test_size=2/3, random_state=42)
    train_weather, _ = train_test_split(checkins_weather, test_size=0.3, random_state=42)

    sim_matrix = contextual_similarity_by_poi(train_weather, VARS, climate_threshold=0.3)

    def recommender_wrapper(user_id, n=10):
        return contextual_recommender_eval(user_id, train_data, sim_matrix, n=n)

    results = evaluate_recommendations(test_data, train_data, recommender_wrapper)
    print(f"KNN Contextual con {VARS} →", results)
