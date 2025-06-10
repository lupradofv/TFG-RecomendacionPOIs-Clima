import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from eval_baselines import evaluate_recommendations  # suponiendo que lo extraes

def get_popularity_recommendations(train_data, seen_pois, n=10):
    popular_pois = train_data.groupby('poi_id')['score'].sum().sort_values(ascending=False)
    recommendations = [poi for poi in popular_pois.index if poi not in seen_pois]
    return recommendations[:n]

def popularity_recommender_eval(user_id, train_data, n=10):
    seen_pois = set(train_data[train_data["user_id"] == user_id]["poi_id"])
    return get_popularity_recommendations(train_data, seen_pois, n)

if __name__ == "__main__":
    CITY = 'Tokyo' # Modifica la ciudad según sea necesario
    FILE_PATH = f'../../data/FoursquareProcessed/{CITY}_checkins_agg.txt'
    checkins = pd.read_csv(FILE_PATH, sep='\t', names=['user_id', 'poi_id', 'score'])

    train_data, test_data = train_test_split(checkins, test_size=0.3, random_state=42)
    _, test_data = train_test_split(test_data, test_size=2/3, random_state=42)

    results = evaluate_recommendations(test_data, train_data,
                                       lambda uid: popularity_recommender_eval(uid, train_data))
    print("Resultados Recomendador por Popularidad:", results)
