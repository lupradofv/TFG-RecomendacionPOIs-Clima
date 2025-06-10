import numpy as np

def evaluate_recommendations(test_data, train_data, recommender_function, *args):
    precision_list, recall_list, epc_list, recommended_sets = [], [], [], []

    item_popularity = train_data.groupby("poi_id")["score"].sum().to_dict()
    total_users = train_data["user_id"].nunique()

    for user_id in test_data["user_id"].unique():
        actual_pois = set(test_data[test_data["user_id"] == user_id]["poi_id"])
        if not actual_pois:
            continue

        try:
            recommended_pois = set(recommender_function(user_id, *args))
        except Exception:
            continue

        if not recommended_pois:
            continue

        precision = len(recommended_pois & actual_pois) / len(recommended_pois)
        recall = len(recommended_pois & actual_pois) / len(actual_pois)
        epc = np.mean([
            1 - (item_popularity.get(poi, 0) / total_users)
            for poi in recommended_pois
        ]) if recommended_pois else 0

        precision_list.append(precision)
        recall_list.append(recall)
        epc_list.append(epc)
        recommended_sets.append(recommended_pois)

    return {
        "Precision": np.mean(precision_list),
        "Recall": np.mean(recall_list),
        "EPC": np.mean(epc_list),
        "Aggregate Diversity": len(set().union(*recommended_sets))
    }
