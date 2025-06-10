import numpy as np

def evaluate_recommendations(test_data, train_data, recommender_function):
    precision_list, recall_list, epc_list, recommended_sets = [], [], [], []
    item_popularity = train_data.groupby("poi_id")["score"].sum().to_dict()
    total_users = train_data["user_id"].nunique()

    for user_id in test_data["user_id"].unique():
        actual_pois = set(test_data[test_data["user_id"] == user_id]["poi_id"])
        train_seen_pois = set(train_data[train_data["user_id"] == user_id]["poi_id"])
        recommended_pois = set(recommender_function(user_id)) - train_seen_pois

        if not actual_pois or not recommended_pois:
            continue

        precision = len(recommended_pois & actual_pois) / len(recommended_pois)
        recall = len(recommended_pois & actual_pois) / len(actual_pois)
        user_epc_values = [1 - (item_popularity.get(poi, 0) / total_users) for poi in recommended_pois]
        epc = np.mean(user_epc_values) if user_epc_values else 0

        recommended_sets.append(recommended_pois)
        precision_list.append(precision)
        recall_list.append(recall)
        epc_list.append(epc)

    aggregate_diversity = len(set().union(*recommended_sets))

    return {
        "Precision": np.mean(precision_list),
        "Recall": np.mean(recall_list),
        "EPC": np.mean(epc_list),
        "Aggregate Diversity": aggregate_diversity
    }
