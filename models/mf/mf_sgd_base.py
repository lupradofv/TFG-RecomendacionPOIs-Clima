import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from eval_mf import evaluate_recommendations

class MatrixFactorizationSGD:
    def __init__(self, n_factors=20, n_epochs=20, learning_rate=0.01, reg=0.05, verbose=False):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.reg = reg
        self.verbose = verbose

    def fit(self, interactions):
        users = interactions["user_id"].unique()
        items = interactions["poi_id"].unique()
        
        self.user_to_idx = {u: i for i, u in enumerate(users)}
        self.item_to_idx = {i: j for j, i in enumerate(items)}
        self.idx_to_item = {j: i for i, j in self.item_to_idx.items()}

        n_users = len(users)
        n_items = len(items)

        self.P = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.Q = np.random.normal(0, 0.1, (n_items, self.n_factors))

        data = interactions[["user_id", "poi_id", "score"]].values

        for epoch in range(self.n_epochs):
            np.random.shuffle(data)
            for user_id, item_id, score in data:
                u = self.user_to_idx[user_id]
                i = self.item_to_idx[item_id]

                pred = self.P[u, :].dot(self.Q[i, :])
                err = score - pred

                self.P[u, :] += self.learning_rate * (err * self.Q[i, :] - self.reg * self.P[u, :])
                self.Q[i, :] += self.learning_rate * (err * self.P[u, :] - self.reg * self.Q[i, :])

            if self.verbose:
                rmse = np.sqrt(np.mean([(s - self.predict(user_id, item_id)) ** 2 for user_id, item_id, s in data]))
                print(f"Epoch {epoch+1}/{self.n_epochs} - RMSE: {rmse:.4f}")

    def predict(self, user_id, item_id):
        if user_id not in self.user_to_idx or item_id not in self.item_to_idx:
            return 0
        u = self.user_to_idx[user_id]
        i = self.item_to_idx[item_id]
        return self.P[u, :].dot(self.Q[i, :])

    def recommend(self, user_id, train_df, n=10):
        if user_id not in self.user_to_idx:
            return []

        u = self.user_to_idx[user_id]
        seen_pois = set(train_df[train_df["user_id"] == user_id]["poi_id"])
        scores = {
            self.idx_to_item[i]: self.P[u, :].dot(self.Q[i, :])
            for i in range(self.Q.shape[0])
            if self.idx_to_item[i] not in seen_pois
        }

        return [poi for poi, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]]


if __name__ == "__main__":
    CITY = "Tokyo" # Modifica la ciudad según sea necesario

    checkins = pd.read_csv(f"../../data/FoursquareProcessed/{CITY}_checkins_agg.txt", sep="\t", names=["user_id", "poi_id", "score"])
    train_df, test_df = train_test_split(checkins, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(test_df, test_size=2/3, random_state=42)

    mf = MatrixFactorizationSGD(
        n_factors=20,
        n_epochs=25,
        learning_rate=0.01,
        reg=1e-4,
        verbose=False
    )
    mf.fit(train_df)

    def mf_wrapper(user_id, model, train_data):
        return model.recommend(user_id, train_data, n=10)

    result = evaluate_recommendations(test_df, train_df, mf_wrapper, mf, train_df)
    print(f"Resultado: {result}")

