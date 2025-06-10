import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from eval_mf import evaluate_recommendations

class ClimatePytorchFM(nn.Module):
    def __init__(self, embedding_dim=16, climate_cols=None, epochs=20, lr=0.01, verbose=True):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.epochs = epochs
        self.lr = lr
        self.verbose = verbose
        self.climate_cols = climate_cols or ["temp"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.user_encoder = LabelEncoder()
        self.poi_encoder = LabelEncoder()
        self.scaler = MinMaxScaler()

    def prepare(self, df):
        df = df.copy()

        for col in self.climate_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mean())

        df["user_id_enc"] = self.user_encoder.fit_transform(df["user_id"])
        df["poi_id_enc"] = self.poi_encoder.fit_transform(df["poi_id"])
        df[self.climate_cols] = self.scaler.fit_transform(df[self.climate_cols])

        self.n_users = df["user_id_enc"].nunique()
        self.n_items = df["poi_id_enc"].nunique()
        self.n_climate = len(self.climate_cols)
        return df

    def build_model(self):
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim).to(self.device)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim).to(self.device)
        self.user_bias = nn.Embedding(self.n_users, 1).to(self.device)
        self.item_bias = nn.Embedding(self.n_items, 1).to(self.device)
        self.climate_linear = nn.Linear(self.n_climate, 1).to(self.device)
        self.global_bias = nn.Parameter(torch.zeros(1).to(self.device))

    def forward(self, user_ids, item_ids, climate_feats):
        user_vec = self.user_embedding(user_ids)
        item_vec = self.item_embedding(item_ids)
        dot = (user_vec * item_vec).sum(dim=1, keepdim=True)
        bias = self.user_bias(user_ids) + self.item_bias(item_ids) + self.climate_linear(climate_feats)
        return self.global_bias + dot + bias

    def fit(self, df):
        df = self.prepare(df)
        self.build_model()

        user_tensor = torch.tensor(df["user_id_enc"].values, dtype=torch.long)
        item_tensor = torch.tensor(df["poi_id_enc"].values, dtype=torch.long)
        climate_tensor = torch.tensor(df[self.climate_cols].values.astype(np.float32))
        score_tensor = torch.tensor(df["score"].values.astype(np.float32)).unsqueeze(1)

        dataset = TensorDataset(user_tensor, item_tensor, climate_tensor, score_tensor)
        dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)

        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)
        criterion = nn.MSELoss()

        self.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for u, i, c, s in dataloader:
                u, i, c, s = u.to(self.device), i.to(self.device), c.to(self.device), s.to(self.device)

                preds = self.forward(u, i, c)
                loss = criterion(preds, s)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item() * u.size(0)

            if self.verbose:
                avg_loss = total_loss / len(df)
                print(f"Epoch {epoch+1}/{self.epochs} - Loss: {avg_loss:.4f}")

        self.train_df = df

    def recommend_with_context(self, user_id, climate_df, n=10):
        if user_id not in self.user_encoder.classes_:
            return []

        user_idx = self.user_encoder.transform([user_id])[0]
        user_tensor = torch.tensor([user_idx], dtype=torch.long, device=self.device)

        all_items = self.train_df["poi_id"].unique()
        item_idxs = self.poi_encoder.transform(all_items)
        item_tensor = torch.tensor(item_idxs, dtype=torch.long, device=self.device)

        user_climate = climate_df[climate_df["user_id"] == user_id]
        if not user_climate.empty:
            climate_values = user_climate[self.climate_cols].mean().values.astype(np.float32)
        else:
            climate_values = self.train_df[self.climate_cols].mean().values.astype(np.float32)

        climate_values = np.tile(climate_values, (len(item_idxs), 1))
        climate_tensor = torch.tensor(climate_values, dtype=torch.float32, device=self.device)

        self.eval()
        with torch.no_grad():
            scores = self.forward(user_tensor.repeat(len(item_idxs)), item_tensor, climate_tensor).flatten()

        top_indices = scores.topk(n).indices.cpu().numpy()
        return [all_items[i] for i in top_indices]

if __name__ == "__main__":
    CITY = "Tokyo"
    climate_vars = ["temp", "precip", "snow", "windspeed", "visibility", "uvindex"]

    agg = pd.read_csv(f"{CITY}_checkins_agg.txt", sep="\t", names=["user_id", "poi_id", "score"])
    weather = pd.read_csv(f"{CITY}_checkins_weather.txt", sep="\t", names=["user_id", "poi_id"] + climate_vars)

    df = weather.merge(agg, on=["user_id", "poi_id"], how="left")
    df["score"] = df["score"].fillna(1)
    train, test = train_test_split(df, test_size=0.3, random_state=42)
    val, test = train_test_split(test, test_size=2/3, random_state=42)

    train["score"] = np.log1p(train["score"])

    model = ClimatePytorchFM(embedding_dim=64, climate_cols=climate_vars, epochs=25, lr=0.008, verbose=True)
    model.fit(train)

    def mf_wrapper(user_id, model, context_df):
        return model.recommend_with_context(user_id, context_df, n=10)

    results = evaluate_recommendations(test, train, mf_wrapper, model, test)
    print(f"Resultados:\n{results}")
