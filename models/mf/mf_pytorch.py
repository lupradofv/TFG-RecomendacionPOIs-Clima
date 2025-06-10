import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from eval_mf import evaluate_recommendations

class SimplePytorchMF(nn.Module):
    def __init__(self, embedding_dim=32, epochs=40, lr=0.01, weight_decay=0.0001, batch_size=512, verbose=True):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.verbose = verbose
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.user_encoder = LabelEncoder()
        self.poi_encoder = LabelEncoder()

    def prepare(self, df):
        df["user_id_enc"] = self.user_encoder.fit_transform(df["user_id"])
        df["poi_id_enc"] = self.poi_encoder.fit_transform(df["poi_id"])
        self.n_users = df["user_id_enc"].nunique()
        self.n_items = df["poi_id_enc"].nunique()
        return df

    def build_model(self):
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim).to(self.device)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim).to(self.device)
        self.user_bias = nn.Embedding(self.n_users, 1).to(self.device)
        self.item_bias = nn.Embedding(self.n_items, 1).to(self.device)
        self.global_bias = nn.Parameter(torch.zeros(1).to(self.device))

    def forward(self, user_ids, item_ids):
        user_vec = self.user_embedding(user_ids)
        item_vec = self.item_embedding(item_ids)
        dot = (user_vec * item_vec).sum(dim=1, keepdim=True)
        bias = self.user_bias(user_ids) + self.item_bias(item_ids)
        return self.global_bias + dot + bias

    def fit(self, df):
        df = self.prepare(df)
        self.build_model()

        users = torch.tensor(df["user_id_enc"].values, dtype=torch.long)
        items = torch.tensor(df["poi_id_enc"].values, dtype=torch.long)
        scores = torch.tensor(df["score"].values.astype(np.float32)).unsqueeze(1)

        dataset = TensorDataset(users, items, scores)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.MSELoss()

        self.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for u, i, s in loader:
                u, i, s = u.to(self.device), i.to(self.device), s.to(self.device)
                preds = self.forward(u, i)
                loss = loss_fn(preds, s)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if self.verbose:
                print(f"Epoch {epoch+1}: Loss={total_loss:.4f}")

    def recommend(self, user_id, n=10):
        if user_id not in self.user_encoder.classes_:
            return []
        user_idx = self.user_encoder.transform([user_id])[0]
        all_items = self.poi_encoder.transform(self.poi_encoder.classes_)
        user_tensor = torch.tensor([user_idx] * len(all_items)).to(self.device)
        item_tensor = torch.tensor(all_items).to(self.device)

        self.eval()
        with torch.no_grad():
            scores = self.forward(user_tensor, item_tensor).flatten()
        top_idxs = scores.topk(n).indices.cpu().numpy()
        return [self.poi_encoder.inverse_transform([all_items[i]])[0] for i in top_idxs]

if __name__ == "__main__":
    CITY = 'London'  # Modifica la ciudad según sea necesario

    df = pd.read_csv(f"data/FoursquareProcessed/{CITY}_checkins_agg.txt", sep="\t", names=["user_id", "poi_id", "score"])
    df["score"] = df["score"].fillna(1)
    train, test = train_test_split(df, test_size=0.3, random_state=42)
    val, test = train_test_split(test, test_size=2/3, random_state=42)

    model = SimplePytorchMF(
        embedding_dim=32,
        epochs=40,
        lr=0.01,
        weight_decay=0.0001,
        batch_size=512,
        verbose=True
    )
    model.fit(train)

    def mf_wrapper(user_id, model):
        return model.recommend(user_id, n=10)

    metrics = evaluate_recommendations(test, train, mf_wrapper, model)

    print(f"Resultados: {metrics}")