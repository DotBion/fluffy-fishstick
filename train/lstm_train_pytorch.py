
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# === Configurable Hyperparameters ===
config = {
    "epochs": 20,
    "batch_size": 16,
    "lr": 1e-3,
    "seq_length": 10,
    "hidden_dim": 64,
    "num_layers": 2,
    "dropout": 0.2,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# === Data Loading ===
data_path = os.getenv("DATA_CSV_PATH", "data_2018.csv")
df = pd.read_csv(data_path)
# df = df.rename(columns={
#     'Unnamed: 0': 'date',
#     'Average Score': 'daily_avg_sentiment_score'
# })
df = df.sort_values(by='date')

feature_cols = ["open", "high", "low", "close", "volume", "daily_avg_sentiment_score"]
data = df[feature_cols].values

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# === Create sequences ===
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length][3])  # predicting 'close' price
    return np.array(X), np.array(y)

X, y = create_sequences(data_scaled, config["seq_length"])
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

# === Dataset Class ===
class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(StockDataset(X_train, y_train), batch_size=config["batch_size"], shuffle=True)
val_loader = DataLoader(StockDataset(X_val, y_val), batch_size=config["batch_size"], shuffle=False)

# === Model ===
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_dim, num_layers, dropout):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out.squeeze()

model = LSTMModel(input_size=len(feature_cols),
                  hidden_dim=config["hidden_dim"],
                  num_layers=config["num_layers"],
                  dropout=config["dropout"]).to(config["device"])

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

# === Training ===
for epoch in range(config["epochs"]):
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(config["device"]), y_batch.to(config["device"])
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * X_batch.size(0)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(config["device"]), y_batch.to(config["device"])
            output = model(X_batch)
            loss = criterion(output, y_batch)
            val_loss += loss.item() * X_batch.size(0)

    print(f"Epoch {epoch+1}/{config['epochs']} - Train Loss: {train_loss/len(train_loader.dataset):.4f} - Val Loss: {val_loss/len(val_loader.dataset):.4f}")

# === Save the model ===
torch.save(model.state_dict(), "lstm_model.pth")
