import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.pytorch
import os

# Define constants
SEQ_LENGTH = 3
BATCH_SIZE = 64
NUM_EPOCHS = 1000
INPUT_SIZE = 4
HIDDEN_SIZE = 32
NUM_LAYERS = 2
OUTPUT_SIZE = 4

# Load dataset
df = pd.read_csv('nifty.csv')
df['Date'] = pd.to_datetime(df['Date '])
data = df[['Open ', 'High ', 'Low ', 'Close ']].values

# Scale data
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# Create sequences
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        sequence = data[i:i+seq_length+1]
        sequences.append(sequence)
    return np.array(sequences)

X = create_sequences(data_scaled, SEQ_LENGTH)
y = X[:, -1]
X = X[:, :-1]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Define the LSTM model
class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(StockLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Initialize the model, criterion, and optimizer
model = StockLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Create data loaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# MLflow Experiment setup
mlflow.set_experiment("stock-price-prediction-lstm")

# Start logging with MLflow
with mlflow.start_run():
    mlflow.log_param("num_epochs", NUM_EPOCHS)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("learning_rate", 0.001)

    # Training loop
    for epoch in range(NUM_EPOCHS):
        epoch_losses = []
        for batch_X, batch_y in train_loader:
            output = model(batch_X)
            loss = criterion(output, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        
        avg_epoch_loss = np.mean(epoch_losses)
        mlflow.log_metric("train_loss", avg_epoch_loss, step=epoch)

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_epoch_loss:.4f}')
    
    # Save model with MLflow
    mlflow.pytorch.log_model(model, "model")

    # Evaluate the model on test data
    model.eval()
    predictions = []
    targets = []
    for batch_X, batch_y in test_loader:
        with torch.no_grad():
            output = model(batch_X)
            predictions.append(output.numpy())
            targets.append(batch_y.numpy())
    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)
    mse = np.mean((predictions - targets) ** 2)
    mlflow.log_metric("test_mse", mse)

    print(f'Mean Squared Error on Test Data: {mse:.4f}')
