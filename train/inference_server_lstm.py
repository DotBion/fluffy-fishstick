from flask import Flask, request, jsonify
import torch
import numpy as np
from lstm_train_pytorch import LSTMModel  # importing from your training file

# Initialize Flask app
app = Flask(__name__)

# === Model configuration (MUST match your training setup) ===
input_size = 6       # open, high, low, close, volume, sentiment
hidden_dim = 64
num_layers = 2
dropout = 0.2

# === Load trained model ===
model = LSTMModel(input_size=input_size, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
model.load_state_dict(torch.load("lstm_model.pth", map_location=torch.device("cpu")))
model.eval()

@app.route('/')
def home():
    return "PyTorch LSTM Inference Server is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Expecting JSON: {"input": [[[...], [...], ...]]}
        data = request.get_json(force=True)

        if 'input' not in data:
            return jsonify({'error': "Missing 'input' key in JSON payload"}), 400

        input_data = np.array(data['input'], dtype=np.float32)

        # Validate input shape
        if len(input_data.shape) != 3 or input_data.shape[2] != input_size:
            return jsonify({
                'error': f"Input must be shape (batch, seq_len, {input_size}), got {input_data.shape}"
            }), 400

        input_tensor = torch.tensor(input_data)

        with torch.no_grad():
            predictions = model(input_tensor).numpy()

        return jsonify({'predictions': predictions.tolist()}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
