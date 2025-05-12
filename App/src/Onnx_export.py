import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_dim, num_layers, dropout):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out.squeeze()

# 1. Instantiate your LSTM model exactly as during training
input_size = 6       # e.g. open, high, low, close, volume, sentiment
hidden_dim = 64
num_layers = 2
dropout = 0.2

model = LSTMModel(input_size, hidden_dim, num_layers, dropout)
model.load_state_dict(torch.load("lstm_model.pth", map_location="cpu"))
model.eval()

batch_size = 1
seq_len    = 10
dummy_input_lstm = torch.randn(batch_size, seq_len, input_size, dtype=torch.float32)

torch.onnx.export(
    model,                         # the loaded LSTMModel
    dummy_input_lstm,              # example input
    "lstm_model.onnx",             # where to save
    export_params=True,            # store trained weights
    opset_version=13,              # ONNX version
    do_constant_folding=True,      # fold constants
    input_names=["input"],         # model inputs
    output_names=["output"],       # model outputs
    dynamic_axes={
        "input": {0: "batch_size", 1: "seq_len"},  # allow dynamic batch & sequence length
        "output": {0: "batch_size"}
    }
)
from onnxruntime.quantization import quantize_dynamic, QuantType

# Paths
model_fp32 = "lstm_model.onnx"
model_int8 = "lstm_model_int8.onnx"
model_fp16 = "lstm_model_fp16.onnx"

# INT8 quantization (weights only)
quantize_dynamic(
    model_fp32,
    model_int8,
    weight_type=QuantType.QInt8
)

# FP16 quantization (better for GPU inference)
quantize_dynamic(
    model_fp32,
    model_fp16,
    weight_type=QuantType.QUInt8,  # QUInt8 produces FP16 weights
)

