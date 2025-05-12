from flask import Flask, request, jsonify
import numpy as np
import onnxruntime as ort
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# Optional: for Triton inference
# import tritonclient.http as httpclient

app = Flask(__name__)

# === ONNX Runtime Session with FP16 quantization ===
# Assumes you've exported and quantized your model to FP16 ONNX format:
#   torch.onnx.export(..., "lstm_model_fp16.onnx", opset_version=12)
#   Optional: use onnxruntime-tools to quantize to FP16:
#     from onnxruntime_tools import optimizer
#     optimizer.optimize_model("lstm_model.onnx", model_type='bert', use_gpu=True, num_heads=8, hidden_size=64)

sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 4
# Enable graph optimizations
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

# Providers: try GPU first, fallback to CPU
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
ort_session = ort.InferenceSession(
    "lstm_model_fp16.onnx",
    sess_options,
    providers=providers
)

# === Triton Client Example (uncomment to use Triton) ===
# triton_client = httpclient.InferenceServerClient(url="localhost:8000")
#
# def triton_infer(input_array):
#     inputs = [
#         httpclient.InferInput(
#             name="input",
#             shape=input_array.shape,
#             datatype="FP16"
#         )
#     ]
#     inputs[0].set_data_from_numpy(input_array.astype(np.float16))
#     results = triton_client.infer(model_name="lstm_model", inputs=inputs)
#     return results.as_numpy("output").astype(np.float32)

@app.route('/')
def home():
    return "ONNX-Optimized LSTM Server is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        payload = request.get_json(force=True)
        if 'input' not in payload:
            return jsonify({'error': "Missing 'input' key in JSON payload"}), 400

        data = np.array(payload['input'], dtype=np.float32)
        # Validate shape
        if data.ndim != 3 or data.shape[2] != 6:
            return jsonify({'error': f"Expected shape (batch, seq_len, 6), got {data.shape}"}), 400

        # Convert to FP16 for inference
        data_fp16 = data.astype(np.float16)

        # Local ONNX Runtime inference
        outputs = ort_session.run(None, {'input': data})
        preds = outputs[0].astype(np.float32).tolist()

        # Or use Triton:
        # preds = triton_infer(data)

        return jsonify({'predictions': preds}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/metrics')
def metrics():
    return Response(
        generate_latest(), 
        mimetype=CONTENT_TYPE_LATEST
    )

if __name__ == '__main__':
    # Use gunicorn + uvicorn for production:
    #   gunicorn -k uvicorn.workers.UvicornWorker optimized_server:app --bind 0.0.0.0:9090 --workers 4
    app.run(host='0.0.0.0', port=9090)
