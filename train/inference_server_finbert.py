from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Initialize Flask app
app = Flask(__name__)

# Load the fine-tuned FinBERT model and tokenizer
model_dir = 'finbert_model'  # folder containing the saved model
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
model.eval()  # Set model to evaluation mode

# Define label mapping
label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

@app.route('/')
def home():
    return "FinBERT Inference Server is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)

        if 'input' not in data:
            return jsonify({'error': "Missing 'input' key in JSON payload"}), 400

        texts = data['input']  # expecting list of strings

        if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
            return jsonify({'error': "'input' should be a list of strings."}), 400

        # Tokenize
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            predicted_classes = torch.argmax(probs, dim=1).tolist()
            predicted_labels = [label_map[i] for i in predicted_classes]
            scores = probs.tolist()

        return jsonify({
            'predictions': predicted_labels,
            'probabilities': scores
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run on port 5000 for local testing
    app.run(host='0.0.0.0', port=5000, debug=True)
