
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

# === Config ===
MODEL_NAME = 'yiyanghkust/finbert-tone'
CSV_PATH = 'tweets_2018_limited.csv'
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# === Custom Dataset ===
class TweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

# === Load and process data ===
df = pd.read_csv(CSV_PATH)

def score_to_label(score):
    if score > 0.05:
        return 2  # positive
    elif score < -0.05:
        return 0  # negative
    else:
        return 1  # neutral

df['label'] = df['va_sentiment_score'].apply(score_to_label)

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['body'], df['label'], test_size=0.2, random_state=42
)

# === Tokenizer and Datasets ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
train_dataset = TweetDataset(train_texts.tolist(), train_labels.tolist(), tokenizer)
val_dataset = TweetDataset(val_texts.tolist(), val_labels.tolist(), tokenizer)

# === Dataloaders ===
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# === Model ===
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
model.to(DEVICE)
optimizer = AdamW(model.parameters(), lr=LR)

# === Training Loop ===
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f'Training Epoch {epoch+1}'):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} Training Loss: {total_loss / len(train_loader):.4f}")

    # Evaluation
    model.eval()
    preds, true_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            preds.extend(torch.argmax(logits, axis=1).cpu().numpy())
            true_labels.extend(batch['labels'].cpu().numpy())

    acc = accuracy_score(true_labels, preds)
    f1 = f1_score(true_labels, preds, average='weighted')
    print(f"Epoch {epoch+1} Validation Accuracy: {acc:.4f} | F1 Score: {f1:.4f}")

# === Save model ===
model.save_pretrained('./finbert_model')
tokenizer.save_pretrained('./finbert_model')
