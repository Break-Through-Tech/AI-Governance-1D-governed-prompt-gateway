import torch
import pandas as pd
import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, roc_auc_score
from huggingface_hub import login

load_dotenv()
dataset_token = os.getenv("dataset_token")
if not dataset_token:
    raise RuntimeError("dataset_token is not set in the environment")
login(token=dataset_token)

# 1. Load Model & Tokenizer
model_id = "meta-llama/Llama-Prompt-Guard-2-86M"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

# 2. Prepare Your DataFrame
# Map 'Type' to binary target: 0 for benign, 1 for toxic/malicious
benign_df = pd.read_csv('./data/behaviors_benign.csv')
benign_df['Type'] = 'Benign'
harmful_df = pd.read_csv('./data/behaviors_harmful.csv')
harmful_df['Type'] = 'Harmful'
concat_df = pd.concat([benign_df, harmful_df]) # combining both datasets

concat_df["binary_label"] = concat_df["Type"].str.lower().map({"benign": 0, "harmful": 1})

# Filter out any rows with unmapped/missing labels if necessary
eval_df = concat_df.dropna(subset=["binary_label"]).copy()
eval_df["binary_label"] = eval_df["binary_label"].astype(int)

# 3. Scoring Function
def predict_probabilities(texts, max_length=512):
    inputs = tokenizer(
        texts, 
        padding=True, 
        truncation=True, 
        max_length=max_length, 
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        
    return probs[:, 1].cpu().numpy()  # Return score for 'MALICIOUS'

# 4. Run Evaluation in Batches
batch_size = 16
malicious_scores = []
texts = eval_df["Goal"].tolist()

for i in range(0, len(texts), batch_size):
    batch_texts = texts[i:i+batch_size]
    scores = predict_probabilities(batch_texts)
    malicious_scores.extend(scores)

eval_df["malicious_score"] = malicious_scores

# 5. Evaluate Metrics
auc = roc_auc_score(eval_df["binary_label"], eval_df["malicious_score"])
print(f"ROC-AUC Score: {auc:.4f}")
for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
    preds = (eval_df["malicious_score"] > threshold).astype(int)
    print(f"\n--- EVALUATION AT THRESHOLD: {threshold} ---")
    print(classification_report(eval_df["binary_label"], preds, target_names=["BENIGN", "HARMFUL"]))