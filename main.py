import spacy
import torch
import torch.nn as nn
import numpy as np
import re
from torchcrf import CRF
from torch.utils.data import Dataset, DataLoader

########################################
# CONFIG
########################################
EMB_DIM = 300
HIDDEN_DIM = 128
EPOCHS = 25
LR = 0.001

LABELS = [
    "O",
    "B-AMOUNT", "I-AMOUNT",
    "B-CURRENCY",
    "B-BENEFICIARY", "I-BENEFICIARY",
    "B-FROM_ACCOUNT", "I-FROM_ACCOUNT",
    "B-PAYMENT_TYPE"
]

label2id = {l: i for i, l in enumerate(LABELS)}
id2label = {i: l for l, i in label2id.items()}

########################################
# LOAD SPACY
########################################
nlp = spacy.load("en_core_web_md")

########################################
# RULE-BASED PRE-TAGS
########################################
def rule_tags(doc):
    tags = ["O"] * len(doc)
    for i, t in enumerate(doc):
        if re.match(r"^[₹$€]\d+", t.text):
            tags[i] = "B-AMOUNT"
        elif t.text.upper() in ["UPI", "NEFT", "RTGS"]:
            tags[i] = "B-PAYMENT_TYPE"
        elif t.text in ["₹", "$", "€"]:
            tags[i] = "B-CURRENCY"
    return tags

########################################
# FEATURE EXTRACTION
########################################
def extract_features(doc, rules):
    feats = []
    for i, t in enumerate(doc):
        feats.append(
            np.concatenate([
                t.vector,                       # 300
                [t.is_upper],                   # 1
                [t.is_digit],                   # 1
                [len(t.text)],                  # 1
                [hash(t.pos_) % 50],            # 1
                [hash(rules[i]) % 20]           # 1
            ])
        )
    return np.array(feats, dtype=np.float32)

########################################
# DATASET
########################################
class FinanceDataset(Dataset):
    def __init__(self, samples):
        self.data = samples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, tags = self.data[idx]
        doc = nlp(text)
        rules = rule_tags(doc)
        X = extract_features(doc, rules)
        y = [label2id[t] for t in tags]
        mask = [1] * len(y)
        return torch.tensor(X), torch.tensor(y), torch.tensor(mask)

########################################
# MODEL
########################################
class BiLSTMCRF(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_labels):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            bidirectional=True,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim * 2, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, x, tags, mask):
        out, _ = self.lstm(x)
        emissions = self.fc(out)
        loss = -self.crf(emissions, tags, mask)
        return loss

    def predict(self, x, mask):
        out, _ = self.lstm(x)
        emissions = self.fc(out)
        return self.crf.decode(emissions, mask)

########################################
# SAMPLE TRAINING DATA
########################################
train_data = [
    (
        "Transferred ₹5000 from HDFC account to Rahul via UPI",
        ["O", "B-AMOUNT", "O", "B-FROM_ACCOUNT", "I-FROM_ACCOUNT",
         "O", "B-BENEFICIARY", "O", "B-PAYMENT_TYPE"]
    ),
    (
        "Sent $200 to Amit using NEFT",
        ["O", "B-AMOUNT", "O", "B-BENEFICIARY", "O", "B-PAYMENT_TYPE"]
    )
]

########################################
# TRAIN
########################################
dataset = FinanceDataset(train_data)
loader = DataLoader(dataset, batch_size=1, shuffle=True)

model = BiLSTMCRF(input_dim=305, hidden_dim=HIDDEN_DIM, num_labels=len(LABELS))
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    total_loss = 0
    for X, y, mask in loader:
        optimizer.zero_grad()
        loss = model(X, y, mask)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f}")

########################################
# INFERENCE
########################################
def predict(text):
    doc = nlp(text)
    rules = rule_tags(doc)
    X = extract_features(doc, rules)
    X = torch.tensor(X).unsqueeze(0)
    mask = torch.tensor([[1] * len(doc)])
    preds = model.predict(X, mask)[0]

    for token, p in zip(doc, preds):
        print(f"{token.text:10s} -> {id2label[p]}")

print("\n--- PREDICTION ---")
predict("Paid ₹1200 to Suresh from SBI account via UPI")
