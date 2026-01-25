import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertModel, DistilBertTokenizerFast
import pandas as pd
from tqdm import tqdm
import os
import time

MODEL_NAME = "distilbert-base-uncased"

MAX_SAMPLES = 30000     # MacBook-safe subset
MAX_LEN = 256
BATCH_SIZE = 4
EPOCHS = 1
LR = 2e-5

SAVE_PATH = "models/qurater_fast.pth"
os.makedirs("models", exist_ok=True)

class QualityModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(MODEL_NAME)
        self.score_layer = torch.nn.Linear(768, 1)

    def forward(self, ids, mask):
        output = self.bert(input_ids=ids, attention_mask=mask)
        cls_vec = output.last_hidden_state[:, 0]
        return self.score_layer(cls_vec).squeeze(-1)


class TextPairs(Dataset):
    def __init__(self, file_path):
        data = pd.read_csv(file_path)

        if len(data) > MAX_SAMPLES:
            data = data.sample(MAX_SAMPLES, random_state=42)

        self.data = data.reset_index(drop=True)
        print(f"Loaded {len(self.data)} pairs")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return (
            row["text_a"],
            row["text_b"],
            float(row["p_b_greater_a"])
        )

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    model = QualityModel().to(device)

    dataset = TextPairs("data/qurating_train.csv")
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    model.train()
    start_time = time.time()

    for epoch in range(EPOCHS):
        total_loss = 0
        bar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for text1, text2, p in bar:
            tokens1 = tokenizer(
                list(text1),
                padding=True,
                truncation=True,
                max_length=MAX_LEN,
                return_tensors="pt"
            ).to(device)

            tokens2 = tokenizer(
                list(text2),
                padding=True,
                truncation=True,
                max_length=MAX_LEN,
                return_tensors="pt"
            ).to(device)

            p = torch.tensor(p, dtype=torch.float32).to(device)

            score1 = model(tokens1["input_ids"], tokens1["attention_mask"])
            score2 = model(tokens2["input_ids"], tokens2["attention_mask"])

            diff = score2 - score1
            loss = -(
                p * torch.log(torch.sigmoid(diff) + 1e-8) +
                (1 - p) * torch.log(torch.sigmoid(-diff) + 1e-8)
            ).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Pairwise accuracy (for monitoring)
            with torch.no_grad():
                preds = (score2 > score1).float()
                acc = (preds == (p > 0.5).float()).float().mean().item()
                bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.3f}")

        print(f"Epoch {epoch+1} avg loss: {total_loss/len(loader):.4f}")

    print(f"Training time: {(time.time() - start_time)/60:.1f} min")
    torch.save(model.state_dict(), SAVE_PATH)
    print("Saved model:", SAVE_PATH)

if __name__ == "__main__":
    train()