import torch
import json
from transformers import DistilBertTokenizerFast
from train_qurater import QualityModel
from tqdm import tqdm
import os
import time


MODEL_NAME = "distilbert-base-uncased"
MODEL_PATH = "models/qurater_fast.pth"
MAX_LEN = 256
BATCH_SIZE = 16


def score_corpus():

    os.makedirs("results", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    if not os.path.exists(MODEL_PATH):
        print("Trained model not found. Train first.")
        return

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    model = QualityModel().to(device)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    corpus_path = "data/demo_corpus.json"

    if not os.path.exists(corpus_path):
        print(f"Corpus not found: {corpus_path}")
        return

    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    texts = [item["text"] for item in corpus]
    print("Loaded texts:", len(texts))

    scores = []
    start = time.time()

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Scoring"):

            batch = texts[i:i + BATCH_SIZE]

            tokens = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_LEN,
                return_tensors="pt"
            ).to(device)

            batch_scores = model(
                tokens["input_ids"],
                tokens["attention_mask"]
            )

            scores.extend(batch_scores.cpu().tolist())

    print(f"Scoring time: {(time.time() - start)/60:.1f} min")

    data = [
        {
            "text": t,
            "quality_score": float(s),
            "criterion": "educational_value"
        }
        for t, s in zip(texts, scores)
    ]

    data.sort(key=lambda x: x["quality_score"], reverse=True)

    output_file = "results/scored_corpus.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    values = [d["quality_score"] for d in data]

    summary = {
        "criterion": "educational_value",
        "total": len(values),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
        "median": sorted(values)[len(values)//2]
    }

    with open("results/scoring_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nTop 3 high-quality texts:")
    for d in data[:3]:
        print(f"Score={d['quality_score']:.4f} | {d['text'][:200]}...")

    print("\nBottom 3 low-quality texts:")
    for d in data[-3:]:
        print(f"Score={d['quality_score']:.4f} | {d['text'][:200]}...")

    print("\nSaved:")
    print(" -", output_file)
    print(" - results/scoring_summary.json")


if __name__ == "__main__":
    score_corpus()