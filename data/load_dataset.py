import csv
from datasets import load_dataset
from tqdm import tqdm
import os

os.makedirs("data" , exist_ok=True)

train_ds = load_dataset("princeton-nlp/QuRating-GPT3.5-Judgments")["train"]

with open("data/qurating_train.csv" , "w" , newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["text_a", "text_b", "p_b_greater_a"])
    writer.writerow(["text_a" , "text_b" , "p_b_greater_a"])

    for ex in tqdm(train_ds , desc="Exporting train"):
        text_a = ex["texts"][0]
        text_b = ex["texts"][1]

        p_b_greater_a = ex["educational_value_average"][0][1]

        if p_b_greater_a < 0:
            continue

        writer.writerow([text_a, text_b, float(p_b_greater_a)])

print("Saved data/qurating_train.csv")

