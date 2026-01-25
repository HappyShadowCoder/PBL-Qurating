import csv
from datasets import load_dataset
from tqdm import tqdm
import os

os.makedirs("data", exist_ok=True)

test_dict = load_dataset("princeton-nlp/QuRating-GPT3.5-Judgments-Test")

with open("data/qurating_test.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["text_a", "text_b", "p_b_greater_a", "domain"])

    for domain, ds in test_dict.items():
        print(f"Processing domain: {domain}, rows={len(ds)}")

        for ex in tqdm(ds, desc=f"Exporting {domain}"):
            text_a = ex["texts"][0]
            text_b = ex["texts"][1]

            # Educational Value criterion
            p_b_greater_a = ex["educational_value_average"][0][1]

            if p_b_greater_a < 0:
                continue

            writer.writerow([
                text_a,
                text_b,
                float(p_b_greater_a),
                domain
            ])

print("Saved data/qurating_test.csv")