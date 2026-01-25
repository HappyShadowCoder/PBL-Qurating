from flask import Flask, request, render_template_string
import torch
import json
import os
import csv
from io import StringIO
from transformers import DistilBertTokenizerFast
from train_qurater import QualityModel

app = Flask(__name__)

MODEL_NAME = "distilbert-base-uncased"
MODEL_PATH = "models/qurater_fast.pth"
MAX_LEN = 256

# ---------- Load model once ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
model = QualityModel().to(device)

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
else:
    print("WARNING: Model not found. Train model first.")


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>QuRater Interactive Quality Scoring</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 40px;
            background-color: #0f1117;
            color: #e6e6e6;
        }
        h1, h2, h3 { color: #ffffff; }
        p, li { color: #cccccc; }

        .upload-box {
            padding: 20px;
            border: 1px solid #333;
            margin-bottom: 20px;
            background-color: #161b22;
            border-radius: 6px;
        }

        .error {
            background-color: #2f0f0f;
            color: #ffb4b4;
            padding: 10px;
            border: 1px solid #ff6b6b;
            border-radius: 6px;
            margin-bottom: 20px;
        }

        button {
            background-color: #238636;
            color: white;
            border: none;
            padding: 10px 16px;
            border-radius: 6px;
            cursor: pointer;
            font-weight: bold;
        }
        button:hover { background-color: #2ea043; }

        table {
            border-collapse: collapse;
            width: 100%;
            margin-top: 20px;
            background-color: #161b22;
        }

        th, td {
            border: 1px solid #30363d;
            padding: 8px;
            vertical-align: top;
        }

        th {
            background-color: #21262d;
            color: #ffffff;
        }

        tr.high { background-color: #0f2f1f; }
        tr.low  { background-color: #2f0f0f; }

        .score {
            font-weight: bold;
            color: #58a6ff;
        }
    </style>
</head>
<body>

<h1>QuRater Interactive Quality Scoring</h1>
<p>Criterion: <b>educational_value</b></p>

{% if error %}
<div class="error">
    <b>Error:</b> {{ error }}
</div>
{% endif %}

<div class="upload-box">
    <h3>Upload File (JSON / CSV / TXT)</h3>

    <ul>
        <li><b>JSON</b>: [{"text": "..."}]</li>
        <li><b>CSV</b>: must contain column named <code>text</code></li>
        <li><b>TXT</b>: one text per line</li>
    </ul>

    <form method="POST" enctype="multipart/form-data">
        <input type="file" name="file" accept=".json,.csv,.txt" required>
        <br><br>
        <button type="submit">Upload & Score</button>
    </form>
</div>

{% if results %}
<h2>Scored Results (sorted by quality)</h2>
<table>
    <tr>
        <th>#</th>
        <th>Quality Score</th>
        <th>Text</th>
    </tr>
    {% for r in results %}
    <tr class="{{ 'high' if r.quality_score > 1 else 'low' if r.quality_score < 0 else '' }}">
        <td>{{ loop.index }}</td>
        <td class="score">{{ "%.3f"|format(r.quality_score) }}</td>
        <td>{{ r.text }}</td>
    </tr>
    {% endfor %}
</table>
{% endif %}

</body>
</html>
"""


def extract_texts_from_file(file):
    filename = file.filename.lower()

    # -------- JSON --------
    if filename.endswith(".json"):
        data = json.load(file)
        return [item.get("text", "") for item in data if "text" in item]

    # -------- CSV --------
    elif filename.endswith(".csv"):
        content = file.read().decode("utf-8")
        reader = csv.DictReader(StringIO(content))

        if "text" not in reader.fieldnames:
            raise ValueError("CSV must contain a column named 'text'")

        return [row["text"] for row in reader if row.get("text")]

    # -------- TXT --------
    elif filename.endswith(".txt"):
        content = file.read().decode("utf-8")
        lines = content.splitlines()
        return [line.strip() for line in lines if line.strip()]

    else:
        raise ValueError("Unsupported file format. Use JSON, CSV, or TXT.")


@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    error = None

    if request.method == "POST":
        print("request.files keys:", request.files.keys())

        file = request.files.get("file")

        if file is None or file.filename == "":
            error = "No file uploaded. Please choose a file and try again."
        else:
            try:
                texts = extract_texts_from_file(file)
                if not texts:
                    error = "No valid 'text' entries found in file."
                else:
                    with torch.no_grad():
                        for t in texts:
                            tokens = tokenizer(
                                t,
                                return_tensors="pt",
                                truncation=True,
                                padding=True,
                                max_length=MAX_LEN
                            ).to(device)

                            score = model(
                                tokens["input_ids"],
                                tokens["attention_mask"]
                            ).item()

                            results.append({
                                "text": t,
                                "quality_score": float(score)
                            })

                    results.sort(key=lambda x: x["quality_score"], reverse=True)

            except Exception as e:
                error = str(e)

    return render_template_string(HTML_TEMPLATE, results=results, error=error)


if __name__ == "__main__":
    app.run(debug=True)