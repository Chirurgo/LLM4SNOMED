from flask import Flask, render_template, request
import json
import os
from dotenv import dotenv_values
app = Flask(__name__, template_folder="Evaluate")
base_dir = os.path.dirname(os.path.abspath(__file__))
config = dotenv_values(os.path.join(base_dir, ".env"))
# ─── LOAD TEST DATA ────────────────────────────────────────────────────────────
# Adjust the path to point at your test JSON
TEST_JSON_PATH = config.get("TEST_DATA_PATH")
if not TEST_JSON_PATH:
    raise RuntimeError("TEST_DATA_PATH is not defined in .env")
if not os.path.isabs(TEST_JSON_PATH):
    TEST_JSON_PATH = os.path.join(base_dir, TEST_JSON_PATH)
with open(TEST_JSON_PATH, encoding="utf-8") as f:
    test_data = json.load(f)

# Build a mapping from combined_id ("{id}_{type}") → document dict
# This ensures uniqueness even if documents share the same numeric "id" but have different "type"
doc_map = {}
for doc in test_data:
    combined_id = f"{doc['id']}_{doc['type']}"
    doc_map[combined_id] = doc

# Available combined IDs for dropdown
doc_ids = sorted(doc_map.keys())


# ─── LOAD PREDICTION MODELS ────────────────────────────────────────────────────
PREDICTION_DIR = os.path.join(base_dir, "DATA", "predictions")
if not os.path.isdir(PREDICTION_DIR):
    raise RuntimeError(f"Prediction directory not found: {PREDICTION_DIR}")
available_models = [
    filename[:-5]  # strip off ".json"
    for filename in os.listdir(PREDICTION_DIR)
    if filename.endswith(".json")
]


# ─── FLASK ROUTE ────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET", "POST"])
def compare():
    # On POST, retrieve the selected combined_id and model name
    selected_combined = request.form.get("doc_id") if request.method == "POST" else None
    selected_model = request.form.get("model") if request.method == "POST" else None

    gold_text = ""
    gold_entities = []
    pred_entities = []

    if selected_combined and selected_model:
        # 1) Fetch the gold document by its combined key
        gold = doc_map.get(selected_combined)
        if gold:
            gold_text = gold.get("text", "")
            gold_entities = gold.get("entity", [])

            # 2) Load the predictions JSON for the selected model
            pred_path = os.path.join(PREDICTION_DIR, f"{selected_model}.json")
            with open(pred_path, encoding="utf-8") as pf:
                predictions = json.load(pf)

            # 3) Find the index of the gold doc in test_data by matching id & type
            for idx, doc in enumerate(test_data):
                if doc["id"] == gold["id"] and doc["type"] == gold["type"]:
                    if idx < len(predictions):
                        pred_entities = predictions[idx]
                    else:
                        pred_entities = []
                    break

    return render_template(
        "compare.html",
        doc_ids=doc_ids,
        model_names=available_models,
        selected_id=selected_combined,
        selected_model=selected_model,
        gold_text=gold_text,
        gold_entities=gold_entities,
        pred_entities=pred_entities,
    )


# ─── MAIN ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug_mode)
