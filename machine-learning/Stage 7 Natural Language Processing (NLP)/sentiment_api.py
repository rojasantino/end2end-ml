#!/usr/bin/env python3
"""
Amazon Review Sentiment API
Stage 7 — NLP Project  |  ML Roadmap

Usage:
    pip install flask scikit-learn pandas numpy
    python sentiment_api.py

Endpoints:
    POST /predict  — JSON body: {"review": "your review text"}
    GET  /health   — health check
"""
import re, pickle
from flask import Flask, request, jsonify

app = Flask(__name__)

# ── Load model ──────────────────────────────────────────────────────────────
with open('sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)

STOP = set("""i me my myself we our ours ourselves you your yours yourself
    yourselves he him his himself she her hers herself it its itself they them
    their theirs themselves what which who whom this that these those am is are
    was were be been being have has had having do does did doing a an the and
    but if or because as until while of at by for with about between into
    through during before after above below to from up down in out on off over
    under again then once here there when where why how all both each few more
    most other some no not only same so than too very can will just should now
    also get got however""".split())

def clean(text):
    t = re.sub(r"[^a-zA-Z\s]", " ", str(text).lower())
    t = re.sub(r"\s+", " ", t).strip()
    return " ".join(w for w in t.split() if w not in STOP and len(w) > 2)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "review" not in data:
        return jsonify({"error": "Missing 'review' field in request body"}), 400

    raw   = data["review"]
    c     = clean(raw)
    pred  = int(model.predict([c])[0])
    label = "Positive" if pred == 1 else "Negative"

    try:
        proba = model.predict_proba([c])[0]
        conf  = float(round(max(proba) * 100, 2))
    except AttributeError:
        conf = None

    return jsonify({
        "sentiment":   label,
        "label_code":  pred,
        "confidence":  conf,
        "clean_text":  c,
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "SentimentClassifier"})


if __name__ == "__main__":
    print("🚀 Sentiment API running on http://localhost:5000")
    print("   POST /predict  — {'review': 'your text'}  →  {'sentiment': 'Positive'}")
    app.run(debug=True, port=5000)
