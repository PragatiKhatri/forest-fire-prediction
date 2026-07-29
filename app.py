"""
app.py
------
Flask REST API for the Forest Fire Weather Index (FWI) predictor.

Endpoints:
  GET  /            -> HTML form for manual testing
  POST /predict      -> form submission, renders result on the page
  POST /api/predict  -> JSON in, JSON out (for programmatic / curl use)
  GET  /health       -> simple liveness check

Run locally:
    pip install -r requirements.txt
    python app.py
  then open http://127.0.0.1:5000
"""
import pickle

import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

with open("model/fwi_model.pkl", "rb") as f:
    bundle = pickle.load(f)

MODEL = bundle["model"]
SCALER = bundle["scaler"]
FEATURES = bundle["features"]   # e.g. ['RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'DC', 'ISI']


def risk_label(fwi: float) -> str:
    if fwi < 5:
        return "Low"
    if fwi < 12:
        return "Moderate"
    if fwi < 22:
        return "High"
    return "Extreme"


def predict_fwi(payload: dict) -> dict:
    missing = [f for f in FEATURES if f not in payload]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    row = pd.DataFrame([[float(payload[f]) for f in FEATURES]], columns=FEATURES)
    scaled = SCALER.transform(row)
    fwi = float(MODEL.predict(scaled)[0])
    fwi = max(0.0, round(fwi, 2))
    return {"fwi": fwi, "risk": risk_label(fwi)}


@app.route("/")
def index():
    return render_template("index.html", features=FEATURES, result=None)


@app.route("/predict", methods=["POST"])
def predict_form():
    try:
        payload = {f: request.form.get(f) for f in FEATURES}
        result = predict_fwi(payload)
        return render_template("index.html", features=FEATURES, result=result, values=payload)
    except (ValueError, TypeError) as e:
        return render_template("index.html", features=FEATURES, result=None, error=str(e))


@app.route("/api/predict", methods=["POST"])
def predict_api():
    try:
        payload = request.get_json(force=True)
        result = predict_fwi(payload)
        return jsonify(result)
    except (ValueError, TypeError) as e:
        return jsonify({"error": str(e)}), 400


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": bundle.get("model_name", "unknown")})


if __name__ == "__main__":
    app.run(debug=True)
