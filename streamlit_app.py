"""
streamlit_app.py
-----------------
Interactive live demo of the Fire Weather Index model. Deploy this file on
Streamlit Community Cloud (share.streamlit.io) for a public "live" link --
see README.md for the 2-minute setup.

Run locally:
    pip install -r requirements.txt
    streamlit run streamlit_app.py
"""
import pickle

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Forest Fire Weather Index Predictor", page_icon="🔥", layout="centered")

with open("model/fwi_model.pkl", "rb") as f:
    bundle = pickle.load(f)

MODEL = bundle["model"]
SCALER = bundle["scaler"]
FEATURES = bundle["features"]
MODEL_NAME = bundle.get("model_name", "model")

DEFAULTS = {
    "RH": 45.0, "Ws": 15.0, "Rain": 0.0,
    "FFMC": 85.0, "DMC": 25.0, "DC": 120.0, "ISI": 10.0,
}
RANGES = {
    "RH": (10, 100), "Ws": (0, 40), "Rain": (0, 15),
    "FFMC": (0, 100), "DMC": (0, 80), "DC": (0, 250), "ISI": (0, 25),
}

st.title("🔥 Fire Weather Index Predictor")
st.caption(
    f"Regression model ({MODEL_NAME}) trained on Algerian-forest-fire-style "
    "weather data. Adjust the readings below to estimate today's FWI."
)

st.divider()
cols = st.columns(2)
inputs = {}
for i, feat in enumerate(FEATURES):
    lo, hi = RANGES.get(feat, (0, 100))
    with cols[i % 2]:
        inputs[feat] = st.slider(feat, float(lo), float(hi), float(DEFAULTS.get(feat, (lo + hi) / 2)))

row = pd.DataFrame([[inputs[f] for f in FEATURES]], columns=FEATURES)
scaled = SCALER.transform(row)
fwi = max(0.0, round(float(MODEL.predict(scaled)[0]), 2))

if fwi < 5:
    risk, color = "Low", "green"
elif fwi < 12:
    risk, color = "Moderate", "orange"
elif fwi < 22:
    risk, color = "High", "red"
else:
    risk, color = "Extreme", "red"

st.divider()
m1, m2 = st.columns(2)
m1.metric("Predicted FWI", fwi)
m2.markdown(f"**Risk level:** :{color}[{risk}]")

st.progress(min(fwi / 33, 1.0))

with st.expander("What is FWI?"):
    st.write(
        "The Fire Weather Index (FWI) is a numeric rating of fire intensity "
        "potential used by fire-danger rating systems worldwide. It's derived "
        "from earlier fire-weather codes (FFMC, DMC, DC, ISI, BUI) that "
        "capture fuel moisture and wind-driven spread potential."
    )

st.caption("Synthetic training data for demo purposes — not for operational fire-risk decisions.")
