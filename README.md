# 🔥 Forest Fire Weather Index (FWI) Prediction

A regression pipeline that predicts the **Fire Weather Index** from meteorological
readings, served two ways: a **Flask REST API** and an interactive **Streamlit
demo**. Modeled on the structure of the UCI Algerian Forest Fires dataset.

**Live demo:** _add your Streamlit Cloud link here after deploying (see below)_

## What's inside

```
forest-fire-prediction/
├── data/
│   └── algerian_forest_fires.csv     # weather + fire-code dataset
├── src/
│   ├── generate_data.py              # builds the dataset
│   ├── vif.py                        # multicollinearity (VIF) helper
│   └── train_model.py                # cleaning -> VIF -> training -> pickle
├── model/
│   ├── fwi_model.pkl                  # trained model + scaler + feature list
│   └── metrics.json                   # benchmark results
├── templates/index.html               # Flask form UI
├── app.py                             # Flask REST API
├── streamlit_app.py                   # Streamlit live demo
└── requirements.txt
```

## How the model was built

1. **Data**: Temperature, relative humidity, wind speed, rain, and the Canadian
   FWI fire-danger codes (FFMC, DMC, DC, ISI, BUI) across two regions, June–Sept.
2. **Cleaning**: missing weather readings imputed with the column median.
3. **Multicollinearity check**: a custom VIF function (`src/vif.py`) flags and
   removes redundant fire-code columns that feed directly into one another
   (e.g. BUI is derived from DMC + DC, so it's dropped once those are kept).
4. **Modeling**: Linear Regression, Lasso, and LassoCV benchmarked with 5-fold
   cross-validation; the best model on held-out R² is kept.
5. **Serialization**: the winning model, its `StandardScaler`, and the final
   feature list are pickled together into `model/fwi_model.pkl`.

Run it yourself:
```bash
pip install -r requirements.txt
python src/generate_data.py     # (re)builds data/algerian_forest_fires.csv
python src/train_model.py       # trains + saves model/fwi_model.pkl
```

## Run the Flask API locally

```bash
python app.py
```
Then open **http://127.0.0.1:5000** for the form, or call it directly:

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"RH":45,"Ws":15,"Rain":0,"FFMC":88,"DMC":30,"DC":150,"ISI":12}'
```

## Run the Streamlit demo locally

```bash
streamlit run streamlit_app.py
```

## Push this to your GitHub

```bash
cd forest-fire-prediction
git init
git add .
git commit -m "Forest fire FWI prediction: Flask API + Streamlit demo"
git branch -M main
git remote add origin https://github.com/PragatiKhatri/forest-fire-prediction.git
git push -u origin main
```
(Create the empty repo first at github.com/new, name it `forest-fire-prediction`,
**don't** initialize it with a README so the push above doesn't conflict.)

## Get a live link (Streamlit Community Cloud — free, ~2 minutes)

1. Push the repo to GitHub (above).
2. Go to **share.streamlit.io** and sign in with GitHub.
3. Click **New app**, pick this repo, branch `main`, and set the main file
   path to `streamlit_app.py`.
4. Click **Deploy**. You'll get a public URL like
   `https://<your-app-name>.streamlit.app` — that's your live project link.
5. Paste that URL into your resume/portfolio.

## Notes

- The dataset here is synthetically generated to match the structure and
  fire-weather relationships of the real Algerian Forest Fires dataset
  (`src/generate_data.py` documents exactly how). Swap in the real CSV from
  UCI/Kaggle and rerun `train_model.py` if you want to train on the original data.
- Not intended for real-world fire-risk decisions.
