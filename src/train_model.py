"""
train_model.py
--------------
End-to-end pipeline for the Forest Fire (FWI) Prediction project:

1. Load + clean the Algerian Forest Fires-style dataset
2. Feature engineering (drop IDs, encode categoricals)
3. Multicollinearity check via VIF, drop redundant fire-code columns
4. Standardize inputs
5. Benchmark Linear Regression, Lasso, and LassoCV (5-fold CV)
6. Pick the best model on held-out R^2, pickle it + the scaler + feature list

Run:
    python src/train_model.py
Produces:
    model/fwi_model.pkl        (dict: {"model", "scaler", "features"})
    model/metrics.json         (benchmark results, for the README/report)
"""
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LassoCV, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

from vif import compute_vif

RANDOM_STATE = 42


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # impute the handful of missing weather readings with the column median
    for col in ["Temperature", "RH", "Rain"]:
        df[col] = df[col].fillna(df[col].median())
    df = df.drop(columns=["day", "month", "year", "Classes", "Region"])
    return df


def select_features_by_vif(X: pd.DataFrame, threshold: float = 10.0) -> list:
    """Iteratively drop the highest-VIF column until everything is under the
    threshold. FFMC/DMC/DC feed directly into ISI/BUI, so they're expected to
    drop out here."""
    cols = list(X.columns)
    while True:
        vif_table = compute_vif(X[cols])
        worst = vif_table.iloc[0]
        if worst["VIF"] < threshold or len(cols) <= 2:
            break
        cols.remove(worst["feature"])
    return cols


def main():
    df = load_and_clean("data/algerian_forest_fires.csv")

    y = df["FWI"]
    X_full = df.drop(columns=["FWI"])

    kept_features = select_features_by_vif(X_full)
    print("Features kept after VIF filtering:", kept_features)
    X = X_full[kept_features]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    candidates = {
        "LinearRegression": LinearRegression(),
        "Lasso(alpha=0.1)": Lasso(alpha=0.1, random_state=RANDOM_STATE),
        "LassoCV": LassoCV(cv=kf, random_state=RANDOM_STATE, max_iter=5000),
    }

    results = {}
    for name, model in candidates.items():
        cv_scores = cross_val_score(model, X_train_s, y_train, cv=kf, scoring="r2")
        model.fit(X_train_s, y_train)
        preds = model.predict(X_test_s)

        results[name] = {
            "cv_r2_mean": round(float(np.mean(cv_scores)), 4),
            "cv_r2_std": round(float(np.std(cv_scores)), 4),
            "test_r2": round(float(r2_score(y_test, preds)), 4),
            "test_rmse": round(float(np.sqrt(mean_squared_error(y_test, preds))), 3),
            "test_mae": round(float(mean_absolute_error(y_test, preds)), 3),
        }
        if name == "LassoCV":
            results[name]["chosen_alpha"] = round(float(model.alpha_), 4)

    print(pd.DataFrame(results).T)

    best_name = max(results, key=lambda n: results[n]["test_r2"])
    best_model = candidates[best_name]
    print(f"\nBest model: {best_name} (test R^2 = {results[best_name]['test_r2']})")

    import pickle
    with open("model/fwi_model.pkl", "wb") as f:
        pickle.dump(
            {"model": best_model, "scaler": scaler, "features": kept_features, "model_name": best_name},
            f,
        )

    with open("model/metrics.json", "w") as f:
        json.dump({"results": results, "best_model": best_name, "features": kept_features}, f, indent=2)

    print("Saved model/fwi_model.pkl and model/metrics.json")


if __name__ == "__main__":
    main()
