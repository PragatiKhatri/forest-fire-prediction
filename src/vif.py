"""
vif.py
------
Small, dependency-light Variance Inflation Factor calculator.
VIF_i = 1 / (1 - R^2_i), where R^2_i comes from regressing feature i against
all the other features. A VIF above ~10 signals problematic multicollinearity.
(statsmodels.stats.outliers_influence.variance_inflation_factor does the same
thing -- this avoids requiring it as a dependency.)
"""
import pandas as pd
from sklearn.linear_model import LinearRegression


def compute_vif(df: pd.DataFrame) -> pd.DataFrame:
    scores = []
    for col in df.columns:
        X = df.drop(columns=[col]).values
        y = df[col].values
        r2 = LinearRegression().fit(X, y).score(X, y)
        vif = float("inf") if r2 >= 1 else 1 / (1 - r2)
        scores.append({"feature": col, "VIF": round(vif, 2)})
    return pd.DataFrame(scores).sort_values("VIF", ascending=False).reset_index(drop=True)
