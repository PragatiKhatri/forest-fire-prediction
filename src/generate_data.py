"""
generate_data.py
-----------------
Builds a synthetic dataset that mirrors the structure and relationships of the
UCI "Algerian Forest Fires" dataset (Bejaia + Sidi-Bel Abbes regions, Jun-Sep
2012). Real weather station data isn't available in this environment, so the
values here are generated from the same fire-weather physics used by the
Canadian FWI system: FFMC/DMC/DC are driven by temperature, humidity, wind and
rain, ISI comes from FFMC + wind, BUI comes from DMC + DC, and FWI is a
nonlinear function of ISI + BUI. That's what gives the strong R^2 seen in
notebooks/model_training -- it mirrors the real dataset's structure.

Run:
    python src/generate_data.py
Produces:
    data/algerian_forest_fires.csv
"""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
REGIONS = ["Bejaia", "Sidi-Bel Abbes"]
MONTHS = [6, 7, 8, 9]           # June - September
DAYS_PER_MONTH = 30
N_PER_REGION = 122               # matches the real dataset's size


def make_region(region_name: str) -> pd.DataFrame:
    rows = []
    for i in range(N_PER_REGION):
        month = MONTHS[i % len(MONTHS)]
        day = (i % DAYS_PER_MONTH) + 1

        temperature = np.clip(RNG.normal(32, 5), 18, 42)
        rh = np.clip(RNG.normal(55, 15), 20, 95)          # relative humidity %
        ws = np.clip(RNG.normal(15, 5), 5, 30)             # wind speed km/h
        rain = max(0, RNG.exponential(0.7) - 0.3)          # mm, mostly dry

        # --- fire weather codes (simplified physical relationships) ---
        ffmc = np.clip(89 - rain * 8 + (rh - 55) * -0.15 + RNG.normal(0, 2), 28, 96)
        dmc = np.clip(15 + (temperature - 25) * 1.6 - rain * 3 + RNG.normal(0, 3), 1, 65)
        dc = np.clip(80 + (temperature - 25) * 6 - rain * 5 + RNG.normal(0, 8), 5, 220)

        isi = np.clip(0.11 * ffmc + 0.18 * ws + RNG.normal(0, 1.3), 0.2, 20)
        bui = np.clip(0.6 * dmc + 0.15 * dc + RNG.normal(0, 2), 0.5, 120)

        fwi = np.clip(0.32 * isi * (bui ** 0.55) - 1.5 + RNG.normal(0, 1.1), 0, 33)
        fire = "fire" if fwi > 23 else "not fire"

        rows.append({
            "day": day, "month": month, "year": 2012,
            "Temperature": round(temperature, 1),
            "RH": round(rh, 1),
            "Ws": round(ws, 1),
            "Rain": round(rain, 1),
            "FFMC": round(ffmc, 1),
            "DMC": round(dmc, 1),
            "DC": round(dc, 1),
            "ISI": round(isi, 1),
            "BUI": round(bui, 1),
            "FWI": round(fwi, 1),
            "Classes": fire,
            "Region": region_name,
        })
    return pd.DataFrame(rows)


def main():
    df = pd.concat([make_region(r) for r in REGIONS], ignore_index=True)
    # sprinkle a few missing values, mirroring the messiness of the real file
    for col in ["Temperature", "RH", "Rain"]:
        idx = RNG.choice(df.index, size=3, replace=False)
        df.loc[idx, col] = np.nan
    df.to_csv("data/algerian_forest_fires.csv", index=False)
    print(f"Wrote data/algerian_forest_fires.csv with {len(df)} rows")


if __name__ == "__main__":
    main()
