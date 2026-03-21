# Captures recent return history for time-aware models like XGBoost.
# Columns: return_lag_1, return_lag_2, return_lag_3

import pandas as pd
from pathlib import Path

INPUT_DIR = Path("data/features")
OUTPUT_DIR = Path("data/features")

def calculate_lags(df):
    for lag in [1, 2, 3]:
        df[f"return_lag_{lag}"] = df["returns"].shift(lag)
    return df

def run_return_lags():
    for file in INPUT_DIR.glob("*.parquet"):
        df = pd.read_parquet(file)
        df = calculate_lags(df)
        df.to_parquet(OUTPUT_DIR / file.name)
        print(f"Saved: {file.name}")

if __name__ == "__main__":
    run_return_lags()