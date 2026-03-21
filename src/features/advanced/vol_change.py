# Detects sudden changes in volatility by comparing short-term vs long-term vol.
# Columns: vol_5, vol_20, vol_change

import pandas as pd
from pathlib import Path

INPUT_DIR = Path("data/features")
OUTPUT_DIR = Path("data/features")

def calculate_vol_change(df):
    df["vol_5"] = df["returns"].rolling(5).std()
    df["vol_20"] = df["returns"].rolling(20).std()
    df["vol_change"] = df["vol_5"] / df["vol_20"]
    return df

def run_vol_change():
    for file in INPUT_DIR.glob("*.parquet"):
        df = pd.read_parquet(file)
        df = calculate_vol_change(df)
        df.to_parquet(OUTPUT_DIR / file.name)
        print(f"Saved: {file.name}")

if __name__ == "__main__":
    run_vol_change()
