# Measures how strong the current trend is using MA50 and MA200.
# Columns: trend_strength

import pandas as pd
from pathlib import Path

INPUT_DIR = Path("data/features")
OUTPUT_DIR = Path("data/features")

def calculate_trend_strength(df):
    df["trend_strength"] = (df["ma50"] - df["ma200"]) / df["ma200"]
    return df

def run_trend_strength():
    for file in INPUT_DIR.glob("*.parquet"):
        df = pd.read_parquet(file)
        df = calculate_trend_strength(df)
        df.to_parquet(OUTPUT_DIR / file.name)
        print(f"Saved: {file.name}")

if __name__ == "__main__":
    run_trend_strength()
