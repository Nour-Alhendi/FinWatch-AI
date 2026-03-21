# Measures price direction and strength over 5 and 10 days.
# Columns: momentum_5, momentum_10

import pandas as pd
from pathlib import Path

INPUT_DIR = Path("data/features")
OUTPUT_DIR = Path("data/features")

def calculate_momentum(df):
    df["momentum_5"] = df["Close"] - df["Close"].shift(5)
    df["momentum_10"] = df["Close"] - df["Close"].shift(10)
    return df

def run_momentum():
    for file in INPUT_DIR.glob("*.parquet"):
        df = pd.read_parquet(file)
        df = calculate_momentum(df)
        df.to_parquet(OUTPUT_DIR / file.name)
        print(f"Saved: {file.name}")

if __name__ == "__main__":
    run_momentum()
