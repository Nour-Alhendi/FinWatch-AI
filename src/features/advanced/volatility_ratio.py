# Measures how volatile a stock is relative to the market (SPX).
# Columns: volatility_ratio

import pandas as pd
from pathlib import Path

INPUT_DIR = Path("data/features")
OUTPUT_DIR = Path("data/features")
SPX_FILE = Path("data/detection") / "^SPX.parquet"

def load_spx_volatility():
    df = pd.read_parquet(SPX_FILE)
    df = df.set_index("Date")
    return df[["volatility"]].rename(columns={"volatility": "spx_volatility"})

def calculate_volatility_ratio(df, spx_vol):
    df = df.join(spx_vol, how="left")
    df["volatility_ratio"] = df["volatility"] / df["spx_volatility"]
    return df

def run_volatility_ratio():
    spx_vol = load_spx_volatility()
    for file in INPUT_DIR.glob("*.parquet"):
        if file.stem == "^SPX":
            continue
        df = pd.read_parquet(file)
        if "Date" in df.columns:
            df = df.set_index("Date")
        df = calculate_volatility_ratio(df, spx_vol)
        df = df.reset_index()
        df.to_parquet(OUTPUT_DIR / file.name)
        print(f"Saved: {file.name}")

if __name__ == "__main__":
    run_volatility_ratio()
