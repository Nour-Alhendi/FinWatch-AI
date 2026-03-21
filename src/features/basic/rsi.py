# Calculates the Relative Strength Index (RSI) over 14 days.
# Values above 70 indicate overbought, below 30 oversold.
# Columns: rsi

import pandas as pd
from pathlib import Path

INPUT_DIR = Path("data/features")
OUTPUT_DIR = Path("data/features")
WINDOW = 14

def calculate_rsi(df):
    delta = df["Close"].diff()          # tägliche Veränderung
    gain = delta.clip(lower=0)          # nur positive Tage
    loss = -delta.clip(upper=0)         # nur negative Tage (positiv gemacht)
    avg_gain = gain.rolling(WINDOW).mean()
    avg_loss = loss.rolling(WINDOW).mean()
    rs = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))
    return df

def run_rsi():
    for file in INPUT_DIR.glob("*.parquet"):
        df = pd.read_parquet(file)
        df = calculate_rsi(df)
        df.to_parquet(OUTPUT_DIR / file.name)
        print(F"Saved: {file.name}")

if __name__ == "__main__":
    run_rsi()