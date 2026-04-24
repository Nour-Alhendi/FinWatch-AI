# Calculates the Relative Strength Index (RSI) over 14 days.
# Values above 70 indicate overbought, below 30 oversold.
# Columns: rsi

import pandas as pd
from pathlib import Path

INPUT_DIR = Path("data/features")
OUTPUT_DIR = Path("data/features")
WINDOW = 14

def calculate_rsi(df):
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # Wilder's smoothing: alpha = 1/WINDOW (same as EWM with adjust=False)
    avg_gain = gain.ewm(alpha=1/WINDOW, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/WINDOW, adjust=False).mean()
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