# Detects market regime using MA50 and MA200, then joins it onto each stock's feature file.
# Columns: regime, ma50, ma200, regime_encoded

import pandas as pd
from pathlib import Path
import numpy as np

INPUT_DIR = Path("data/features")
OUTPUT_DIR = Path("data/features")

# Detect the current market regime using MA50 & MA200 (Trend Following)
def detect_regime():
    df = pd.read_parquet(INPUT_DIR / "^SPX.parquet")
    df = df.set_index("Date")
    df["ma200"] = df["Close"].rolling(window=200).mean()
    df["ma50"] = df["Close"].rolling(window=50).mean()

    conditions = [
        (df["Close"] > df["ma200"] * 1.01) & (df["ma50"] > df["ma200"]),
        (df["Close"] < df["ma200"] * 0.99) & (df["ma50"] < df["ma200"]),
        (df["Close"] < df["ma200"] * 0.99) & (df["ma50"] > df["ma200"]),
        (df["Close"] > df["ma200"] * 1.01) & (df["ma50"] < df["ma200"])
    ]
    choices = ["bull", "bear", "transition_down", "transition_up"]

    df["regime"] = np.select(conditions, choices, default="sideways")
    df.loc[df["ma200"].isna(), "regime"] = "unknown"

    return df[["regime", "ma200", "ma50"]]

# Loop over all files and join regime
def run_trend_context():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    regime_df = detect_regime()
    for file in OUTPUT_DIR.glob("*.parquet"):
        df = pd.read_parquet(file)
        df = df.drop(columns=[c for c in ["regime", "ma200", "ma50", "regime_encoded"] if c in df.columns])
        df = df.set_index("Date").join(regime_df, how="left").reset_index()
        regime_map = {"bull": 1, "bear": -1, "transition_up": 0.5, "transition_down": -0.5, "sideways": 0, "unknown": 0}
        df["regime_encoded"] = df["regime"].map(regime_map)
        df.to_parquet(OUTPUT_DIR / file.name)
        print(f"Saved: {file.name}")

if __name__ == "__main__":
    run_trend_context()
