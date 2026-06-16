# Calculates 20-day rolling standard deviation of returns as a volatility measure.
# Columns: volatility

import pandas as pd
from pathlib import Path

ROOT       = Path(__file__).resolve().parents[3]
INPUT_DIR  = ROOT / "data/features"
OUTPUT_DIR = ROOT / "data/features"

# calculate volatility
def volatility(file_path):
    df = pd.read_parquet(file_path)
    df["volatility"] = df["returns"].rolling(20).std()
    return df

# loops overall files and save results
def run_volatility():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for file in INPUT_DIR.glob("*.parquet"):
        df = volatility(file)
        df.to_parquet(OUTPUT_DIR/file.name)
        print(f"Saved: {file.name}")

# Entry Point
if __name__ == "__main__":
    run_volatility()