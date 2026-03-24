# Calculates 20-day and 60-day rolling mean and standard deviation of returns.
# Columns: rolling_mean, rolling_std, rolling_mean_60, rolling_std_60

import pandas as pd
from pathlib import Path

INPUT_DIR =  Path("data/features")
OUTPUT_DIR = Path("data/features")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# calculate rolling stats
def rolling_stats(file_path):
    df = pd.read_parquet(file_path)
    df["rolling_mean"] = df["returns"].rolling(20).mean()
    df["rolling_std"]  = df["returns"].rolling(20).std()
    return df

# loops overall files and save results
def run_rolling_stats():
    for file in INPUT_DIR.glob("*.parquet"):
        df = rolling_stats(file)
        df.to_parquet(OUTPUT_DIR/file.name)
        print(f"Saved {file.name}")

# Entry Point 
if __name__ == "__main__":
    run_rolling_stats()




