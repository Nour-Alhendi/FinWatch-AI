# Volume trend feature — compares short-term vs long-term average volume.
# volume_trend > 1 → volume increasing, < 1 → volume decreasing
# Columns: volume_ma_5, volume_ma_20, volume_trend

import pandas as pd
from pathlib import Path

INPUT_DIR = Path("data/features")
OUTPUT_DIR = Path("data/features")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


def run_volume_trend():
    for file in INPUT_DIR.glob("*.parquet"):
        df = pd.read_parquet(file)

        df["volume_ma_5"] = df["Volume"].rolling(5, min_periods=1).mean()
        df["volume_ma_20"] = df["Volume"].rolling(20, min_periods=1).mean()

        df["volume_trend"] = df["volume_ma_5"] / df["volume_ma_20"]

        df["volume_trend"].replace([float("inf"), -float("inf")], 0)

        df.to_parquet(OUTPUT_DIR / file.name)
        print(f"Saved: {file.name}")


if __name__ == "__main__":
    run_volume_trend()