import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from features.basic.basic_pipeline import run as run_basic
from features.context.context_pipeline import run_context_pipeline as run_context
from features.advanced.advanced_pipeline import run_advanced_features as run_advanced
from features.anomaly_types.anomaly_typing import compute_group_scores

FEATURES_DIR = Path(__file__).resolve().parents[2] / "data/features"


def run_feature_pipeline() -> pd.DataFrame:
    # Layer 3: compute all base, context, and advanced features — writes to data/features/*.parquet
    run_basic()
    run_context()
    run_advanced()

    # Loop over every ticker file and append anomaly group scores
    print("--- Anomaly Typing ---")
    frames = []
    for file in sorted(FEATURES_DIR.glob("*.parquet")):
        ticker = file.stem
        try:
            # Load ticker history, score it, overwrite the file with new columns
            df = pd.read_parquet(file)
            df = compute_group_scores(df)
            df.to_parquet(file)
            frames.append(df)
            print(f"Scored: {ticker}")
        except Exception as e:
            # Skip tickers with missing columns or corrupt files
            print(f"Skipped {ticker}: {e}")

    # Combine all scored tickers into one DataFrame and return
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


if __name__ == "__main__":
    run_feature_pipeline()
