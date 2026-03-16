from pathlib import Path
import pandas as pd
from sklearn.ensemble import IsolationForest

INPUT_DIR = Path("data/features")
OUTPUT_DIR = Path("data/detection")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Isolation Forest Anomaly Detection
def isolation_forest(file_path):
    df = pd.read_parquet(file_path)
    features = ["returns", "volatility", "z_score", "rolling_std"]
    model = IsolationForest(contamination=0.05)
    X = df[features].dropna()
    model.fit(X)
    df.loc[X.index, "if_anomaly"] = model.predict(X)
    return df

# Run Isolation Forest on all files and save results
def run_isolation_forest():
    for file in INPUT_DIR.glob("*.parquet"):
        df = isolation_forest(file)
        df.to_parquet(OUTPUT_DIR/file.name)
        print(f"saved {file.name}")

# Entry point
if __name__ == "__main__":
    run_isolation_forest()