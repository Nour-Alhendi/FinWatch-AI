import pandas as pd
from pathlib import Path

INPUT_DIR = Path("data/detection")
OUTPUT_DIR = Path("data/detection")

# maps anomaly_score (0-3) to severity label based on how many models flagged the row
def classify_severity(row):
    score = row["anomaly_score"]
    if score == 0:
        return None       # no anomaly
    if score == 1:
        return "minor"    # 1 model flagged
    if score == 2:
        return "warning"  # 2 models flagged
    if score >= 3:
        return "critical" # all 3 models flagged

def severity(file_path):
    df = pd.read_parquet(file_path)
    df["severity"] = df.apply(classify_severity, axis=1)
    return df

def run_severity():
    for file in INPUT_DIR.glob("*.parquet"):
        df = severity(file)
        df.to_parquet(OUTPUT_DIR / file.name)
        print(f"Saved: {file.name}")

if __name__ == "__main__":
    run_severity()
