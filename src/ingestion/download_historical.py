import yaml
from pathlib import Path
from data.twelve_data_loader import backfill

ROOT = Path(__file__).resolve().parents[2]


def run():
    with open(ROOT / "config/assets.yaml", "r") as f:
        config = yaml.safe_load(f)

    assets     = config["assets"]
    references = config.get("references", [])

    output_dir = ROOT / "data/raw/raw_clean"
    ref_dir    = ROOT / "data/raw/references"

    output_dir.mkdir(parents=True, exist_ok=True)
    ref_dir.mkdir(parents=True, exist_ok=True)

    print("=== Downloading reference symbols ===")
    for asset in references:
        ticker = asset["ticker"]
        backfill(ticker, ref_dir / f"{ticker}.parquet")

    print("\n=== Downloading asset symbols ===")
    for asset in assets:
        ticker = asset["ticker"]
        backfill(ticker, output_dir / f"{ticker}.parquet")

    print("\nDownload finished.")


if __name__ == "__main__":
    run()
