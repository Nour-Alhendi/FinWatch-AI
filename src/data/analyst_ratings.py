# FinWatch AI — Analyst Ratings Pipeline
# Fetches analyst recommendation trends from Finnhub for all tickers
# and saves to data/analyst_ratings.parquet

import os
import sys
import time
import pandas as pd
import finnhub
from pathlib import Path

# Make `finwatch` importable when running this script directly
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from finwatch.data.loader import COMPANY_NAMES

OUTPUT_PATH = Path("data/analyst_ratings.parquet")
client = finnhub.Client(api_key=os.environ["FINNHUB_API_KEY"])


def fetch_ratings(tickers: list) -> pd.DataFrame:
    rows = []
    for ticker in tickers:
        try:
            data = client.recommendation_trends(ticker)
            if not data:
                print(f"  {ticker}: no data")
                continue
            latest = data[0]  # most recent month
            total = (latest["strongBuy"] + latest["buy"] +
                     latest["hold"] + latest["sell"] + latest["strongSell"])
            bullish = latest["strongBuy"] + latest["buy"]
            bearish = latest["sell"] + latest["strongSell"]
            rows.append({
                "ticker":        ticker,
                "period":        latest["period"],
                "strong_buy":    latest["strongBuy"],
                "buy":           latest["buy"],
                "hold":          latest["hold"],
                "sell":          latest["sell"],
                "strong_sell":   latest["strongSell"],
                "total":         total,
                "bullish":       bullish,
                "bearish":       bearish,
                "bullish_pct":   round(bullish / total, 3) if total > 0 else None,
                "consensus":     _consensus(bullish, latest["hold"], bearish, total),
            })
            print(f"  {ticker}: {bullish}/{total} bullish ({round(bullish/total*100)}%)" if total > 0 else f"  {ticker}: ok")
        except Exception as e:
            print(f"  {ticker}: ERROR {e}")
        time.sleep(0.5)
    return pd.DataFrame(rows)


def _consensus(bullish, hold, bearish, total) -> str:
    if total == 0:
        return "N/A"
    bull_pct = bullish / total
    bear_pct = bearish / total
    if bull_pct >= 0.65:
        return "STRONG BUY"
    elif bull_pct >= 0.50:
        return "BUY"
    elif bear_pct >= 0.30:
        return "SELL"
    elif hold / total >= 0.40:
        return "HOLD"
    else:
        return "NEUTRAL"


def run():
    tickers = list(COMPANY_NAMES.keys())
    print(f"Fetching analyst ratings for {len(tickers)} tickers...")
    df = fetch_ratings(tickers)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(df)} rows to {OUTPUT_PATH}")
    print(df[["ticker", "consensus", "bullish_pct", "total"]].to_string(index=False))


if __name__ == "__main__":
    run()