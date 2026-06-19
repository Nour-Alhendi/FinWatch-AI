# FinWatch AI — Correlation Risk Pipeline
# Computes 60-day return correlation matrix across all tickers
# Flags high-correlation clusters and concentration risk
# Saves to data/correlation_risk.parquet

import pandas as pd
import numpy as np
from pathlib import Path
from finwatch.data.loader import COMPANY_NAMES

DETECTION_DIR  = Path("data/detection")
OUTPUT_MATRIX  = Path("data/correlation_matrix.parquet")
OUTPUT_RISK    = Path("data/correlation_risk.parquet")
LOOKBACK       = 60   # days
HIGH_CORR      = 0.65 # threshold

def run():
    # 1. Load returns for all tickers
    returns = {}
    for ticker in COMPANY_NAMES.keys():
        f = DETECTION_DIR / f"{ticker}.parquet"
        if not f.exists():
            continue
        df = pd.read_parquet(f, columns=["Date", "returns"])
        df = df.dropna().sort_values("Date").tail(LOOKBACK)
        returns[ticker] = df.set_index("Date")["returns"]

    ret_df = pd.DataFrame(returns).dropna(axis=1, how="all")
    print(f"Computing correlation matrix for {len(ret_df.columns)} tickers...")

    # 2. Correlation matrix
    corr = ret_df.corr()
    corr.to_parquet(OUTPUT_MATRIX)

    # 3. Per-ticker risk metrics
    rows = []
    for ticker in corr.columns:
        others = corr[ticker].drop(ticker)
        high_corr_peers = others[others >= HIGH_CORR].index.tolist()
        avg_corr = round(others.mean(), 3)
        rows.append({
            "ticker":            ticker,
            "avg_correlation":   avg_corr,
            "high_corr_count":   len(high_corr_peers),
            "high_corr_peers":   ", ".join(high_corr_peers[:5]),
            "concentration_risk": "HIGH" if len(high_corr_peers) >= 5
                                  else "MEDIUM" if len(high_corr_peers) >= 2
                                  else "LOW",
        })

    risk_df = pd.DataFrame(rows).sort_values("avg_correlation", ascending=False)
    OUTPUT_RISK.parent.mkdir(parents=True, exist_ok=True)
    risk_df.to_parquet(OUTPUT_RISK, index=False)

    print(f"\nTop 10 most correlated tickers:")
    print(risk_df[["ticker","avg_correlation","high_corr_count","concentration_risk"]].head(10).to_string(index=False))
    print(f"\nSaved correlation matrix → {OUTPUT_MATRIX}")
    print(f"Saved correlation risk   → {OUTPUT_RISK}")

if __name__ == "__main__":
    run()