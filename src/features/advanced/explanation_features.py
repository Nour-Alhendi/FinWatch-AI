"""
FinWatch AI — Layer 3: Explanation Features
============================================
Pre-computes price context features needed by the LLM Narrator (Layer 7C).
Keeps all math in the Feature layer — LLM only reads, never computes.

Output: data/explanations/explanation_features.parquet
  One row per ticker with:
    ema_20         — 20-day EMA of Close (latest value)
    price_3m_high  — max Close over last 90 calendar days
    price_3m_low   — min Close over last 90 calendar days
    price_vs_avg   — simple label: "above" / "below" / "near"
"""

import pandas as pd
from pathlib import Path

ROOT       = Path(__file__).resolve().parents[3]
INPUT_DIR  = ROOT / "data/features"
OUT_DIR    = ROOT / "data/explanations"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH   = OUT_DIR / "explanation_features.parquet"


def run():
    rows = []

    for f in sorted(INPUT_DIR.glob("*.parquet")):
        if f.stem.startswith("^"):
            continue

        df = pd.read_parquet(f)
        if "Close" not in df.columns or len(df) < 5:
            continue

        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

        closes = df["Close"].astype(float)

        ema_20 = float(closes.ewm(span=20, adjust=False).mean().iloc[-1])

        cutoff = df["Date"].max() - pd.Timedelta(days=90)
        df_3m  = df[df["Date"] >= cutoff]
        price_3m_high = float(df_3m["Close"].max())
        price_3m_low  = float(df_3m["Close"].min())

        current_price = float(closes.iloc[-1])
        ema_diff_pct  = ((current_price / ema_20) - 1) * 100 if ema_20 else 0
        if ema_diff_pct > 2:
            price_vs_avg = "above"
        elif ema_diff_pct < -2:
            price_vs_avg = "below"
        else:
            price_vs_avg = "near"

        rows.append({
            "ticker":         f.stem,
            "current_price":  round(current_price, 4),
            "ema_20":         round(ema_20, 4),
            "ema_diff_pct":   round(ema_diff_pct, 2),
            "price_3m_high":  round(price_3m_high, 4),
            "price_3m_low":   round(price_3m_low, 4),
            "price_vs_avg":   price_vs_avg,
        })

    result = pd.DataFrame(rows)
    result.to_parquet(OUT_PATH, index=False)
    print(f"[explanation_features] Saved {len(result)} tickers → {OUT_PATH}")
    return result


if __name__ == "__main__":
    run()
