# FinWatch AI — Macro Context Pipeline
# Fetches 10Y Treasury, Dollar Index, VIX from yfinance
# Saves to data/macro_context.parquet

import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import date

OUTPUT_PATH = Path("data/macro_context.parquet")

MACRO_SYMBOLS = {
    "treasury_10y": "^TNX",
    "dollar_index":  "DX-Y.NYB",
    "vix":           "^VIX",
}

def interpret_macro(treasury, dollar, vix) -> dict:
    return {
        "rate_env":    "high" if treasury > 4.5 else "moderate" if treasury > 3.5 else "low",
        "dollar_env":  "strong" if dollar > 103 else "weak" if dollar < 97 else "neutral",
        "vix_env":     "fear" if vix > 25 else "elevated" if vix > 18 else "calm",
        "risk_off":    vix > 25 or treasury > 5.0 or dollar > 105,
    }

def run():
    row = {"date": str(date.today())}
    for name, sym in MACRO_SYMBOLS.items():
        t = yf.Ticker(sym)
        row[name] = round(t.fast_info["last_price"], 3)

    interp = interpret_macro(row["treasury_10y"], row["dollar_index"], row["vix"])
    row.update(interp)

    df = pd.DataFrame([row])
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    print("Macro context saved:")
    print(df.T.to_string(header=False))

if __name__ == "__main__":
    run()