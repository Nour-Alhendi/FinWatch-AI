# FinWatch AI — Earnings Calendar Pipeline
# Fetches upcoming earnings for all monitored tickers
# Saves to data/earnings_calendar.parquet

import os
import finnhub
import pandas as pd
from pathlib import Path
from datetime import date, timedelta
from finwatch.data.loader import COMPANY_NAMES

OUTPUT_PATH = Path("data/earnings_calendar.parquet")
client = finnhub.Client(api_key=os.environ["FINNHUB_API_KEY"])

def run():
    today = date.today()
    to    = today + timedelta(days=30)

    print(f"Fetching earnings {today} → {to}...")
    cal = client.earnings_calendar(
        _from=str(today), to=str(to), symbol=""
    )["earningsCalendar"]

    our_tickers = set(COMPANY_NAMES.keys())
    rows = []
    for e in cal:
        if e["symbol"] in our_tickers:
            days_until = (pd.to_datetime(e["date"]).date() - today).days
            rows.append({
                "ticker":          e["symbol"],
                "earnings_date":   e["date"],
                "days_until":      days_until,
                "hour":            e.get("hour", ""),
                "eps_estimate":    e.get("epsEstimate"),
                "revenue_estimate":e.get("revenueEstimate"),
                "earnings_soon":   days_until <= 7,
                "earnings_today":  days_until == 0,
            })
            print(f"  {e['symbol']}: {e['date']} ({days_until}d)")

    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["ticker","earnings_date","days_until","hour",
                 "eps_estimate","revenue_estimate","earnings_soon","earnings_today"]
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(df)} upcoming earnings to {OUTPUT_PATH}")

if __name__ == "__main__":
    run()