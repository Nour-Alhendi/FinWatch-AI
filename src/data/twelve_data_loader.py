"""
Twelve Data API adapter for FinWatch AI.

Output schema (byte-for-byte identical to old yfinance output):
  columns : Date, Open, High, Low, Close, Volume
  Date    : datetime64[ns]  (regular column, not the index)
  OHLC    : float64  (adjusted prices)
  Volume  : Int64
  index   : RangeIndex

Rate limits (Basic free tier):
  RATE_LIMIT_PER_MIN = 8  credits/minute
  RATE_LIMIT_PER_DAY = 800 credits/day

Modes:
  backfill(ticker, out_path)     — fetch ~10 years, write full parquet
  incremental(ticker, out_path)  — fetch last 5 days, append+dedupe into existing parquet
"""

import os
import time
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from twelvedata import TDClient

logger = logging.getLogger(__name__)

# ── Rate-limit constants (free Basic tier) ────────────────────────────────────
RATE_LIMIT_PER_MIN: int = 8
RATE_LIMIT_PER_DAY: int = 800

# Sleep after EVERY API call attempt (including failures) to respect 8/min
# Using 8s (slightly over 7.5) as a safety buffer
_SLEEP_PER_CALL: float = 8.0

# On 429: wait this many seconds before retrying
_RETRY_WAIT: float = 65.0

# Twelve Data symbol mapping from Yahoo format
_YAHOO_TO_TD: dict[str, str] = {
    "^SPX":  "SPX",
    "^GSPC": "SPX",
    "^VIX":  "VIX",
}
_SPX_FALLBACK = "SPY"


def _get_client() -> TDClient:
    key = os.getenv("TWELVE_DATA_API_KEY")
    if not key:
        raise RuntimeError(
            "TWELVE_DATA_API_KEY not set. Add it to your .env file."
        )
    return TDClient(apikey=key)


def _last_trading_day() -> pd.Timestamp:
    """Most recent NYSE trading day strictly before today (skips weekends + US holidays)."""
    import pandas_market_calendars as mcal
    nyse = mcal.get_calendar("NYSE")
    today = pd.Timestamp.today().normalize()
    # Look back up to 10 calendar days to find the last valid session
    window_start = today - pd.Timedelta(days=10)
    schedule = nyse.schedule(start_date=window_start, end_date=today - pd.Timedelta(days=1))
    if schedule.empty:
        # Extreme fallback: step back over weekends only
        candidate = today - pd.Timedelta(days=1)
        while candidate.weekday() >= 5:
            candidate -= pd.Timedelta(days=1)
        return candidate
    return schedule.index[-1].normalize()


def is_up_to_date(path: Path) -> bool:
    """True if parquet exists and its latest Date >= last NYSE trading day."""
    if not path.exists():
        return False
    try:
        df = pd.read_parquet(path, columns=["Date"])
        if df.empty:
            return False
        latest = pd.to_datetime(df["Date"]).max().normalize()
        return latest >= _last_trading_day()
    except Exception:
        return False


def _map_symbol(yahoo_ticker: str) -> str:
    return _YAHOO_TO_TD.get(yahoo_ticker, yahoo_ticker)


def _response_to_df(ts) -> pd.DataFrame:
    df = ts.as_pandas()
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.reset_index()
    df = df.rename(columns={"datetime": "Date"})

    col_map = {"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
    df = df.rename(columns=col_map)

    df["Date"] = pd.to_datetime(df["Date"])
    for col in ["Open", "High", "Low", "Close"]:
        df[col] = df[col].astype(float)
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").round(0).astype("Int64")

    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    return df.sort_values("Date").reset_index(drop=True)


def _single_call(td: TDClient, sym: str, start_date: str, end_date: str, outputsize: int) -> pd.DataFrame:
    """
    Make one API call for sym. Sleeps _SLEEP_PER_CALL after every attempt.
    On 429 waits _RETRY_WAIT then retries once.
    Raises on any other error.
    """
    for attempt in range(2):
        try:
            ts = td.time_series(
                symbol=sym,
                interval="1day",
                start_date=start_date,
                end_date=end_date,
                outputsize=outputsize,
                timezone="America/New_York",
            )
            df = _response_to_df(ts)
            time.sleep(_SLEEP_PER_CALL)
            return df
        except Exception as e:
            err_str = str(e)
            time.sleep(_SLEEP_PER_CALL)

            if '"code":429' in err_str and attempt == 0:
                print(f"\n  [rate limit] waiting {int(_RETRY_WAIT)}s...", end=" ", flush=True)
                time.sleep(_RETRY_WAIT)
                continue

            raise


def _fetch(
    td: TDClient,
    ticker: str,
    start_date: str,
    end_date: str,
    outputsize: int,
) -> tuple[pd.DataFrame, str]:
    """
    Fetch time series for ticker. For ^SPX: try SPX first, fall back to SPY.
    Returns (dataframe, actual_td_symbol_used).
    """
    td_symbol = _map_symbol(ticker)
    symbols_to_try = [td_symbol]

    if ticker in ("^SPX", "^GSPC") and td_symbol == "SPX":
        symbols_to_try.append(_SPX_FALLBACK)

    last_err = None
    for sym in symbols_to_try:
        try:
            df = _single_call(td, sym, start_date, end_date, outputsize)
            if not df.empty:
                if sym != td_symbol:
                    logger.warning(f"^SPX unavailable on free tier — using {sym} as proxy.")
                return df, sym
        except Exception as e:
            last_err = e
            logger.warning(f"  {sym} failed: {e}")

    raise RuntimeError(f"All attempts failed for {ticker}: {last_err}")


def backfill(ticker: str, out_path: Path, years: int = 10) -> bool:
    """Download ~10 years of daily OHLCV and write to out_path (Parquet)."""
    if is_up_to_date(out_path):
        print(f"already up to date: {ticker}")
        return True

    td = _get_client()
    end_date   = datetime.today() - timedelta(days=1)
    start_date = end_date - timedelta(days=365 * years)

    print(f"Downloading {ticker}...", end=" ", flush=True)
    try:
        df, sym_used = _fetch(
            td, ticker,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            outputsize=5000,
        )
        if df.empty:
            print("no data")
            return False
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)
        print(f"saved ({len(df)} rows, latest: {df['Date'].max().date()}) [td:{sym_used}]")
        return True
    except Exception as e:
        print(f"failed: {e}")
        return False


def incremental(ticker: str, out_path: Path, lookback_days: int = 5) -> bool:
    """Fetch last N trading days and append+dedupe into existing Parquet."""
    if is_up_to_date(out_path):
        print(f"already up to date: {ticker}")
        return True

    td = _get_client()
    end_date   = datetime.today() - timedelta(days=1)
    start_date = end_date - timedelta(days=lookback_days)

    print(f"Incremental update {ticker}...", end=" ", flush=True)
    try:
        df_new, sym_used = _fetch(
            td, ticker,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            outputsize=lookback_days + 5,
        )
        if df_new.empty:
            print("no new data")
            return False

        if out_path.exists():
            df_existing = pd.read_parquet(out_path)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined["Date"] = pd.to_datetime(df_combined["Date"])
            df_combined = (
                df_combined
                .drop_duplicates(subset=["Date"])
                .sort_values("Date")
                .reset_index(drop=True)
            )
        else:
            df_combined = df_new

        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_combined.to_parquet(out_path, index=False)
        print(f"updated ({len(df_combined)} rows total) [td:{sym_used}]")
        return True
    except Exception as e:
        print(f"failed: {e}")
        return False
