import streamlit as st
import pandas as pd
import numpy as np
import ast
from pathlib import Path

# ── Universe ──────────────────────────────────────────────────────────────────
COMPANY_NAMES = {
    "AAPL":"Apple","MSFT":"Microsoft","NVDA":"Nvidia","AMD":"AMD","GOOG":"Alphabet",
    "IBM":"IBM","NOK":"Nokia","INTC":"Intel",
    "PLTR":"Palantir","META":"Meta","CRM":"Salesforce","SNOW":"Snowflake","AI":"C3.ai",
    "NBIS":"Nebius Group",
    "PANW":"Palo Alto","CRWD":"CrowdStrike","NET":"Cloudflare",
    "TSLA":"Tesla","AMZN":"Amazon",
    "JPM":"JPMorgan","BAC":"Bank of America","GS":"Goldman Sachs","BLK":"BlackRock",
    "V":"Visa","MA":"Mastercard",
    "JNJ":"J&J","PFE":"Pfizer","UNH":"UnitedHealth","ABBV":"AbbVie",
    "LLY":"Eli Lilly","AMGN":"Amgen",
    "PG":"P&G","KO":"Coca-Cola","COST":"Costco","WMT":"Walmart",
    "XOM":"ExxonMobil","CVX":"Chevron","COP":"ConocoPhillips","EOG":"EOG Resources",
    "CAT":"Caterpillar","HON":"Honeywell","BA":"Boeing","GE":"GE Aerospace","RTX":"Raytheon",
    "ENPH":"Enphase","NEE":"NextEra","FSLR":"First Solar",
    "AVGO":"Broadcom","QCOM":"Qualcomm","MU":"Micron","MRVL":"Marvell","ASML":"ASML Holding",
    "IREN":"IREN Energy","MARA":"Marathon Digital","RIOT":"Riot Platforms","COIN":"Coinbase",
    "IONQ":"IonQ","QBTS":"D-Wave Quantum",
}

ETF_STOCKS = {
    "Technology (XLK)":             ["AAPL","MSFT","NVDA","AMD","GOOG","IBM","NOK","INTC"],
    "AI & Robotics (BOTZ)":         ["PLTR","META","CRM","SNOW","AI","NBIS"],
    "Cybersecurity":                 ["PANW","CRWD","NET"],
    "Financials (XLF)":             ["JPM","BAC","GS","BLK","V","MA"],
    "Healthcare (XLV)":             ["JNJ","PFE","UNH","ABBV","LLY","AMGN"],
    "Consumer Staples (XLP)":       ["PG","KO","COST","WMT"],
    "Energy (XLE)":                 ["XOM","CVX","COP","EOG"],
    "Consumer Discret. (XLY)":      ["TSLA","AMZN"],
    "Industrials (XLI)":            ["CAT","HON","BA","GE","RTX"],
    "Clean Energy (ICLN)":          ["ENPH","NEE","FSLR"],
    "Semiconductors (XLK proxy)":   ["AVGO","QCOM","MU","MRVL","ASML"],
    "Crypto & Mining":               ["IREN","MARA","RIOT","COIN"],
    "Quantum Computing":             ["IONQ","QBTS"],
}

SECTORS = {
    "Technology":        ["AAPL","MSFT","NVDA","AMD","GOOG","IBM","NOK","INTC"],
    "Semiconductors":    ["AVGO","QCOM","MU","MRVL","ASML"],
    "AI & Software":     ["PLTR","META","CRM","SNOW","AI","NBIS"],
    "Cybersecurity":     ["PANW","CRWD","NET"],
    "Consumer":          ["TSLA","AMZN"],
    "Financials":        ["JPM","BAC","GS","BLK","V","MA"],
    "Healthcare":        ["JNJ","PFE","UNH","ABBV","LLY","AMGN"],
    "Consumer Staples":  ["PG","KO","COST","WMT"],
    "Energy":            ["XOM","CVX","COP","EOG"],
    "Industrials":       ["CAT","HON","BA","GE","RTX"],
    "Clean Energy":      ["ENPH","NEE","FSLR"],
    "Crypto & Mining":   ["IREN","MARA","RIOT","COIN"],
    "Quantum Computing": ["IONQ","QBTS"],
}

PERIOD_MAP = {"5D":5,"1M":21,"Q":63,"1Y":252,"5Y":1260,"10Y":2520}
SEV_COLOR  = {"CRITICAL":"#e05252","WARNING":"#c49a3c","WATCH":"#5a7ab4","NORMAL":"#4a9e6a","POSITIVE_MOMENTUM":"#4a9e6a","REVIEW":"#7a6ab4"}
SEV_SHORT  = {"CRITICAL":"CRIT","WARNING":"WARN","WATCH":"WTCH","NORMAL":"NORM","POSITIVE_MOMENTUM":"POS+","REVIEW":"REV"}
SEV_CLS    = {"CRITICAL":"s-critical","WARNING":"s-warning","WATCH":"s-watch","NORMAL":"s-normal","POSITIVE_MOMENTUM":"s-positive","REVIEW":"s-review"}

# Display labels for trading signals — internal values (ENTRY/HOLD/EXIT) remain
# unchanged in decision logic, backtests, and portfolio.json.
SIGNAL_DISPLAY: dict[str, str] = {
    "ENTRY":      "FAVORABLE",
    "BUY_SIGNAL": "FAVORABLE",
    "HOLD":       "MONITOR",
    "NEUTRAL":    "NEUTRAL",
    "WATCH":      "MONITOR",
    "EXIT":       "ELEVATED",
    "REDUCE":     "ELEVATED",
}


def safe_list(val):
    if isinstance(val, list): return val
    try: return ast.literal_eval(str(val))
    except: return []


def find_sector(ticker):
    for s, ts in SECTORS.items():
        if ticker in ts:
            return s
    return list(SECTORS.keys())[0]


@st.cache_data
def load_decisions():
    p = Path("data/decisions/decisions.parquet")
    if p.exists():
        df = pd.read_parquet(p)
        df = df[df["ticker"].isin(set(COMPANY_NAMES))]
        df["_s"] = df["severity"].map({"CRITICAL":0,"WARNING":1,"WATCH":2,"REVIEW":3,"POSITIVE_MOMENTUM":4,"NORMAL":5}).fillna(99)
        return df.sort_values("_s").drop(columns=["_s"])
    tickers = list(COMPANY_NAMES.keys())
    np.random.seed(42)
    sevs = ["CRITICAL"]*35+["WARNING"]*5+["WATCH"]*3+["NORMAL"]*2
    np.random.shuffle(sevs)
    return pd.DataFrame({
        "ticker":tickers,"date":["2026-03-20"]*len(tickers),
        "severity":sevs[:len(tickers)],"action":["ESCALATE"]*len(tickers),
        "confidence":np.random.uniform(0,1,len(tickers)).round(2),
        "context":["idiosyncratic"]*len(tickers),
        "momentum_signal":np.random.choice(["falling","neutral","rising"],len(tickers)),
        "summary":["Multiple risk indicators triggered."]*len(tickers),
        "caution_flag":[None]*len(tickers),
        "direction":["down"]*len(tickers),"p_down":[0.7]*len(tickers),
        "trading_signal":["EXIT"]*len(tickers),
        "p_drawdown":[0.7]*len(tickers),
    })


@st.cache_data
def load_detection(ticker):
    p = Path(f"data/detection/{ticker}.parquet")
    if p.exists():
        df = pd.read_parquet(p)
        df["Date"] = pd.to_datetime(df["Date"])
        return df.sort_values("Date").reset_index(drop=True)
    return None


@st.cache_data
def load_news():
    p = Path("data/explanations/llm_news_enriched.parquet")
    if p.exists():
        return pd.read_parquet(p)
    return None


@st.cache_data
def load_spx():
    p = Path("data/raw/references/^SPX.parquet")
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df.columns = [c.capitalize() if c.lower() in ("open","high","low","close","volume") else c for c in df.columns]
    if "Date" not in df.columns and df.index.name == "Date":
        df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").reset_index(drop=True)


@st.cache_data
def load_anomaly_precision():
    p = Path("data/backtesting/anomaly_precision.parquet")
    if p.exists():
        return pd.read_parquet(p)
    return None


@st.cache_data(ttl=3600)
def load_analyst_ratings():
    p = Path("data/analyst_ratings.parquet")
    if p.exists():
        return pd.read_parquet(p)
    return None


@st.cache_data(ttl=3600)
def load_earnings_calendar():
    p = Path("data/earnings_calendar.parquet")
    if p.exists():
        return pd.read_parquet(p)
    return None


@st.cache_data(ttl=3600)
def load_macro_context():
    p = Path("data/macro_context.parquet")
    if p.exists():
        return pd.read_parquet(p)
    return None


@st.cache_data(ttl=3600)
def load_correlation_risk():
    p = Path("data/correlation_risk.parquet")
    if p.exists():
        return pd.read_parquet(p)
    return None


@st.cache_data(ttl=300)
def load_price_summary():
    result = {}
    det_dir = Path("data/detection")
    if not det_dir.exists():
        return result
    for ticker in set(COMPANY_NAMES):
        f = det_dir / f"{ticker}.parquet"
        if not f.exists():
            continue
        try:
            tmp = pd.read_parquet(f, columns=["Date", "Close"])
            tmp = tmp.dropna(subset=["Close"]).sort_values("Date").tail(2)
            if len(tmp) >= 2:
                prev = float(tmp.iloc[-2]["Close"])
                last = float(tmp.iloc[-1]["Close"])
                result[ticker] = (last, (last - prev) / prev * 100)
            elif len(tmp) == 1:
                result[ticker] = (float(tmp.iloc[-1]["Close"]), 0.0)
        except Exception:
            pass
    return result


def rel_time(ts: "int | float") -> str:
    """Convert a Unix timestamp to a human-readable relative string (e.g. '2h ago')."""
    from datetime import datetime, timezone
    if not ts:
        return ""
    try:
        diff = datetime.now(timezone.utc) - datetime.fromtimestamp(float(ts), tz=timezone.utc)
        secs = int(diff.total_seconds())
        if secs < 60:   return "just now"
        if secs < 3600: return f"{secs // 60}m ago"
        if secs < 86400: return f"{secs // 3600}h ago"
        return f"{secs // 86400}d ago"
    except Exception:
        return ""


@st.cache_data(ttl=1800)
def fetch_stock_news(ticker: str, limit: int = 1) -> list:
    """Fetch recent company news for a ticker from Finnhub (cached 30 min)."""
    import os, requests
    from datetime import date, timedelta
    api_key = os.environ.get("FINNHUB_API_KEY", "")
    if not api_key:
        return []
    today = date.today()
    try:
        r = requests.get(
            "https://finnhub.io/api/v1/company-news",
            params={
                "symbol": ticker,
                "from":   (today - timedelta(days=7)).isoformat(),
                "to":     today.isoformat(),
                "token":  api_key,
            },
            timeout=8,
        )
        r.raise_for_status()
        arts = r.json()
        if not isinstance(arts, list):
            return []
        return [a for a in arts[:limit] if a.get("headline")]
    except Exception:
        return []
