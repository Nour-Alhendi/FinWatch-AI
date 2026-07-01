
# FinWatch AI — Agent Tools
# Each function is a tool the AI Agent can call
# to answer user questions about stocks and risk.

import pandas as pd
from pathlib import Path
from finwatch.data.loader import SECTORS

ROOT = Path(__file__).resolve().parents[2]

_SIGNAL_DISPLAY = {
    "ENTRY":      "FAVORABLE",
    "BUY_SIGNAL": "FAVORABLE",
    "HOLD":       "MONITOR",
    "WATCH":      "MONITOR",
    "EXIT":       "ELEVATED",
    "REDUCE":     "ELEVATED",
    "NEUTRAL":    "NEUTRAL",
}

def _display_signal(raw: str) -> str:
    """Translate internal trading-action codes to public risk-posture labels."""
    return _SIGNAL_DISPLAY.get(str(raw).upper(), raw)

FEATURES_DIR     = ROOT / "data/features"
DETECTION_DIR    = ROOT / "data/detection"
DECISIONS_DIR    = ROOT / "data/decisions"
NEWS_DIR         = ROOT / "data/news"
EXPLANATIONS_DIR = ROOT / "data/explanations"


def get_stock_analysis(ticker: str) -> dict:
    """Get the full analysis for a single stock: severity, trading signal,
    drawdown probability, anomaly type, confidence, and context.
    Use when the user asks about the overall status or assessment of a stock.

    Args:
        ticker: Stock symbol, e.g. 'AAPL'
    """
    ticker = ticker.upper()
    df = pd.read_parquet(DECISIONS_DIR / "decisions.parquet")
    row = df[df["ticker"] == ticker]
    if row.empty:
        return {"error": f"Ticker {ticker} not found"}
    row = row.iloc[-1]
    features_path = FEATURES_DIR / f"{ticker}.parquet"
    anomaly_type = "unknown"
    if features_path.exists():
        feat_df = pd.read_parquet(features_path)
        if "anomaly_type" in feat_df.columns:
            anomaly_type = feat_df["anomaly_type"].iloc[-1]
    return {
        "ticker":       ticker,
        "severity":     row["severity"],
        "signal":       _display_signal(row["trading_signal"]),
        "p_drawdown":   row["p_drawdown"],
        "anomaly_type": anomaly_type,
        "confidence":   row["confidence"],
        "context":      row["context"],
    }


def get_risk_metrics(ticker: str) -> dict:
    """Get risk metrics for a stock: Value at Risk (95%), Expected Shortfall (95%),
    ES ratio, and 30-day max drawdown.
    Use when the user asks about risk, volatility, potential losses, VaR, or expected shortfall.

    Args:
        ticker: Stock symbol, e.g. 'AAPL'
    """    
    ticker = ticker.upper()
    detection_path = DETECTION_DIR / f"{ticker}.parquet"
    if not detection_path.exists():
        return {"error": f"No data for {ticker}"}
    df = pd.read_parquet(detection_path)
    row = df.iloc[-1]
    return {
        "ticker":           ticker,
        "var_95":           row["var_95"],
        "es_95":            row["es_95"],
        "es_ratio":         row["es_ratio"],
        "max_drawdown_30d": row["max_drawdown_30d"],
    }


def explain_anomaly(ticker: str) -> dict:
    """Explain why a stock is flagged as anomalous: the dominant anomaly type,
    the four anomaly group scores (volume, volatility, price, context), the main
    SHAP driver, the top-3 SHAP features, and a narrative explanation.
    Use when the user asks why a stock is anomalous or what is driving the anomaly.

    Args:
        ticker: Stock symbol, e.g. 'AAPL'
    """
    ticker = ticker.upper()
    explanations_path = EXPLANATIONS_DIR / "explanations.parquet"
    if not explanations_path.exists():
        return {"error": "No explanations data available"}
    detection_path = DETECTION_DIR / f"{ticker}.parquet"
    if not detection_path.exists():
        return {"error": f"No detection data for {ticker}"}
    explanations_df = pd.read_parquet(explanations_path)
    exp_filtered = explanations_df[explanations_df["ticker"] == ticker]
    if len(exp_filtered) == 0:
        return {"error": f"No explanations for {ticker}"}
    exp_row = exp_filtered.iloc[-1]
    detection_df = pd.read_parquet(detection_path)
    det_row = detection_df.iloc[-1]
    return {
        "anomaly_type":             det_row["anomaly_type"],
        "volume_anomaly_score":     det_row["volume_anomaly_score"],
        "volatility_anomaly_score": det_row["volatility_anomaly_score"],
        "price_anomaly_score":      det_row["price_anomaly_score"],
        "context_anomaly_score":    det_row["context_anomaly_score"],
        "driver":                   exp_row["driver"],
        "driver_shap":              exp_row["driver_shap"],
        "top3_shap":                exp_row["top3_shap"],
        "narrative_text":           exp_row["narrative_text"],
        "confirmation":             exp_row["confirmation"],
        "conflict":                 exp_row["conflict"],
    }


def get_market_context(ticker: str) -> dict:
    """Get the market context for a stock: market regime (bull/bear), volatility
    regime, whether the anomaly is market-wide, the context anomaly score, and a
    text context summary.
    Use when the user asks about market conditions, regime, or the broader context of a stock.

    Args:
        ticker: Stock symbol, e.g. 'AAPL'
    """    
    ticker = ticker.upper()
    detection_path = DETECTION_DIR / f"{ticker}.parquet"
    if not detection_path.exists():
        return {"error": f"No data for {ticker}"}
    df = pd.read_parquet(detection_path)
    row = df.iloc[-1]
    decisions_df = pd.read_parquet(DECISIONS_DIR / "decisions.parquet")
    dec_row = decisions_df[decisions_df["ticker"] == ticker]
    if dec_row.empty:
        context = "N/A"
    else:
        context = dec_row.iloc[-1]["context"]
    return {
        "ticker":               ticker,
        "regime":               row["regime"],
        "vol_regime":           row["vol_regime"],
        "is_market_wide":       row["is_market_wide"],
        "context_anomaly_score": row["context_anomaly_score"],
        "context":              context,
    }

def get_news_sentiment(ticker: str) -> dict:
    """Get news sentiment for a stock: sentiment labels, VADER score, FinBERT score,
    combined sentiment score, number of articles, an LLM summary, and top headlines.
    Use when the user asks about news, sentiment, or what the media is saying about a stock.

    Args:
        ticker: Stock symbol, e.g. 'AAPL'
    """
    ticker = ticker.upper()
    news_path = EXPLANATIONS_DIR / "llm_news_enriched.parquet"
    if not news_path.exists():
        return {"error": "No news data available"}
    news_df = pd.read_parquet(news_path)
    filtered = news_df[news_df["ticker"] == ticker]
    if len(filtered) == 0:
        return {"error": f"No news data for {ticker}"}
    row = filtered.iloc[-1]
    return {
        "ticker":               ticker,
        "news_sentiment":       row["news_sentiment"].tolist() if hasattr(row["news_sentiment"], 'tolist') else row["news_sentiment"],
        "vader_score":          row["vader_score"],
        "finbert_score":        row["finbert_score"],
        "news_sentiment_score": row["news_sentiment_score"],
        "n_articles":           row["n_articles"],
        "llm_summary":          row["llm_summary"],
        "top_news":             row["top_news"].tolist() if hasattr(row["top_news"], 'tolist') else row["top_news"],
    }

def get_trend_analysis(ticker: str) -> dict:
    """Get trend indicators for a stock: 50/200-day moving averages, 5/10-day
    momentum, trend strength, volume trend, and RSI.
    Use when the user asks about trend, momentum, moving averages, or technical direction.

    Args:
        ticker: Stock symbol, e.g. 'AAPL'
    """
    ticker = ticker.upper()
    detection_path = DETECTION_DIR / f"{ticker}.parquet"
    if not detection_path.exists():
        return {"error": f"No data for {ticker}"}
    df = pd.read_parquet(detection_path)
    row = df.iloc[-1]
    return {
        "ticker":         ticker,
        "ma50":           row["ma50"],
        "ma200":          row["ma200"],
        "momentum_5":     row["momentum_5"],
        "momentum_10":    row["momentum_10"],
        "trend_strength": row["trend_strength"],
        "volume_trend":   row["volume_trend"],
        "rsi":            row["rsi"],
    }

def get_portfolio_overview() -> dict:
    """Get an overview of all monitored stocks: total count, severity distribution,
    trading signal distribution, and a per-stock summary.
    Use when the user asks about the overall portfolio, all stocks, or a market-wide summary.
    Takes no arguments.
    """    
    df = pd.read_parquet(DECISIONS_DIR / "decisions.parquet")
    # letzte Zeile pro Ticker
    latest = df.sort_values("date").groupby("ticker").last().reset_index()
    
    latest["signal"] = latest["trading_signal"].map(_display_signal)
    stocks = latest[["ticker", "severity", "signal", "p_drawdown", "anomaly_type", "confidence"]].to_dict(orient="records")

    return {
        "total_stocks":    len(latest),
        "severity_counts": latest["severity"].value_counts().to_dict(),
        "signal_counts":   {_display_signal(k): v for k, v in latest["trading_signal"].value_counts().to_dict().items()},
        "stocks":          stocks,
    }

def get_macro_context() -> dict:
    """Get current macro environment: 10Y Treasury yield, Dollar Index, VIX,
    and their interpretations (rate_env, dollar_env, vix_env, risk_off flag).
    Use when the user asks about macro conditions, interest rates, dollar, or
    market-wide risk environment. Takes no arguments.
    """
    path = ROOT / "data/macro_context.parquet"
    if not path.exists():
        return {"error": "No macro data available"}
    row = pd.read_parquet(path).iloc[-1]
    return {
        "treasury_10y": row["treasury_10y"],
        "dollar_index":  row["dollar_index"],
        "vix":           row["vix"],
        "rate_env":      row["rate_env"],
        "dollar_env":    row["dollar_env"],
        "vix_env":       row["vix_env"],
        "risk_off":      bool(row["risk_off"]),
    }


def get_earnings_calendar() -> dict:
    """Get upcoming earnings dates for monitored stocks in the next 30 days.
    Use when the user asks about earnings, upcoming events, or why a stock
    might be volatile soon. Takes no arguments.
    """
    path = ROOT / "data/earnings_calendar.parquet"
    if not path.exists():
        return {"error": "No earnings data available"}
    df = pd.read_parquet(path)
    if df.empty:
        return {"earnings": [], "message": "No upcoming earnings in next 30 days"}
    return {
        "earnings": df.sort_values("days_until")[
            ["ticker", "earnings_date", "days_until", "earnings_soon"]
        ].to_dict(orient="records"),
        "soon": df[df["earnings_soon"] == True]["ticker"].tolist(),
    }


def get_correlation_risk(ticker: str) -> dict:
    """Get correlation risk for a stock: average correlation with the portfolio,
    number of highly correlated peers, and concentration risk level.
    Use when the user asks about correlation, diversification, or cluster risk.

    Args:
        ticker: Stock symbol, e.g. 'AAPL'
    """
    ticker = ticker.upper()
    path = ROOT / "data/correlation_risk.parquet"
    if not path.exists():
        return {"error": "No correlation data available"}
    df = pd.read_parquet(path)
    row = df[df["ticker"] == ticker]
    if row.empty:
        return {"error": f"No correlation data for {ticker}"}
    row = row.iloc[0]
    return {
        "ticker":             ticker,
        "avg_correlation":    row["avg_correlation"],
        "high_corr_count":    int(row["high_corr_count"]),
        "high_corr_peers":    row["high_corr_peers"],
        "concentration_risk": row["concentration_risk"],
    }

def get_sector_analysis() -> dict:
    """Get a sector-level breakdown: for each sector, the tickers, severity counts,
    signal counts, and average drawdown probability.
    Use when the user asks about sectors, which sector is riskiest, or sector comparisons.
    Takes no arguments.
    """
    df = pd.read_parquet(DECISIONS_DIR / "decisions.parquet")
    latest = df.sort_values("date").groupby("ticker").last().reset_index()
    
    # Sektor-Zuordnung aus loader.py
    ticker_to_sector = {t: s for s, tickers in SECTORS.items() for t in tickers}
    latest["sector"] = latest["ticker"].map(ticker_to_sector).fillna("Other")

    result = {}
    for sector, group in latest.groupby("sector"):
        result[sector] = {
            "tickers":         group["ticker"].tolist(),
            "severity_counts": group["severity"].value_counts().to_dict(),
            "signal_counts":   {_display_signal(k): v for k, v in group["trading_signal"].value_counts().to_dict().items()},
            "avg_p_drawdown":  round(group["p_drawdown"].mean(), 3),
        }
    return result