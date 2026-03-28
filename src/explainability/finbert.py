"""
FinWatch AI — Layer 7E: News + Sentiment Analysis
==================================================
Takes LLM Narrator output and enriches it with news sentiment using VADER.
VADER (Valence Aware Dictionary and sEntiment Reasoner) is lightweight,
requires no model download, and is well-suited for financial news headlines.

Output: ticker + llm_summary + top_news + news_sentiment
"""

import os
import time
import logging
import pandas as pd
from pathlib import Path
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests

ROOT = Path(__file__).resolve().parents[2]

FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY")
NEWS_ENDPOINT   = "https://finnhub.io/api/v1/company-news"

_ANALYZER = SentimentIntensityAnalyzer()


def fetch_news(ticker: str, limit: int = 3, retries: int = 3):
    """Returns (headlines, sources). Finnhub proxy URLs are unreliable, so we store source names instead."""
    if not FINNHUB_API_KEY:
        return [], []

    from datetime import date, timedelta
    today = date.today()
    params = {
        "symbol": ticker,
        "from":   (today - timedelta(days=30)).isoformat(),
        "to":     today.isoformat(),
        "token":  FINNHUB_API_KEY,
    }

    for attempt in range(retries):
        try:
            r = requests.get(NEWS_ENDPOINT, params=params, timeout=10)
            r.raise_for_status()
            articles = r.json()
            if not isinstance(articles, list):
                logging.warning(f"[finbert] Unexpected response for {ticker}: {articles}")
                return [], []
            valid   = [a for a in articles[:limit] if "headline" in a]
            headlines = [a["headline"] for a in valid]
            sources   = [a.get("source", "") for a in valid]
            return headlines, sources
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response else "?"
            if status == 429:
                wait = 2 ** attempt
                logging.warning(f"[finbert] Rate-limited for {ticker}, retrying in {wait}s...")
                time.sleep(wait)
            else:
                logging.error(f"[finbert] HTTP {status} for {ticker}: {e}")
                return [], []
        except Exception as e:
            logging.error(f"[finbert] Request failed for {ticker}: {e}")
            return [], []

    logging.error(f"[finbert] All {retries} retries failed for {ticker}.")
    return [], []


def analyze_sentiment(headline: str) -> str:
    """Classify a headline as positive / negative / neutral using VADER compound score."""
    score = _ANALYZER.polarity_scores(headline)["compound"]
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    else:
        return "neutral"


def enrich_with_news(df: pd.DataFrame) -> pd.DataFrame:
    """Adds top news + FinBERT sentiment to LLM output."""
    enriched = []
    total = len(df)
    for i, (_, row) in enumerate(df.iterrows(), 1):
        ticker    = row["ticker"]
        print(f"  [{i}/{total}] {ticker}...")
        news_list, source_list = fetch_news(ticker, retries=1)
        time.sleep(1.5)
        sentiments = [analyze_sentiment(h) for h in news_list]
        enriched.append({
            "ticker":         ticker,
            "llm_summary":    row["llm_summary"],
            "top_news":       news_list,
            "news_sources":   source_list,
            "news_sentiment": sentiments,
        })
    return pd.DataFrame(enriched)


if __name__ == "__main__":
    llm_path = ROOT / "data/explanations/llm_summaries.parquet"
    df_llm   = pd.read_parquet(llm_path)

    enriched_df = enrich_with_news(df_llm)
    out_path    = ROOT / "data/explanations/llm_news_enriched.parquet"
    enriched_df.to_parquet(out_path, index=False)
    print(f"Saved enriched data: {out_path}")
