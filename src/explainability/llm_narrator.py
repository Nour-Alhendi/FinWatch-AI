"""
FinWatch AI — Layer 7C: LLM Narrator
======================================
Takes structured Narrative Engine output + detection + decision data
and produces a full professional analysis using Groq (Llama 3.3 70B).

Supported languages: english, german, arabic

Performance: by default only processes CRITICAL + WARNING tickers to stay
within Groq free tier limits (30 req/min). Use severity_filter=None for all.
"""

import os
import json
import time
import logging
import pandas as pd
from pathlib import Path
from datetime import date
from dotenv import load_dotenv
from groq import Groq

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

MODEL       = "llama-3.3-70b-versatile"
TEMPERATURE = 0.2
MAX_TOKENS  = 700   # reduced from 1100 — saves ~35% tokens per call

DAILY_TOKEN_BUDGET = 80_000   # stay safely under Groq free tier (100k/day)
_tokens_used = 0              # tracked across calls in this run

LANGUAGE_INSTRUCTION = {
    "english": "Respond in English.",
    "german":  "Antworte auf Deutsch.",
    "arabic":  "أجب باللغة العربية.",
}

DEFAULT_SEVERITY_FILTER = {"CRITICAL", "WARNING"}

CACHE_PATH = ROOT / "data/explanations/llm_cache.json"


def _load_cache() -> dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text())
    return {}


def _save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2))


def _cache_key(ticker: str) -> str:
    return f"{date.today().isoformat()}_{ticker}"


def _build_prompt(row: dict, language: str) -> str:
    lang = LANGUAGE_INSTRUCTION.get(language, LANGUAGE_INSTRUCTION["english"])

    # ── Read-only: all values must come from upstream layers ──────────────────

    # Features (Layer 3)
    ret_1d        = row.get("returns", 0)
    vol_ratio     = row.get("vol_ratio", 1.0)
    drawdown      = row.get("max_drawdown_30d", 0)

    # Explanation features — must be pre-computed in Layer 3
    current_price   = row.get("current_price")
    ema_20          = row.get("ema_20")
    price_3m_high   = row.get("price_3m_high")
    price_3m_low    = row.get("price_3m_low")
    monthly_summary = row.get("monthly_summary", "N/A")

    if current_price is None:
        raise ValueError(f"[{row.get('ticker')}] Missing current_price — must be computed upstream in Layer 3")
    if ema_20 is None:
        raise ValueError(f"[{row.get('ticker')}] Missing ema_20 — must be computed upstream in Layer 3")
    if price_3m_high is None or price_3m_low is None:
        raise ValueError(f"[{row.get('ticker')}] Missing price_3m_high/low — must be computed upstream in Layer 3")

    ema_diff_pct = row.get("ema_diff_pct")
    if ema_diff_pct is None:
        raise ValueError(f"[{row.get('ticker')}] Missing ema_diff_pct — must be computed upstream in Layer 3")

    # Decision (Layer 6)
    caution        = row.get("caution_flag", "")
    sentiment_note = row.get("sentiment_note", "")

    # Anomaly (Layer 4)
    anomaly_score = int(row.get("anomaly_score", 0))
    z_anom     = row.get("z_anomaly", False)
    z_anom_60  = row.get("z_anomaly_60", False)
    if_anom    = row.get("if_anomaly", False)
    ae_anom    = row.get("ae_anomaly", False)
    mkt_wide   = row.get("is_market_wide", False)
    sec_wide   = row.get("is_sector_wide", False)

    # ── Derived labels only (no numeric invention) ─────────────────────────
    detectors = []
    if z_anom:    detectors.append("Z-Score (30D)")
    if z_anom_60: detectors.append("Z-Score (60D)")
    if if_anom:   detectors.append("Isolation Forest")
    if ae_anom:   detectors.append("LSTM Autoencoder")

    scope            = "market-wide" if mkt_wide else "sector-wide" if sec_wide else "stock-specific"
    anomaly_detected = "Yes" if anomaly_score >= 1 else "No"

    # sentiment_label derived from Layer 6 sentiment_note label
    if "bearish" in sentiment_note:
        sentiment_label = "negative"
    elif "bullish" in sentiment_note:
        sentiment_label = "positive"
    else:
        sentiment_label = "neutral"

    return f"""You are a system that explains financial model outputs in simple language.
You do NOT generate new insights. You ONLY explain the provided data.

{lang}

Write a short, clear report for a non-expert investor.

RULES:
- Use only the provided data
- Do not invent numbers
- Avoid technical jargon (no RSI, EMA, OBV, drawdown probability, annualised volatility)
- Be direct and confident
- Translate all section headers and content to the language specified above

STRUCTURE:

## What is happening
Describe the current situation in 2–3 simple sentences.
- Current price: {current_price:.2f}
- Daily move: {ret_1d*100:+.2f}%
- Position vs recent average: {ema_diff_pct:+.1f}%
- Volume: {"high" if vol_ratio > 1.2 else "low" if vol_ratio < 0.8 else "normal"}

## Why this happened
Explain in 1–2 sentences:
- Scope: {scope}
- Sentiment: {sentiment_label}

## Is this risky?
State clearly:
- Risk level: {row['severity']}

Then give:
- ⚠️ one concern
- ✅ one stabilizing factor

## What to do now
Give ONE clear action in plain language based on signal: {row.get('trading_signal', 'NEUTRAL')}

- EXIT → suggest reducing exposure
- ENTRY → suggest careful buying
- HOLD or NEUTRAL → suggest waiting

## Final Verdict
Write ONE strong sentence summarizing the situation clearly.

End with:
"This is not financial advice."

---

DATA:
Ticker: {row['ticker']}
Date: {row.get('date', 'latest')}
Current Price: {current_price:.2f}
Daily Move: {ret_1d*100:+.2f}%
Position vs Recent Average: {ema_diff_pct:+.1f}%
Volume vs Normal: {vol_ratio:.1f}x
3M High: {price_3m_high:.2f} | 3M Low: {price_3m_low:.2f}
Monthly Price History: {monthly_summary}
Max Drop Last 30 Days: {drawdown*100:.1f}%
Anomaly Detected: {anomaly_detected} ({anomaly_score}/4 models triggered)
Anomaly Scope: {scope}
Risk Level: {row['severity']}
Trading Signal: {row.get('trading_signal', 'NEUTRAL')}
News Sentiment: {sentiment_label}
Caution Flags: {caution if caution else 'None'}"""


def summarize(row: dict, language: str = "english", retries: int = 4) -> str:
    """Generate full analysis for one ticker. Falls back to narrative_text on failure."""
    global _tokens_used

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return row.get("narrative_text", "")

    # Stop if daily budget is nearly exhausted
    if _tokens_used >= DAILY_TOKEN_BUDGET:
        logging.warning(f"[llm_narrator] Daily token budget ({DAILY_TOKEN_BUDGET:,}) reached — skipping {row.get('ticker', '?')}")
        return row.get("narrative_text", "")

    client = Groq(api_key=api_key)
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": _build_prompt(row, language)}],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            used = getattr(response.usage, "total_tokens", MAX_TOKENS + 700)
            _tokens_used += used
            return response.choices[0].message.content.strip()
        except ValueError as e:
            # Data error — retrying won't help
            logging.error(f"[llm_narrator] {row.get('ticker', '?')} data error: {e} — skipping.")
            return row.get("narrative_text", "")
        except Exception as e:
            err_str = str(e)
            if "tokens per day" in err_str or "TPD" in err_str:
                logging.warning(f"[llm_narrator] Daily token limit hit — stopping narrator for today.")
                _tokens_used = DAILY_TOKEN_BUDGET
                return row.get("narrative_text", "")
            wait = 2 ** attempt
            logging.warning(
                f"[llm_narrator] {row.get('ticker', '?')} attempt {attempt+1} failed: {e} "
                f"— retrying in {wait}s"
            )
            time.sleep(wait)

    logging.error(f"[llm_narrator] All retries failed for {row.get('ticker', '?')}, using fallback.")
    return row.get("narrative_text", "")


def run(
    explanations_path: str,
    language: str = "english",
    severity_filter=DEFAULT_SEVERITY_FILTER,
) -> pd.DataFrame:
    """
    Run LLM Narrator on explanations.parquet.
    Enriches each row with detection + decision data before calling the LLM.

    Args:
        explanations_path: path to explanations.parquet
        language:          "english" | "german" | "arabic"
        severity_filter:   only process these severities (None = all tickers)

    Returns:
        DataFrame with ticker + llm_summary columns.
    """
    df           = pd.read_parquet(explanations_path)
    decisions_df  = None
    decisions_path = ROOT / "data/decisions/decisions.parquet"
    if decisions_path.exists():
        decisions_df = pd.read_parquet(decisions_path)

    if severity_filter:
        to_process = df[df["severity"].isin(severity_filter)]
        skipped    = df[~df["severity"].isin(severity_filter)].copy()
        skipped["llm_summary"] = skipped["narrative_text"]
    else:
        to_process = df
        skipped    = pd.DataFrame()

    print(f"\nLLM Narrator — {language.upper()}  |  model: {MODEL}")
    print(f"Processing {len(to_process)} tickers (filter: {severity_filter or 'all'})")
    print("=" * 65)

    cache   = _load_cache()
    results = []
    for _, row in to_process.iterrows():
        ticker   = row["ticker"]
        key      = _cache_key(ticker)

        # Cache hit — skip Groq entirely
        if key in cache:
            print(f"\n{ticker} [{row['severity']}] (cached)")
            results.append({"ticker": ticker, "llm_summary": cache[key]})
            continue

        row_dict = row.to_dict()

        # Enrich with decision data — only what LLM actually needs
        if decisions_df is not None:
            dec_rows = decisions_df[decisions_df["ticker"] == ticker]
            if not dec_rows.empty:
                for col in ["caution_flag", "sentiment_note", "trading_signal"]:
                    if col in dec_rows.columns:
                        row_dict[col] = dec_rows.iloc[0][col]

        summary = summarize(row_dict, language=language)
        cache[key] = summary
        _save_cache(cache)

        results.append({"ticker": ticker, "llm_summary": summary})
        print(f"\n{ticker} [{row['severity']}]")
        print(f"  {summary[:120]}...")
        time.sleep(2)   # respect Groq rate limit

    result_df = pd.DataFrame(results)

    if not skipped.empty:
        skipped_df = skipped[["ticker", "llm_summary"]].reset_index(drop=True)
        result_df  = pd.concat([result_df, skipped_df], ignore_index=True)

    out_path = ROOT / "data/explanations/llm_summaries.parquet"
    result_df.to_parquet(out_path, index=False)
    print(f"\nSaved: {out_path}")

    return result_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--language", default="english",
                        choices=["english", "german", "arabic"])
    parser.add_argument("--all", action="store_true",
                        help="Process all tickers, not just CRITICAL/WARNING")
    args = parser.parse_args()

    run(
        explanations_path=str(ROOT / "data/explanations/explanations.parquet"),
        language=args.language,
        severity_filter=None if args.all else DEFAULT_SEVERITY_FILTER,
    )
