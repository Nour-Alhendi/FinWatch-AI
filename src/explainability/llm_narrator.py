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
import time
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

MODEL       = "llama-3.3-70b-versatile"
TEMPERATURE = 0.2
MAX_TOKENS  = 600

LANGUAGE_INSTRUCTION = {
    "english": "Respond in English.",
    "german":  "Antworte auf Deutsch.",
    "arabic":  "أجب باللغة العربية.",
}

DEFAULT_SEVERITY_FILTER = {"CRITICAL", "WARNING"}


def _load_detection_latest(ticker: str, detection_dir: Path) -> dict:
    """Load latest row from detection parquet for a ticker."""
    f = detection_dir / f"{ticker}.parquet"
    if not f.exists():
        return {}
    df = pd.read_parquet(f)
    df["Date"] = pd.to_datetime(df["Date"])
    r = df.sort_values("Date").iloc[-1]
    return r.to_dict()


def _build_prompt(row: dict, language: str) -> str:
    lang = LANGUAGE_INSTRUCTION.get(language, LANGUAGE_INSTRUCTION["english"])

    # Technical values
    rsi        = row.get("rsi", 50)
    mom5       = row.get("momentum_5", 0)
    mom10      = row.get("momentum_10", 0)
    vol        = row.get("volatility", 0)
    ret_1d     = row.get("returns", 0)
    drawdown   = row.get("max_drawdown_30d", 0)
    regime     = row.get("regime", "unknown")
    excess     = row.get("excess_return", 0)
    vol_ma20   = row.get("volume_ma20", 1) or 1
    volume     = row.get("Volume", 0)
    vol_ratio  = (volume / vol_ma20) if vol_ma20 > 0 else 1.0
    obv        = row.get("obv_signal", 0)
    confirm    = row.get("confirmation", "neutral")

    # Direction model
    direction  = row.get("direction", "stable")
    p_down     = float(row.get("p_down", 0.33))
    p_up       = max(0.0, 1 - p_down - 0.15)
    mom_sig    = row.get("momentum_signal", "neutral")

    # Anomaly
    anomaly_score = int(row.get("anomaly_score", 0))
    z_anom     = row.get("z_anomaly", False)
    z_anom_60  = row.get("z_anomaly_60", False)
    if_anom    = row.get("if_anomaly", False)
    ae_anom    = row.get("ae_anomaly", False)
    z_score    = row.get("z_score", 0)
    ae_error   = row.get("ae_error", 0)
    mkt_wide   = row.get("is_market_wide", False)
    sec_wide   = row.get("is_sector_wide", False)

    # SHAP / explainability
    driver      = row.get("driver", "unknown")
    top3        = row.get("top3_shap", "")
    narrative   = row.get("narrative", "")
    conflict    = row.get("conflict", "")
    caution     = row.get("caution_flag", "")

    # Derived
    vol_ann = vol * 100 * 16

    detectors = []
    if z_anom:    detectors.append("Z-Score (30D)")
    if z_anom_60: detectors.append("Z-Score (60D)")
    if if_anom:   detectors.append("Isolation Forest")
    if ae_anom:   detectors.append("LSTM Autoencoder")

    scope = ("market-wide" if mkt_wide
             else "sector-wide" if sec_wide
             else "stock-specific")

    anomaly_detected = "Yes" if anomaly_score > 1 else "No"
    models_used      = ", ".join(detectors) if detectors else "None"
    news_sentiment   = row.get("news_sentiment", "neutral")
    existing_summary = row.get("narrative", row.get("summary", ""))

    return f"""You are a sharp equity research analyst. Write short, direct, actionable analysis.
{lang}

RULES:
- MAX 2 sentences per section. No filler words.
- No jargon. Speak like a trader, not a textbook.
- Use ONLY the data below. Do not invent numbers.
- Models only in brackets: e.g. [LSTM-AE, Z-Score].

Structure EXACTLY as follows:

---

**Executive Summary**
One sentence on trend + risk level. One sentence on what the investor should know right now.

---

**Technical Analysis**
RSI + momentum in one sentence. What it signals in one sentence.

---

**Anomaly & Risk Signals**
One sentence: anomaly yes/no and which models [in brackets]. One sentence on what it means.

---

**AI Forecast**
One sentence: direction + probabilities. One sentence: main driver in plain language.

---

**Risk Assessment**
One sentence: risk level + scope (market/sector/stock). One sentence: key risk factor for investor.

---

**Investment Strategy**
TWO lines — both are REQUIRED:
→ Not holding: [specific entry condition, e.g. "Wait for RSI > 40 and momentum turning positive before buying."]
→ Holding: [specific action based on data, e.g. "Trim position — P(down)={p_down*100:.0f}% with falling momentum confirms further downside." OR "Hold — direction stable and no anomaly."]

Base this on the actual data:
- If direction=down AND p_down > 0.50: holding → reduce/trim
- If direction=up AND p_up > 0.50: not holding → look for entry when momentum confirms
- If RSI < 35 and momentum_signal=rising: potential recovery entry signal
- If anomaly_score >= 2: flag as elevated risk for both scenarios
Do NOT write "WAIT" for both. Give different advice for each scenario.

---

DATA:
Ticker: {row['ticker']}
Date: {row.get('date', 'latest')}
Price Change (1D): {ret_1d*100:+.2f}%
RSI: {rsi:.1f}
Momentum (5D / 10D): {mom5:+.3f} / {mom10:+.3f} ($ price change over 5 / 10 days)
Trend / Regime: {regime}
Volume vs 20D avg: {vol_ratio:.1f}x ({confirm})
OBV Signal: {obv:+.3f}
Annualised Volatility: {vol_ann:.1f}%
Max Drawdown 30D: {drawdown*100:.1f}%
Excess Return vs Market: {excess*100:+.2f}%
Anomaly Detected: {anomaly_detected} ({anomaly_score}/4 detectors triggered)
Models Used: {models_used}
Anomaly Scope: {scope}
AI Direction Forecast (5D): {direction.upper()} | P(down)={p_down*100:.0f}% | P(up)={p_up*100:.0f}%
Main Risk Driver: {driver}
Top Contributing Factors: {top3}
Signal Pattern: {narrative}
Risk Level: {row['severity']}
Recommended Action: {row['action']}
Model Confidence: {int(row['confidence']*100)}%
Signal Conflicts: {conflict if conflict else 'None'}
Caution Flags: {caution if caution else 'None'}
News Sentiment: {news_sentiment}
Existing Analysis Summary: {existing_summary}"""


def summarize(row: dict, language: str = "english", retries: int = 4) -> str:
    """Generate full analysis for one ticker. Falls back to narrative_text on failure."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
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
            return response.choices[0].message.content.strip()
        except Exception as e:
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
    detection_dir = ROOT / "data/detection"
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

    results = []
    for _, row in to_process.iterrows():
        ticker   = row["ticker"]
        row_dict = row.to_dict()

        # Enrich with latest detection data
        det = _load_detection_latest(ticker, detection_dir)
        row_dict.update(det)

        # Enrich with full decision data (direction, p_down, momentum_signal, caution_flag)
        if decisions_df is not None:
            dec_rows = decisions_df[decisions_df["ticker"] == ticker]
            if not dec_rows.empty:
                for col in ["direction", "p_down", "momentum_signal", "caution_flag", "summary"]:
                    if col in dec_rows.columns:
                        row_dict[col] = dec_rows.iloc[0][col]

        summary = summarize(row_dict, language=language)
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
