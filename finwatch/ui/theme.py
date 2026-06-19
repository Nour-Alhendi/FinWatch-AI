"""
FinWatch UI — shared label mappings.
Import feat_label / anomaly_label wherever raw pipeline names must be shown to the user.
Fallback: name.replace("_", " ").title()
"""
from __future__ import annotations

_FEATURE_LABELS: dict[str, dict[str, str]] = {
    "vix_level":                {"en": "VIX Level",          "de": "VIX Level"},
    "spx_return":               {"en": "S&P 500 Return",     "de": "S&P 500 Rendite"},
    "macd_hist":                {"en": "MACD Histogram",     "de": "MACD Histogramm"},
    "beta":                     {"en": "Beta",               "de": "Beta"},
    "trend_strength":           {"en": "Trend Strength",     "de": "Trendstärke"},
    "price_vs_ma200":           {"en": "Price vs. MA200",    "de": "Preis vs. MA200"},
    "price_vs_ma200_stock":     {"en": "Price vs. MA200",    "de": "Preis vs. MA200"},
    "rolling_std_60":           {"en": "Volatility (60d)",   "de": "Volatilität (60d)"},
    "max_drawdown_30d":         {"en": "Max. Drawdown 30d",  "de": "Max. Drawdown 30d"},
    "volume_zscore":            {"en": "Volume Z-Score",     "de": "Volumen Z-Score"},
    "volatility_ratio":         {"en": "Volatility Ratio",   "de": "Volatilitäts-Ratio"},
    "momentum_5":               {"en": "Momentum 5d",        "de": "Momentum 5d"},
    "momentum_10":              {"en": "Momentum 10d",       "de": "Momentum 10d"},
    "rsi":                      {"en": "RSI",                "de": "RSI"},
    "obv_signal":               {"en": "OBV Signal",         "de": "OBV Signal"},
    "corr_spx":                 {"en": "SPX Correlation",    "de": "Korrelation S&P 500"},
    "excess_return":            {"en": "Excess Return",      "de": "Excess Return"},
    "es_ratio":                 {"en": "ES Ratio",           "de": "ES Ratio"},
    "context_anomaly_score":    {"en": "Context",      "de": "Kontext"},
    "volume_anomaly_score":     {"en": "Volume",       "de": "Volumen"},
    "volatility_anomaly_score": {"en": "Volatility",   "de": "Volatilität"},
    "price_anomaly_score":      {"en": "Price",        "de": "Preis"},
    "ma50":                     {"en": "MA50",               "de": "MA50"},
    "ma200":                    {"en": "MA200",              "de": "MA200"},
    "drawdown":                 {"en": "Drawdown 30d",       "de": "Drawdown 30d"},
    "anomaly_score":            {"en": "Anomaly Score",      "de": "Anomalie-Score"},
    "p_drawdown":               {"en": "P(Drawdown)",        "de": "P(Drawdown)"},
    "var_95":                   {"en": "VaR 95%",            "de": "VaR 95%"},
    "es_95":                    {"en": "ES 95%",             "de": "ES 95%"},
    "confidence":               {"en": "Confidence",         "de": "Konfidenz"},
}

_ANOMALY_LABELS: dict[str, dict[str, str]] = {
    "volatility_risk":   {"en": "Volatility Risk",      "de": "Volatilitätsrisiko"},
    "volume_flow":       {"en": "Unusual Trading Flow", "de": "Ungewöhnlicher Handelsfluss"},
    "price_breakout":    {"en": "Price Breakout",       "de": "Preisausbruch"},
    "context_shift":     {"en": "Market Context Shift", "de": "Marktkontext-Verschiebung"},
    "unknown":           {"en": "Unknown",              "de": "Unbekannt"},
    "normal":            {"en": "Normal",               "de": "Normal"},
    "NORMAL":            {"en": "Normal",               "de": "Normal"},
}


# ── Muted text color palette — single source of truth ──────────────────────
# All values contrast-tested on #0a0a0f / #111118 backgrounds.
TEXT_MUTED    = "#8080a4"   # section labels, captions, dim text (5:1 contrast)
CHART_AXIS    = "#7a9ab0"   # chart axis/tick labels, empty state hints
COLOR_NEUTRAL = "#7a95ab"   # neutral semantic signal / fallback state


def _lang() -> str:
    try:
        import streamlit as st
        return st.session_state.get("lang", "en") or "en"
    except Exception:
        return "en"


# Flat English dicts kept for any code that imports and iterates them directly
FEATURE_LABELS: dict[str, str] = {k: v["en"] for k, v in _FEATURE_LABELS.items()}
ANOMALY_LABELS: dict[str, str] = {k: v["en"] for k, v in _ANOMALY_LABELS.items()}


def feat_label(raw: str) -> str:
    """Return a human-readable feature label in the current UI language."""
    lang = _lang()
    entry = _FEATURE_LABELS.get(raw)
    if entry:
        return entry.get(lang) or entry["en"]
    return raw.replace("_", " ").title()


def anomaly_label(raw: str) -> str:
    """Return a human-readable anomaly label in the current UI language."""
    lang = _lang()
    key = raw.replace(" anomaly", "").strip()
    entry = _ANOMALY_LABELS.get(key)
    if entry:
        return entry.get(lang) or entry["en"]
    return key.replace("_", " ").title()
