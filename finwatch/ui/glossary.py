"""Glossary — financial term definitions + HTML tooltip helper."""

GLOSSARY: dict[str, dict[str, str]] = {
    "var_95": {
        "en": "Value at Risk (95%) — maximum expected 1-day loss in 95% of market scenarios",
        "de": "Value at Risk (95%) — max. Tagesverlust in 95% der Marktszenarien",
    },
    "es_95": {
        "en": "Expected Shortfall (95%) — average loss in the worst 5% of scenarios; more conservative than VaR",
        "de": "Expected Shortfall (95%) — Ø-Verlust in den 5% schlechtesten Szenarien; konservativer als VaR",
    },
    "es_ratio": {
        "en": "ES / VaR ratio — values >1.3 signal a heavy-tailed loss distribution (tail risk elevated)",
        "de": "ES / VaR Verhältnis — Werte >1.3 signalisieren ein schweres Verlustverteilungsende",
    },
    "p_drawdown": {
        "en": "Drawdown probability — XGBoost + LightGBM model estimate: chance of a >10% decline in 10 days",
        "de": "Drawdown-Wahrscheinlichkeit — XGBoost + LightGBM Schätzung: Chance auf >10% Rückgang in 10 Tagen",
    },
    "confidence": {
        "en": "Detector consensus — fraction of anomaly models (IsoForest, LSTM-AE, Z-score) that agree",
        "de": "Detektorkonsens — Anteil der Anomalie-Modelle (IsoForest, LSTM-AE, Z-Score) die übereinstimmen",
    },
    "regime": {
        "en": "Market regime — statistical environment derived from returns and volatility: bull / bear / volatile / calm",
        "de": "Marktregime — aus Renditen und Volatilität abgeleitetes Umfeld: Bulle / Bär / volatil / ruhig",
    },
    "vol_regime": {
        "en": "Volatility regime — SPX volatility classification: low (<0.5%) / moderate / high (>1.5%)",
        "de": "Volatilitätsregime — SPX-Klassifikation: niedrig (<0.5%) / moderat / hoch (>1.5%)",
    },
    "anomaly_score": {
        "en": "Composite anomaly score (0–1) — higher = more unusual behavior across all detectors",
        "de": "Anomalie-Score (0–1) — höher = ungewöhnlicheres Verhalten über alle Detektoren",
    },
    "shap": {
        "en": "SHAP value — how much this feature pushed the anomaly score up (+) or down (−) relative to baseline",
        "de": "SHAP-Wert — Beitrag des Merkmals zum Anomalie-Score relativ zur Baseline",
    },
    "rsi": {
        "en": "Relative Strength Index (0–100): >70 overbought (correction likely), <30 oversold (bounce possible)",
        "de": "Relative Stärke Index (0–100): >70 überkauft (Korrektur möglich), <30 überverkauft",
    },
    "ma50": {
        "en": "50-day simple moving average — medium-term trend indicator",
        "de": "50-Tage gleitender Durchschnitt — mittelfristiger Trendindikator",
    },
    "ma200": {
        "en": "200-day simple moving average — long-term trend. Cross above = golden cross; below = death cross",
        "de": "200-Tage Durchschnitt — Langfristtrend. Kreuz nach oben = goldenes Kreuz",
    },
    "drawdown": {
        "en": "Maximum drawdown over 30 days — largest peak-to-trough decline in the last month",
        "de": "Max. Drawdown über 30 Tage — größter Spitze-zu-Tief Verlust im letzten Monat",
    },
    "trading_signal": {
        "en": "AI trading signal derived from severity, drawdown probability, and momentum",
        "de": "KI-Handelssignal aus Schwere, Drawdown-Wahrscheinlichkeit und Momentum",
    },
    "obv_signal": {
        "en": "OBV signal — volume_zscore × return; positive = high volume confirming upward move (institutional buying)",
        "de": "OBV-Signal — Volumen-Z × Rendite; positiv = hohes Volumen bestätigt Aufwärtsbewegung",
    },
    "momentum": {
        "en": "Price momentum — rate of change over 5 or 10 days. Positive = price accelerating upward",
        "de": "Preismomentum — Änderungsrate über 5 oder 10 Tage. Positiv = aufwärts beschleunigend",
    },
    "volume_anomaly_score": {
        "en": "Volume anomaly subscore — driven by volume_zscore and OBV signal",
        "de": "Volumen-Anomalie-Subscore — getrieben von Volumen-Z-Score und OBV-Signal",
    },
    "volatility_anomaly_score": {
        "en": "Volatility anomaly subscore — driven by volatility spike and GARCH-like regime shifts",
        "de": "Volatilitäts-Anomalie-Subscore — Volatilitätsspitzen und Regime-Wechsel",
    },
    "price_anomaly_score": {
        "en": "Price anomaly subscore — driven by momentum, RSI extremes, and moving average crossovers",
        "de": "Preis-Anomalie-Subscore — Momentum, RSI-Extreme und gleitende Durchschnitte",
    },
    "context_anomaly_score": {
        "en": "Context anomaly subscore — driven by market regime, sector correlation, and SPX divergence",
        "de": "Kontext-Anomalie-Subscore — Marktregime, Sektorkorrelation und SPX-Abweichung",
    },
}


def term(key: str, label: str | None = None, lang: str | None = None) -> str:
    """Return an HTML <span> with a CSS tooltip for the given glossary key."""
    if lang is None:
        try:
            import streamlit as st
            lang = st.session_state.get("lang", "en") or "en"
        except Exception:
            lang = "en"
    entry = GLOSSARY.get(key, {})
    tip = entry.get(lang) or entry.get("en", "")
    display = label if label else key.replace("_", " ").upper()
    if not tip:
        return display
    escaped = tip.replace('"', "&quot;").replace("'", "&#39;")
    return f'<span class="tip" data-tip="{escaped}">{display}</span>'
