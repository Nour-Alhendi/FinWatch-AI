"""
FinWatch AI — Layer 6: Decision Engine (v3)
============================================
Clean, p_drawdown-first design. Anomaly as context/confirmation only.

Changes from v2:
  1. WARNING trigger: anomaly removed — only p_drawdown decides
     Anomalie bleibt als Context-Flag (bestätigt oder widerspricht)
  2. REVIEW block removed — anomaly_w alone no longer overrides severity
  3. CRITICAL: anomaly_is_bearish bleibt als Doppel-Check (unverändert)

Primary signals (in order of reliability):
  1. p_drawdown              — ML probability of >5% drawdown in 20 days (AUC 0.715)
  2. drawdown (30d actual)   — already happened, hard fact
  3. anomaly_score_weighted  — 4-model ensemble (context/confirmation only)
  4. RSI + momentum          — technical confirmation
  5. news_sentiment_score    — soft signal (Groq contextual)
  6. excess_return           — vs market (filters false positives)

Severity Levels:
  CRITICAL         — Strong p_drawdown + anomaly confirmed
  WARNING          — Elevated p_drawdown signal
  WATCH            — Single weak signal, monitor
  NORMAL           — No significant signals
  POSITIVE_SIGNAL  — Low drawdown risk + positive technicals
"""

from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
import pandas as pd

_ROOT           = Path(__file__).resolve().parents[2]
_PRECISION_PATH = _ROOT / "data/backtesting/signal_precision.parquet"

_HISTORICAL_PRECISION: dict = {}

def _load_precision():
    global _HISTORICAL_PRECISION
    if not _PRECISION_PATH.exists():
        return
    df = pd.read_parquet(_PRECISION_PATH)
    mp = dict(zip(df["signal"], df["precision"]))
    _HISTORICAL_PRECISION = {
        "CRITICAL":        mp.get("severity_critical", 0.48),
        "WARNING":         max(mp.get("severity_warning", 0.36) - mp.get("severity_critical", 0.48) * 0.35, 0.30),
        "WATCH":           0.28,
        "NORMAL":          1.0 - mp.get("severity_warning", 0.36),
        "POSITIVE_SIGNAL": 1.0 - mp.get("severity_warning", 0.36),
    }

_load_precision()


# ── Thresholds ─────────────────────────────────────────────────────────────

P_DRAWDOWN_CRITICAL = 0.60
P_DRAWDOWN_WARNING  = 0.45
P_DRAWDOWN_LOW      = 0.30

ANOMALY_STRONG  = 0.50
ANOMALY_MEDIUM  = 0.30
ANOMALY_WEAK    = 0.20

DRAWDOWN_WARNING_CAP  = -0.08
DRAWDOWN_CRITICAL_CAP = -0.15

RSI_OVERBOUGHT = 70
RSI_OVERSOLD   = 30

ES_RATIO_HIGH = 2.0

SENTIMENT_NEGATIVE = -0.10
SENTIMENT_POSITIVE = +0.10


# ── Data Classes ────────────────────────────────────────────────────────────

@dataclass
class AnomalyInput:
    ticker:                 str
    date:                   str

    p_drawdown:             float = 0.35
    drawdown_risk:          str   = "low"

    anomaly_score:          int   = 0
    anomaly_score_weighted: float = 0.0
    market_anomaly:         bool  = False
    sector_anomaly:         bool  = False

    rsi:            float = 50.0
    momentum_5:     float = 0.0
    momentum_10:    float = 0.0
    drawdown:       float = 0.0
    obv_signal:     float = 0.0
    volatility:     float = 0.02
    excess_return:  float = 0.0
    es_ratio:       float = 1.0
    vix_level:      float = 20.0

    vader_score:          float = 0.0
    finbert_score:        float = 0.0
    news_sentiment_score: float = 0.0

    days_to_next_earnings: Optional[int]   = None
    insider_sentiment:     float           = 0.0
    put_call_ratio:        float           = 0.85
    options_fear:          int             = 0

    pe_ratio:       Optional[float] = None
    pe_forward:     Optional[float] = None
    pb_ratio:       Optional[float] = None
    revenue_growth: Optional[float] = None

    price_vs_ma200: float = 0.0
    price_vs_ma50:  float = 0.0
    regime:         str   = "unknown"
    volume_trend:   float = 1.0
    trend_strength: float = 0.0

    vix_change:          float = 0.0
    price_vs_ema20:      float = 0.0
    golden_cross:        bool  = False
    rsi_oversold_bounce: bool  = False
    macd_cross_bullish:  bool  = False
    hh_hl:               bool  = False
    volume_breakout:     bool  = False

    death_cross:               bool = False
    ll_lh:                     bool = False
    macd_cross_bearish:        bool = False
    panic_volume:              bool = False
    volume_spike_no_recovery:  bool = False
    rsi_below_50_high_vol:     bool = False
    rsi_oversold_no_bounce:    bool = False

    low_volume:               bool = False
    consolidation:            bool = False
    consolidation_above_ma50: bool = False
    rsi_divergence_bullish:   bool = False

    anomaly_type:            str   = "unknown"
    volume_flow_score:       float = 0.0
    volatility_risk_score:   float = 0.0
    price_momentum_score:    float = 0.0
    market_context_score:    float = 0.0


@dataclass
class DecisionOutput:
    ticker:           str
    date:             str
    severity:         str
    action:           str
    confidence:       float
    context:          str

    p_drawdown:             float = 0.0
    anomaly_score:          int   = 0
    anomaly_score_weighted: float = 0.0
    drawdown_risk:          str   = "low"
    momentum_signal:        str   = "neutral"
    caution_flag:           bool  = False
    override_reason:        str   = ""
    summary:                str   = ""
    sentiment_note:         str   = ""
    trading_signal:         str   = "NEUTRAL"
    anomaly_type:           str   = "unknown"


# ── Core Logic ──────────────────────────────────────────────────────────────

def _dynamic_drawdown_threshold(volatility: float) -> tuple[float, float]:
    warning  = max(DRAWDOWN_WARNING_CAP,  -1.0 * volatility * 20)
    critical = max(DRAWDOWN_CRITICAL_CAP, -2.0 * volatility * 20)
    return warning, critical


def _momentum_label(mom5: float, mom10: float) -> str:
    if mom5 > 0.02 and mom10 > 0.01:   return "positive"
    if mom5 < -0.02 and mom10 < -0.01: return "negative"
    if mom5 < -0.02 and mom10 > 0.01:  return "pullback"
    if mom5 > 0.02 and mom10 < -0.01:  return "bounce"
    return "neutral"


def _confidence(inp: AnomalyInput, severity: str) -> float:
    is_risk = severity in ("CRITICAL", "WARNING", "WATCH")

    if _HISTORICAL_PRECISION:
        base = _HISTORICAL_PRECISION.get(severity, 0.35)

        agree = 0.0
        if is_risk:
            agree += 0.05 * (inp.p_drawdown - P_DRAWDOWN_WARNING) / (1 - P_DRAWDOWN_WARNING)
            agree += 0.03 * inp.anomaly_score_weighted
            if inp.news_sentiment_score <= SENTIMENT_NEGATIVE:
                agree += 0.02
        else:
            agree += 0.05 * (P_DRAWDOWN_LOW - inp.p_drawdown) / P_DRAWDOWN_LOW
            agree += 0.03 * (1 - inp.anomaly_score_weighted)
            if inp.news_sentiment_score >= SENTIMENT_POSITIVE:
                agree += 0.02

        return round(min(max(base + agree, 0.10), 0.95), 2)

    signals = 0.0
    total   = 9.0

    if is_risk:
        signals += 3 * inp.p_drawdown
    else:
        signals += 3 * (1 - inp.p_drawdown)

    if is_risk:
        signals += 3 * inp.anomaly_score_weighted
    else:
        signals += 3 * (1 - inp.anomaly_score_weighted)

    dd_warn, _ = _dynamic_drawdown_threshold(inp.volatility)
    if is_risk and inp.drawdown <= dd_warn:
        signals += 2
    elif not is_risk and inp.drawdown > dd_warn * 0.5:
        signals += 2

    if is_risk and inp.news_sentiment_score <= SENTIMENT_NEGATIVE:
        signals += 1
    elif not is_risk and inp.news_sentiment_score >= SENTIMENT_POSITIVE:
        signals += 1
    else:
        signals += 0.5

    return round(signals / total, 2)


def _vix_thresholds(vix: float) -> tuple[float, float, float]:
    if vix < 15:
        return 0.55, 0.68, 0.42
    elif vix < 20:
        return 0.50, 0.63, 0.40
    elif vix < 25:
        return P_DRAWDOWN_WARNING, P_DRAWDOWN_CRITICAL, 0.38
    else:
        return P_DRAWDOWN_WARNING, P_DRAWDOWN_CRITICAL, 0.35


def decide(inp: AnomalyInput) -> DecisionOutput:
    """
    Core decision logic.
    v3 changes:
      - WARNING: nur p_drawdown entscheidet (anomaly raus)
      - Anomalie als Context-Flag: bestätigt oder widerspricht p_drawdown
      - REVIEW block entfernt
      - CRITICAL: anomaly_is_bearish bleibt als Doppel-Check
    """
    dd_warn, dd_crit        = _dynamic_drawdown_threshold(inp.volatility)
    mom_label               = _momentum_label(inp.momentum_5, inp.momentum_10)
    anomaly_w               = inp.anomaly_score_weighted
    p_dd                    = inp.p_drawdown
    caution                 = False
    override_reason         = ""

    p_warn, p_crit, p_watch = _vix_thresholds(inp.vix_level)

    # ── Anomalie-Kontext: bestätigt oder widerspricht p_drawdown ──────────────
    # Wird als Context-String verwendet, entscheidet NICHT über Severity
    anomaly_context = ""
    if anomaly_w >= ANOMALY_MEDIUM:
        # Anomalie bestätigt Risiko-Signal
        anomaly_context = " | Anomalie bestätigt erhöhte Marktspannung"
    elif anomaly_w >= ANOMALY_WEAK and anomaly_w < ANOMALY_MEDIUM:
        # Schwache Anomalie — nur Hinweis
        anomaly_context = " | schwaches Anomalie-Signal — beobachten"

    # Option B: Anomalie ohne ML-Bestätigung = unbekanntes Muster
    # Wenn Anomalie hoch aber p_drawdown noch niedrig → Frühwarnung
    anomaly_early_warning = (
        anomaly_w >= ANOMALY_MEDIUM
        and p_dd < P_DRAWDOWN_WARNING
    )

    # ── CRITICAL: p_drawdown + anomaly als Doppel-Check (unverändert) ─────────
    anomaly_is_bearish = (
        anomaly_w >= ANOMALY_MEDIUM
        and (inp.excess_return < 0 or inp.momentum_5 < -0.02 or inp.drawdown < dd_warn * 0.5)
    )

    if p_dd >= p_crit and anomaly_is_bearish:
        severity = "CRITICAL"
        action   = "ESCALATE"
        context  = "high drawdown probability + anomaly confirmed"

    elif inp.drawdown <= dd_crit:
        severity = "CRITICAL"
        action   = "ESCALATE"
        context  = "severe drawdown in progress"
        override_reason = f"drawdown={inp.drawdown:.1%} ≤ {dd_crit:.1%}"

    # ── WARNING: NUR p_drawdown entscheidet (ÄNDERUNG v3) ────────────────────
    elif p_dd >= p_warn:
        severity = "WARNING"
        action   = "MONITOR"
        context  = f"p_drawdown={p_dd:.0%}"
        # Anomalie als Kontext hinzufügen — entscheidet aber nicht
        context += anomaly_context

    elif inp.drawdown <= dd_warn:
        severity = "WARNING"
        action   = "MONITOR"
        context  = f"drawdown={inp.drawdown:.1%}"
        override_reason = "actual drawdown threshold"

    # ── WATCH: schwaches Signal ────────────────────────────────────────────────
    elif anomaly_w >= ANOMALY_WEAK or p_dd >= p_watch:
        severity = "WATCH"
        action   = "OBSERVE"
        context  = "weak signal — monitor"
        # Option B: Anomalie früher als p_drawdown → expliziter Hinweis
        if anomaly_early_warning:
            context = "Anomalie ohne ML-Bestätigung — unbekanntes Muster möglich"

    elif (p_dd < P_DRAWDOWN_LOW
          and inp.rsi < RSI_OVERBOUGHT
          and mom_label in ("positive", "pullback")
          and anomaly_w < ANOMALY_WEAK):
        severity = "POSITIVE_SIGNAL"
        action   = "NONE"
        context  = f"low drawdown risk ({p_dd:.0%}) + {mom_label} momentum"

    else:
        severity = "NORMAL"
        action   = "NONE"
        context  = "no significant signals"

    # ── Bearish technical confirmation ─────────────────────────────────────────
    bearish = []
    if inp.death_cross:              bearish.append("death cross")
    if inp.ll_lh:                    bearish.append("LH+LL")
    if inp.macd_cross_bearish:       bearish.append("MACD bearish cross")
    if inp.panic_volume:             bearish.append("panic selling")
    if inp.volume_spike_no_recovery: bearish.append("vol spike no recovery")
    if inp.rsi_oversold_no_bounce:   bearish.append("RSI still declining")
    if inp.rsi_below_50_high_vol:    bearish.append("RSI<50 high vol")
    if inp.price_vs_ma200 < 0 and inp.price_vs_ma50 < 0:
                                     bearish.append("below MA50+MA200")
    if inp.vix_change > 0.10:        bearish.append(f"VIX spike +{inp.vix_change:.0%}")

    if bearish:
        context += f" | bearish: {', '.join(bearish[:3])}"
        if len(bearish) >= 3 and severity == "WATCH":
            severity = "WARNING"
            action   = "MONITOR"
            override_reason = f"{len(bearish)} bearish signals converging"
        if inp.death_cross and severity == "NORMAL":
            severity = "WATCH"
            action   = "OBSERVE"

    # ── Neutral / consolidation zone ───────────────────────────────────────────
    rsi_neutral = 45 <= inp.rsi <= 55
    neutral = []
    if rsi_neutral:        neutral.append("RSI neutral 45-55")
    if inp.low_volume:     neutral.append("low volume")
    if inp.consolidation and not inp.consolidation_above_ma50:
        neutral.append("consolidating below MA50")
    elif inp.consolidation and inp.consolidation_above_ma50:
        context += " | bullish coil above MA50"

    if neutral and severity == "NORMAL":
        if len(neutral) >= 2:
            severity = "WATCH"
            action   = "OBSERVE"
            override_reason = "sideways/uncertain market"
        context += f" | {', '.join(neutral)}"
    elif neutral and severity == "WATCH":
        context += f" | {', '.join(neutral)}"

    # ── Caution flags ──────────────────────────────────────────────────────────
    if inp.rsi >= RSI_OVERBOUGHT and p_dd >= 0.40:
        caution = True
        context += " | overbought RSI"

    if inp.es_ratio >= ES_RATIO_HIGH and severity not in ("CRITICAL",):
        if severity == "NORMAL":
            severity = "WATCH"
            action   = "OBSERVE"
        context += f" | tail risk (ES={inp.es_ratio:.1f})"

    if severity == "CRITICAL" and inp.market_anomaly and inp.excess_return > -0.03:
        severity = "WARNING"
        action   = "MONITOR"
        override_reason = "market-wide event, stock not underperforming"
        context  = "broad market stress (not stock-specific)"

    if (severity == "WARNING"
            and inp.excess_return > 0.02
            and p_dd < p_crit
            and anomaly_w < ANOMALY_STRONG):
        severity = "WATCH"
        action   = "OBSERVE"
        override_reason = "stock outperforming market — reduced false positive risk"
        context  += " | outperforming market (likely upward anomaly)"

    # ── Sentiment ──────────────────────────────────────────────────────────────
    sentiment_note = ""
    if inp.news_sentiment_score <= SENTIMENT_NEGATIVE:
        if severity in ("WARNING", "WATCH"):
            severity = "WARNING"
            context += " | negative news confirms"
        sentiment_note = f"bearish news ({inp.news_sentiment_score:+.2f})"
    elif inp.news_sentiment_score >= SENTIMENT_POSITIVE:
        if severity == "POSITIVE_SIGNAL":
            context += " | positive news confirms"
        sentiment_note = f"bullish news ({inp.news_sentiment_score:+.2f})"

    # ── Fundamental signals ────────────────────────────────────────────────────
    # Earnings: nur Hinweis, kein Severity-Einfluss (wie besprochen)
    if inp.days_to_next_earnings is not None and inp.days_to_next_earnings <= 7:
        context += f" | earnings in {inp.days_to_next_earnings}d — interpret signals with caution"

    if inp.insider_sentiment <= -0.3:
        if severity in ("WATCH", "WARNING"):
            severity = "WARNING"
            action   = "MONITOR"
        context += f" | insider selling ({inp.insider_sentiment:+.2f})"

    if inp.options_fear:
        context += f" | options fear (P/C={inp.put_call_ratio:.2f})"
        if inp.put_call_ratio > 1.5 and severity == "WATCH":
            severity = "WARNING"
            action   = "MONITOR"
            override_reason = f"extreme options fear (P/C={inp.put_call_ratio:.2f})"
        elif inp.put_call_ratio > 1.2 and severity == "WARNING":
            severity = "CRITICAL"
            action   = "EXIT"
            override_reason = f"options fear confirms WARNING → CRITICAL (P/C={inp.put_call_ratio:.2f})"

    # ── REVIEW: ENTFERNT (v3) ─────────────────────────────────────────────────
    # Anomalie allein überschreibt keine Severity mehr.
    # Anomalie-ohne-ML-Bestätigung wird als WATCH mit Context-Flag behandelt (siehe oben).

    # ── Regime-aware adjustments ───────────────────────────────────────────────
    if inp.regime == "bear":
        if severity == "WATCH":
            severity = "WARNING"
            action   = "MONITOR"
            context += " | bear market (elevated risk)"
        elif severity in ("NORMAL", "POSITIVE_SIGNAL"):
            caution = True
            context += " | bear market regime"

    if inp.regime == "transition_down" and severity == "POSITIVE_SIGNAL":
        caution = True
        context += " | market transitioning down"

    if inp.price_vs_ma200 < -0.10 and severity in ("NORMAL", "POSITIVE_SIGNAL"):
        caution = True
        context += f" | below MA200 ({inp.price_vs_ma200:+.1%})"

    # ── Valuation ──────────────────────────────────────────────────────────────
    valuation_note = ""

    if inp.pe_ratio is not None and inp.pe_ratio < 0:
        valuation_note = "negative earnings"
        if severity == "POSITIVE_SIGNAL":
            severity = "NORMAL"
            action   = "NONE"
            context += " | negative earnings (no entry)"

    elif inp.pe_ratio is not None and inp.pe_ratio > 50:
        valuation_note = f"overvalued P/E={inp.pe_ratio:.0f}"
        if severity == "POSITIVE_SIGNAL":
            severity = "NORMAL"
            action   = "NONE"
            context += f" | overvalued (P/E={inp.pe_ratio:.0f}, no entry)"

    elif ((inp.pe_ratio is not None and inp.pe_ratio < 15)
          or (inp.pb_ratio is not None and inp.pb_ratio < 1.5)):
        valuation_note = (
            f"cheap P/E={inp.pe_ratio:.0f}" if inp.pe_ratio is not None and inp.pe_ratio < 15
            else f"cheap P/B={inp.pb_ratio:.2f}"
        )

    if inp.revenue_growth is not None and inp.revenue_growth < -0.05:
        valuation_note += f" rev{inp.revenue_growth:+.0%}"
        if severity in ("WATCH", "WARNING", "CRITICAL"):
            context += f" | revenue declining ({inp.revenue_growth:+.0%})"

    if (inp.pe_forward is not None and inp.pe_ratio is not None
            and inp.pe_forward > inp.pe_ratio * 1.2
            and severity == "POSITIVE_SIGNAL"):
        context += f" | earnings contraction expected (fwdPE={inp.pe_forward:.0f})"

    # ── Anomaly type label — appended to context if known ─────────────────────
    if inp.anomaly_type != "unknown":
        context += f" | dominant anomaly: {inp.anomaly_type}"

    # ── Trading Signal ─────────────────────────────────────────────────────────
    overbought_exit = (
        inp.rsi > 78
        and mom_label in ("positive", "bounce")
    )

    if severity == "CRITICAL":
        trading_signal = "EXIT"
        exit_reason    = "risk"

    elif severity == "WARNING":
        recovering    = inp.momentum_5 > 0.03 or inp.rsi_divergence_bullish
        strong_signal = p_dd >= 0.50 or anomaly_w >= 0.35

        if strong_signal and not recovering:
            trading_signal = "EXIT"
            exit_reason    = "risk"
        elif strong_signal and recovering:
            trading_signal = "HOLD"
            exit_reason    = ""
            note = "RSI divergence — potential reversal" if inp.rsi_divergence_bullish else "recovering momentum"
            context += f" | {note} — hold, monitor closely"
        else:
            trading_signal = "HOLD"
            exit_reason    = ""

    elif overbought_exit and severity not in ("CRITICAL", "WARNING"):
        trading_signal = "EXIT"
        exit_reason    = "overbought"
        context += f" | overbought RSI={inp.rsi:.0f}, +{inp.price_vs_ma200:.0%} above MA200 → take profits"

    elif severity == "WATCH":
        trading_signal = "HOLD"
        exit_reason    = ""

    elif severity == "POSITIVE_SIGNAL":
        exit_reason  = ""
        above_ma200  = inp.price_vs_ma200 > -0.10
        regime_ok    = inp.regime not in ("bear", "transition_down")
        valuation_ok = inp.pe_ratio is None or (inp.pe_ratio > 0 and inp.pe_ratio < 60)
        revenue_ok   = inp.revenue_growth is None or inp.revenue_growth >= -0.10

        bullish = []
        if inp.golden_cross:                bullish.append("golden cross")
        if inp.price_vs_ema20 > 0:          bullish.append("above EMA20")
        if inp.rsi_oversold_bounce:         bullish.append("RSI bounce")
        if inp.rsi_divergence_bullish:      bullish.append("RSI divergence")
        if inp.macd_cross_bullish:          bullish.append("MACD cross")
        if inp.hh_hl:                       bullish.append("HH+HL")
        if inp.volume_breakout:             bullish.append("vol breakout")
        if inp.consolidation_above_ma50:    bullish.append("bullish coil")
        if inp.obv_signal > 0:              bullish.append("OBV buying")
        if 50 <= inp.rsi <= 70:             bullish.append("RSI bullish zone")
        if inp.price_vs_ma50 > 0:          bullish.append("above MA50")
        bullish_count = len(bullish)

        if above_ma200 and regime_ok and valuation_ok and revenue_ok and bullish_count >= 2:
            trading_signal = "ENTRY"
            context += f" | {bullish_count} bullish signals: {', '.join(bullish[:3])}"
        else:
            trading_signal = "HOLD"
            if bullish_count > 0:
                context += f" | bullish technicals developing ({bullish_count}: {', '.join(bullish[:2])})"

    elif severity == "NORMAL":
        exit_reason    = ""
        trading_signal = "NEUTRAL"

    else:
        exit_reason    = ""
        trading_signal = "NEUTRAL"

    confidence = _confidence(inp, severity)

    # ── Summary ────────────────────────────────────────────────────────────────
    summary_parts = []
    summary_parts.append(f"P(drawdown>5% / 20d) = {p_dd:.0%}")
    summary_parts.append(f"Anomaly = {anomaly_w:.2f}")
    summary_parts.append(f"RSI = {inp.rsi:.0f}")
    if inp.drawdown < -0.03:
        summary_parts.append(f"30d drawdown = {inp.drawdown:.1%}")
    if sentiment_note:
        summary_parts.append(sentiment_note)
    if inp.insider_sentiment <= -0.3:
        summary_parts.append(f"insiders selling ({inp.insider_sentiment:+.2f})")
    if inp.days_to_next_earnings is not None and inp.days_to_next_earnings <= 7:
        summary_parts.append(f"earnings in {inp.days_to_next_earnings}d")
    if inp.options_fear:
        summary_parts.append(f"options fear (P/C={inp.put_call_ratio:.2f})")
    if valuation_note:
        summary_parts.append(valuation_note)
    summary = " | ".join(summary_parts)

    return DecisionOutput(
        ticker=inp.ticker, date=inp.date,
        severity=severity, action=action,
        confidence=confidence, context=context,
        p_drawdown=p_dd,
        anomaly_score=inp.anomaly_score,
        anomaly_score_weighted=anomaly_w,
        drawdown_risk=inp.drawdown_risk,
        momentum_signal=mom_label,
        caution_flag=caution,
        override_reason=override_reason,
        summary=summary,
        sentiment_note=sentiment_note,
        trading_signal=trading_signal,
        anomaly_type=inp.anomaly_type,
    )


def run_decision_engine(records: list) -> list:
    """Process list of dicts → list of DecisionOutput."""
    results = []
    for r in records:
        inp = AnomalyInput(**{
            k: v for k, v in r.items()
            if k in AnomalyInput.__dataclass_fields__
        })
        results.append(decide(inp))
    return results
