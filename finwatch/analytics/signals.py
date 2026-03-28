def calculate_risk_drivers(r: dict, row: dict) -> list:
    """
    Returns a list of (level, reason) tuples explaining WHY risk is elevated.
    level: "high" | "medium" | "low"
    Checks technical indicators, anomaly detectors, AI forecast, and market context.
    """
    drivers = []

    # ── Anomaly detectors ──────────────────────────────────────────────────────
    n_anom = sum([
        bool(r.get("z_anomaly")),
        bool(r.get("z_anomaly_60")),
        bool(r.get("if_anomaly")),
        bool(r.get("ae_anomaly")),
    ])
    if n_anom >= 3:
        drivers.append(("high",   f"{n_anom}/4 anomaly detectors triggered simultaneously"))
    elif n_anom == 2:
        drivers.append(("medium", f"{n_anom}/4 anomaly detectors triggered"))
    elif n_anom == 1:
        drivers.append(("low",    "1/4 anomaly detector triggered"))

    # ── RSI extremes ───────────────────────────────────────────────────────────
    rsi = r.get("rsi", 50)
    if rsi > 75:
        drivers.append(("high",   f"Overbought conditions (RSI {rsi:.1f} — correction risk)"))
    elif rsi > 70:
        drivers.append(("medium", f"Approaching overbought (RSI {rsi:.1f})"))
    elif rsi < 25:
        drivers.append(("high",   f"Extreme oversold (RSI {rsi:.1f} — potential panic selling)"))
    elif rsi < 30:
        drivers.append(("medium", f"Oversold territory (RSI {rsi:.1f})"))

    # ── Volatility spike ───────────────────────────────────────────────────────
    vol = r.get("volatility", 0)
    vol_ann = vol * 100 * 16
    if vol_ann > 60:
        drivers.append(("high",   f"Extreme volatility spike — annualised {vol_ann:.0f}%"))
    elif vol_ann > 40:
        drivers.append(("high",   f"Elevated volatility — annualised {vol_ann:.0f}%"))
    elif vol_ann > 25:
        drivers.append(("medium", f"Above-average volatility — annualised {vol_ann:.0f}%"))

    # ── Drawdown ───────────────────────────────────────────────────────────────
    drawdown = r.get("max_drawdown_30d", 0) or 0
    if abs(drawdown) > 0.20:
        drivers.append(("high",   f"Severe 30D drawdown: {drawdown*100:.1f}%"))
    elif abs(drawdown) > 0.10:
        drivers.append(("high",   f"Sharp 30D drawdown: {drawdown*100:.1f}%"))
    elif abs(drawdown) > 0.05:
        drivers.append(("medium", f"Moderate 30D drawdown: {drawdown*100:.1f}%"))

    # ── Momentum ───────────────────────────────────────────────────────────────
    mom_sig = row.get("momentum_signal", "neutral") if row else "neutral"
    if mom_sig == "falling":
        drivers.append(("medium", "Falling price momentum — downward pressure building"))

    # ── AI forecast ────────────────────────────────────────────────────────────
    direction = row.get("direction", "stable") if row else "stable"
    p_down    = float(row.get("p_down", 0.33) if row else 0.33)
    if direction == "down" and p_down > 0.65:
        drivers.append(("high",   f"AI forecasts DOWN with {int(p_down*100)}% probability (5D)"))
    elif direction == "down" and p_down > 0.50:
        drivers.append(("medium", f"AI leans bearish — P(down) = {int(p_down*100)}%"))

    # ── Market context ─────────────────────────────────────────────────────────
    excess = r.get("excess_return", 0) or 0
    if excess < -0.03:
        drivers.append(("medium", f"Significantly underperforming market ({excess*100:.1f}%)"))
    elif excess < -0.01:
        drivers.append(("low",    f"Underperforming market by {excess*100:.1f}%"))

    # ── Volume anomaly ─────────────────────────────────────────────────────────
    vol_ma20  = r.get("volume_ma20", 1) or 1
    volume    = r.get("Volume", 0) or 0
    vol_ratio = volume / vol_ma20 if vol_ma20 > 0 else 1
    if vol_ratio > 3.0:
        drivers.append(("high",   f"Extreme volume spike — {vol_ratio:.1f}x average (unusual activity)"))
    elif vol_ratio > 2.0:
        drivers.append(("medium", f"Volume surge — {vol_ratio:.1f}x 20D average"))

    # ── OBV / selling pressure ─────────────────────────────────────────────────
    obv = r.get("obv_signal", 0) or 0
    if obv < -0.5:
        drivers.append(("medium", "Strong selling pressure (OBV negative)"))
    elif obv < 0:
        drivers.append(("low",    "Mild selling pressure (OBV slightly negative)"))

    # ── Market / sector contagion ──────────────────────────────────────────────
    if r.get("is_market_wide"):
        drivers.append(("high",   "Market-wide risk event — broad index affected"))
    elif r.get("is_sector_wide"):
        drivers.append(("medium", "Sector-wide risk event — peers also affected"))

    # ── Consecutive down days ──────────────────────────────────────────────────
    ret = r.get("returns", 0) or 0
    if ret < -0.05:
        drivers.append(("high",   f"Large single-day drop: {ret*100:.1f}%"))
    elif ret < -0.02:
        drivers.append(("medium", f"Notable daily decline: {ret*100:.1f}%"))

    # Sort: high first, then medium, then low
    order = {"high": 0, "medium": 1, "low": 2}
    drivers.sort(key=lambda x: order.get(x[0], 9))
    return drivers


def calculate_risk_score(r) -> float:
    """Returns a 0–1 risk score based on how many detectors fired."""
    score = 0
    if r.get("z_anomaly"):    score += 1
    if r.get("z_anomaly_60"): score += 1
    if r.get("if_anomaly"):   score += 1
    if r.get("ae_anomaly"):   score += 1
    return score / 4


def calculate_severity(score: float) -> str:
    if score > 0.75:  return "CRITICAL"
    if score > 0.5:   return "WARNING"
    if score > 0.25:  return "WATCH"
    return "NORMAL"
