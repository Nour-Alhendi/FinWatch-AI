"""
Stock Deep-Dive — Screen 2
3-panel: price chart + RSI | radar + metric cards + SHAP + narrative.
"""

from __future__ import annotations
import ast
import html
import re
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

from data.loader import (
    COMPANY_NAMES, SEV_COLOR, SEV_SHORT, SIGNAL_DISPLAY,
    load_detection, load_news, load_analyst_ratings,
    load_earnings_calendar, load_correlation_risk,
    fetch_stock_news, rel_time,
)
from ui.glossary import term
from ui.theme import feat_label, anomaly_label, TEXT_MUTED, CHART_AXIS, COLOR_NEUTRAL
from ui.i18n import t


_SEV_CLS = {
    "CRITICAL": "s-critical", "WARNING": "s-warning", "WATCH": "s-watch",
    "NORMAL": "s-normal", "POSITIVE_MOMENTUM": "s-positive", "REVIEW": "s-review",
}
_SIG_COLORS = {
    "EXIT": "#e05252", "HOLD": "#c49a3c", "WATCH": "#5a7ab4",
    "ENTRY": "#4a9e6a", "BUY_SIGNAL": "#4a9e6a", "NEUTRAL": "#4a9e6a", "REDUCE": "#7a6ab4",
}
_CONS_COLORS = {
    "STRONG BUY": "#4a9e6a", "BUY": "#00c4b4", "NEUTRAL": "#8b8b9e",
    "HOLD": "#c49a3c", "SELL": "#e05252",
}
_POSTURE_COLORS = {
    "ELEVATED": "#e05252", "MONITOR": "#c49a3c", "FAVORABLE": "#4a9e6a",
}


def _pd_color(p: float) -> str:
    return "#e05252" if p >= 0.70 else "#c49a3c" if p >= 0.40 else "#4a9e6a"
_PERIOD_OFFSETS = {
    "1M": pd.DateOffset(months=1), "3M": pd.DateOffset(months=3),
    "6M": pd.DateOffset(months=6), "All": None,
}


def _parse_shap(val) -> list[tuple[str, float]]:
    """Parse top3_shap regardless of storage format.
    Handles:
      - Python list/tuple literals: [('feat', 0.12), ...]
      - String format: "feat1(+0.73), feat2(-0.19), feat3(+0.10)"
    """
    if isinstance(val, list):
        return [(str(x[0]), float(x[1])) for x in val if len(x) >= 2]
    s = str(val).strip()
    # Try Python literal first
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, (list, tuple)):
            return [(str(x[0]), float(x[1])) for x in parsed if len(x) >= 2]
    except Exception:
        pass
    # Fallback: "feature_name(+0.123)" format
    result = []
    for m in re.finditer(r"([\w\-]+)\(([+-]?\d+\.?\d*)\)", s):
        result.append((m.group(1), float(m.group(2))))
    return result


def _render_outlook(dec_row: pd.Series, analyst_row: "pd.Series | None") -> None:
    """Side-by-side comparison: what FinWatch AI predicts vs. what analysts say."""
    signal  = str(dec_row.get("trading_signal", "—"))
    p_dd    = float(dec_row.get("p_drawdown", 0))
    conf    = float(dec_row.get("confidence", 0))
    sev     = str(dec_row.get("severity", "NORMAL"))
    sig_col = _SIG_COLORS.get(signal, "#8b8b9e")
    sev_col = SEV_COLOR.get(sev, "#8b8b9e")

    fw_html = f"""
    <div class="outlook-col">
        <div class="otl-source">FinWatch AI</div>
        <div class="otl-verdict" style="color:{sig_col}">{SIGNAL_DISPLAY.get(signal, signal)}</div>
        <div class="otl-horizon">{t("outlook_forecast")}</div>
        <div class="otl-rows">
            <div class="otl-row">
                <span class="otl-key">P(Drawdown)</span>
                <span class="otl-val" style="color:{_pd_color(p_dd)}">{p_dd*100:.0f}%</span>
            </div>
            <div class="otl-row">
                <span class="otl-key">Severity</span>
                <span class="otl-val" style="color:{sev_col}">{sev}</span>
            </div>
            <div class="otl-row">
                <span class="otl-key">{t("outlook_confidence")}</span>
                <span class="otl-val">{conf*100:.0f}%</span>
            </div>
        </div>
    </div>
    """

    if analyst_row is not None:
        consensus   = str(analyst_row.get("consensus", "N/A"))
        bull_pct    = float(analyst_row.get("bullish_pct") or 0)
        total       = int(analyst_row.get("total") or 0)
        strong_buy  = int(analyst_row.get("strong_buy") or 0)
        buy         = int(analyst_row.get("buy") or 0)
        hold        = int(analyst_row.get("hold") or 0)
        sell        = int(analyst_row.get("sell") or 0)
        strong_sell = int(analyst_row.get("strong_sell") or 0)
        period      = str(analyst_row.get("period", ""))[:7]
        cons_col    = _CONS_COLORS.get(consensus, "#8b8b9e")

        def _seg(n: int, color: str, label: str) -> str:
            w = round(n / total * 100) if total > 0 else 0
            return (f'<span class="ab-seg" style="width:{w}%;background:{color}" '
                    f'title="{label}: {n}"></span>')

        bar = (
            _seg(strong_buy, "#4a9e6a", "Strong Buy") +
            _seg(buy, "#00c4b4", "Buy") +
            _seg(hold, "#c49a3c", "Hold") +
            _seg(sell, "#e05252", "Sell") +
            _seg(strong_sell, "#7a2a28", "Strong Sell")
        )

        parts: list[str] = []
        if strong_buy:  parts.append(f'<span style="color:#4a9e6a">{strong_buy} Strong Buy</span>')
        if buy:         parts.append(f'<span style="color:#00c4b4">{buy} Buy</span>')
        if hold:        parts.append(f'<span style="color:#c49a3c">{hold} Hold</span>')
        if sell:        parts.append(f'<span style="color:#e05252">{sell} Sell</span>')
        if strong_sell: parts.append(f'<span style="color:#8c3a36">{strong_sell} Strong Sell</span>')
        breakdown = " &nbsp;·&nbsp; ".join(parts) if parts else "—"

        _analysts_lbl = t("outlook_analysts")
        _analysts_meta = t("outlook_analysts_meta", total=str(total), period=period)
        an_html = f"""
        <div class="outlook-col">
            <div class="otl-source">{_analysts_lbl} &nbsp;<span class="otl-meta">{_analysts_meta}</span></div>
            <div class="otl-verdict" style="color:{cons_col}">{consensus}</div>
            <div class="otl-horizon">{t("outlook_bullish")}: {bull_pct*100:.0f}%</div>
            <div class="otl-bar-wrap">{bar}</div>
            <div class="otl-breakdown">{breakdown}</div>
        </div>
        """
    else:
        an_html = (
            f'<div class="outlook-col">'
            f'<div class="otl-source">{t("outlook_analysts")}</div>'
            f'<div class="otl-no-data">{t("outlook_no_data")}<br>'
            f'<span class="otl-hint">{t("outlook_run_hint")}</span></div>'
            f'</div>'
        )

    st.markdown(
        f'<div class="outlook-section">'
        f'{fw_html}'
        f'<div class="outlook-sep"></div>'
        f'{an_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


def _render_header(ticker: str, det_df: pd.DataFrame | None, dec_row: pd.Series, price_data: dict, earnings_row=None) -> None:
    company  = COMPANY_NAMES.get(ticker, ticker)
    sev      = str(dec_row.get("severity", "NORMAL"))
    signal   = str(dec_row.get("trading_signal", "—"))
    context  = anomaly_label(str(dec_row.get("context", "—")))
    date_str = str(dec_row.get("date", ""))[:10]

    price, pct = price_data.get(ticker, (None, 0.0))
    price_str = f"${price:,.2f}" if price is not None else "—"
    pct_str   = f"{'+' if pct >= 0 else ''}{pct:.1f}%"
    pct_cls   = "up" if pct >= 0 else "dn"

    sev_color = SEV_COLOR.get(sev, COLOR_NEUTRAL)
    sig_color = _SIG_COLORS.get(signal, COLOR_NEUTRAL)

    # Regime strip values from latest detection row
    regime     = "—"
    vol_regime = "—"
    if det_df is not None and not det_df.empty:
        r = det_df.iloc[-1]
        regime     = str(r.get("regime", "—"))
        vol_regime = str(r.get("vol_regime", "—"))

    vol_col = {"low": "#4a9e6a", "moderate": "#c49a3c", "high": "#e05252"}.get(vol_regime.lower(), TEXT_MUTED)

    # Earnings badge
    earn_html = ""
    if earnings_row is not None:
        try:
            import pandas as _pd
            dt       = _pd.to_datetime(earnings_row["earnings_date"])
            days     = int(earnings_row.get("days_until", 0))
            date_lbl = f"{dt.strftime('%B')} {dt.day}"
            _earn_tip = "Elevated volatility near earnings dates may not reflect a structural anomaly."
            earn_html = (
                f'<span class="reg-sep">·</span>'
                f'<span class="reg-earn tip" data-tip="{_earn_tip}">'
                f'⚡ Earnings {date_lbl} (in {days} days)</span>'
            )
        except Exception:
            pass

    st.markdown(
        f"""
        <div class="dd-header">
            <div class="dd-name-block">
                <div class="dd-company-row">
                    <span class="dd-company">{company}</span>
                    <span class="dd-ticker">{ticker}</span>
                    <span class="dd-context">{context}</span>
                </div>
            </div>
            <div class="dd-price-block">
                <div class="dd-price">{price_str}</div>
                <div class="dd-pct {pct_cls}">{pct_str}</div>
            </div>
            <div class="dd-badge-block">
                <div class="badge-labeled">
                    <span class="badge-caption">{t("badge_severity")}</span>
                    <span class="sev-pill {_SEV_CLS.get(sev,'s-normal')}" style="border-color:{sev_color}33">{sev}</span>
                </div>
                <div class="badge-labeled">
                    <span class="badge-caption">{t("badge_signal")}</span>
                    <span class="sig-pill tip" data-tip="FAVORABLE — lower-risk profile · MONITOR — mixed/watch · ELEVATED — elevated risk" style="color:{sig_color};border-color:{sig_color}33">{SIGNAL_DISPLAY.get(signal, signal)}</span>
                </div>
            </div>
        </div>
        <div class="dd-regime-strip">
            <span class="reg-item">
                <span class="reg-dot" style="background:{TEXT_MUTED}"></span>
                <span class="reg-val">{regime.upper()}</span>
            </span>
            <span class="reg-sep">·</span>
            <span class="reg-item">
                <span class="reg-dot" style="background:{vol_col}"></span>
                <span class="reg-val" style="color:{vol_col}">{vol_regime.upper()} VOL</span>
            </span>
            <span class="reg-sep">·</span>
            <span class="reg-date">{date_str}</span>
            {earn_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_price_chart(det_df: pd.DataFrame, ticker: str, period: str) -> None:
    if det_df is None or "Close" not in det_df.columns:
        st.markdown('<div class="no-data">No price data</div>', unsafe_allow_html=True)
        return

    df = det_df.copy()
    x_end  = df["Date"].iloc[-1]
    offset = _PERIOD_OFFSETS.get(period)
    if offset is not None:
        df = df[df["Date"] >= (x_end - offset)].reset_index(drop=True)

    if df.empty:
        st.markdown('<div class="no-data">No data for period</div>', unsafe_allow_html=True)
        return

    close = df["Close"]
    ma50  = df["ma50"]  if "ma50"  in df.columns else close.rolling(50).mean()
    ma200 = df["ma200"] if "ma200" in df.columns else close.rolling(200).mean()

    y_lo = close.min() * 0.97
    y_hi = close.max() * 1.03

    _price_lbl = t("price")
    fig = go.Figure()

    # Price line + fill
    fig.add_trace(go.Scatter(
        x=df["Date"], y=close,
        mode="lines", name=_price_lbl,
        line=dict(color="#00c4b4", width=1.5),
        fill="tozeroy", fillcolor="rgba(0,196,180,0.03)",
        hovertemplate=f"<b>%{{x|%d %b '%y}}</b><br>{_price_lbl} $%{{y:,.2f}}<extra></extra>",
    ))
    # MA50 — subtle
    fig.add_trace(go.Scatter(
        x=df["Date"], y=ma50,
        mode="lines", name="MA50",
        line=dict(color="rgba(255,255,255,0.30)", width=1.0, dash="dash"),
        hovertemplate="MA50 $%{y:,.2f}<extra></extra>",
    ))
    # MA200 — very subtle
    fig.add_trace(go.Scatter(
        x=df["Date"], y=ma200,
        mode="lines", name="MA200",
        line=dict(color="rgba(255,255,255,0.15)", width=1.0, dash="dot"),
        hovertemplate="MA200 $%{y:,.2f}<extra></extra>",
    ))

    # Anomaly markers — early detection only (z_anomaly / ae_anomaly)
    has_anomalies = False
    early_cols = [c for c in ["z_anomaly", "ae_anomaly"] if c in df.columns]
    if early_cols:
        early_hits = df[early_cols].any(axis=1)
        anom_df    = df[early_hits]
        if not anom_df.empty:
            has_anomalies = True
            _anom_confirmed = t("anomaly_confirmed")
            fig.add_trace(go.Scatter(
                x=anom_df["Date"], y=anom_df["Close"],
                mode="markers",
                marker=dict(color="#c49a3c", size=5, symbol="circle"),
                name=t("anomaly"),
                hovertemplate=f"<b>%{{x|%d %b '%y}}</b><br>{_anom_confirmed}<extra></extra>",
            ))

    # Right-side labels
    last_close = float(close.iloc[-1])
    last_ma50  = float(ma50.dropna().iloc[-1]) if len(ma50.dropna()) else None
    last_ma200 = float(ma200.dropna().iloc[-1]) if len(ma200.dropna()) else None
    fig.add_annotation(x=1.01, xref="paper", y=last_close, yref="y",
                       text=f"${last_close:.2f}", xanchor="left", showarrow=False,
                       font=dict(size=10, color="#00c4b4", family="IBM Plex Mono"))
    if last_ma50:
        fig.add_annotation(x=1.01, xref="paper", y=last_ma50, yref="y",
                           text=f"MA50 ${last_ma50:.2f}", xanchor="left", showarrow=False,
                           font=dict(size=10, color="rgba(255,255,255,0.35)", family="IBM Plex Mono"))
    if last_ma200:
        fig.add_annotation(x=1.01, xref="paper", y=last_ma200, yref="y",
                           text=f"MA200 ${last_ma200:.2f}", xanchor="left", showarrow=False,
                           font=dict(size=10, color="rgba(255,255,255,0.2)", family="IBM Plex Mono"))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#111118",
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.09)", color=TEXT_MUTED,
                   tickfont=dict(size=12, family="IBM Plex Mono")),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.09)", color=TEXT_MUTED,
                   tickfont=dict(size=12, family="IBM Plex Mono"),
                   tickprefix="$", side="right", range=[y_lo, y_hi]),
        margin=dict(t=8, b=8, l=4, r=110),
        height=320, showlegend=False,
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#111118", bordercolor="rgba(255,255,255,0.06)",
                        font=dict(size=12, family="IBM Plex Mono", color="#f0f0f5")),
    )

    anom_item = (
        f'&ensp;<span class="leg-item">'
        f'<span class="leg-dot" style="background:#c49a3c"></span>{t("anomaly")}</span>'
    ) if has_anomalies else ""
    st.markdown(
        f'<div class="chart-legend">'
        f'<span class="leg-item"><span class="leg-line" style="background:#00c4b4"></span>{t("price")}</span>'
        f'<span class="leg-item"><span class="leg-dashed" style="border-color:rgba(255,255,255,0.30)"></span>MA50</span>'
        f'<span class="leg-item"><span class="leg-dotted" style="border-color:rgba(255,255,255,0.15)"></span>MA200</span>'
        f'{anom_item}'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _render_rsi(det_df: pd.DataFrame, period: str) -> None:
    if det_df is None or "rsi" not in det_df.columns:
        return

    df = det_df.copy()
    x_end  = df["Date"].iloc[-1]
    offset = _PERIOD_OFFSETS.get(period)
    if offset is not None:
        df = df[df["Date"] >= (x_end - offset)].reset_index(drop=True)

    rsi    = df["rsi"]
    rsi_ma = rsi.rolling(14, min_periods=1).mean()

    fig = go.Figure()
    fig.add_hline(y=70, line=dict(color="#e05252", width=0.6, dash="dot"))
    fig.add_hline(y=30, line=dict(color="#4a9e6a", width=0.6, dash="dot"))
    fig.add_hline(y=50, line=dict(color="rgba(255,255,255,0.05)", width=0.5))
    fig.add_trace(go.Scatter(x=df["Date"], y=rsi_ma.round(2), mode="lines", name="RSI MA14",
                             line=dict(color="rgba(255,255,255,0.20)", width=1.0, dash="dot"),
                             hovertemplate="RSI MA14: %{y:.1f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=df["Date"], y=rsi.round(2), mode="lines", name="RSI",
                             line=dict(color="#c49a3c", width=1.4),
                             hovertemplate="RSI: %{y:.1f}<extra></extra>"))

    rsi_last = float(rsi.iloc[-1])
    fig.add_annotation(x=1.01, xref="paper", y=rsi_last, yref="y",
                       text=f"RSI {rsi_last:.1f}", xanchor="left", showarrow=False,
                       font=dict(size=10, color="#c49a3c", family="IBM Plex Mono"))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#111118",
        xaxis=dict(showgrid=False, color=TEXT_MUTED, tickfont=dict(size=12, family="IBM Plex Mono")),
        yaxis=dict(showgrid=False, color=TEXT_MUTED, tickfont=dict(size=12, family="IBM Plex Mono"), range=[0, 100]),
        margin=dict(t=4, b=4, l=4, r=80), height=110,
        showlegend=False, hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _radar_axis_desc() -> dict[str, tuple[str, str]]:
    return {
        "volatility_anomaly_score": (t("radar_axis_volatility_label"), t("radar_axis_volatility_desc")),
        "volume_anomaly_score":     (t("radar_axis_volume_label"),     t("radar_axis_volume_desc")),
        "price_anomaly_score":      (t("radar_axis_price_label"),      t("radar_axis_price_desc")),
        "context_anomaly_score":    (t("radar_axis_context_label"),    t("radar_axis_context_desc")),
    }


def _radar_axis_phrase() -> dict[str, str]:
    return {
        "volatility_anomaly_score": t("radar_axis_volatility_phrase"),
        "volume_anomaly_score":     t("radar_axis_volume_phrase"),
        "price_anomaly_score":      t("radar_axis_price_phrase"),
        "context_anomaly_score":    t("radar_axis_context_phrase"),
    }


def _radar_interpretation(keys: list[str], vals: list[float]) -> tuple[str, str]:
    """Return (interpretation_text, css_color) based on the raw axis values."""
    max_val = max(vals)
    spread  = max_val - min(vals)
    max_key = keys[vals.index(max_val)]
    phrase  = _radar_axis_phrase()[max_key]

    if max_val < 0.20:
        return t("radar_all_normal"), "#4a9e6a"
    if max_val >= 0.65 and spread >= 0.20:
        return t("radar_main_driver", phrase=phrase), "#e05252"
    if spread < 0.15 and max_val >= 0.25:
        return t("radar_broad"), "#c49a3c"
    if max_val >= 0.40:
        return t("radar_elevated", phrase=phrase), "#c49a3c"
    return t("radar_slight"), "#8b8b9e"


def _render_radar_chart(r: pd.Series) -> tuple[list, list]:
    """Render only the radar chart. Returns (keys, vals) for _render_radar_explanation."""
    keys = ["volume_anomaly_score", "volatility_anomaly_score", "price_anomaly_score", "context_anomaly_score"]
    cats = [feat_label(k) for k in keys]
    vals = [float(r.get(k, 0)) for k in keys]
    max_v = max(max(vals), 1e-6)
    norm  = [v / max_v for v in vals]
    norm_closed = norm + [norm[0]]
    cats_closed = cats + [cats[0]]

    fig = go.Figure(go.Scatterpolar(
        r=norm_closed, theta=cats_closed,
        fill="toself",
        line=dict(color="#00c4b4", width=1.4),
        fillcolor="rgba(0,196,180,0.08)",
        hovertemplate="%{theta}: %{r:.2f}<extra></extra>",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=False, range=[0, 1]),
            angularaxis=dict(
                color=TEXT_MUTED,
                tickfont=dict(size=12, family="IBM Plex Mono", color=CHART_AXIS),
                linecolor="rgba(255,255,255,0.18)",
                gridcolor="rgba(255,255,255,0.12)",
            ),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=36, b=36, l=60, r=60),
        height=230, showlegend=False,
    )
    st.markdown('<div class="section-label">ANOMALY PROFILE</div>', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    return keys, vals


def _render_radar_explanation(keys: list, vals: list) -> None:
    """Render the legend + interpretation line (shown below the top row)."""
    _axis_desc = _radar_axis_desc()
    legend_rows = "".join(
        f'<div class="re-row">'
        f'<span class="re-axis">{short}</span>'
        f'<span class="re-desc">{desc}</span>'
        f'</div>'
        for key in keys
        for short, desc in [_axis_desc[key]]
    )
    interp_text, interp_col = _radar_interpretation(keys, vals)
    st.markdown(
        f'<div class="radar-explain">'
        f'<div class="re-key">{t("radar_key")}</div>'
        f'<div class="re-legend">{legend_rows}</div>'
        f'<div class="re-interp" style="color:{interp_col}">{interp_text}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_metric_cards(r: pd.Series, dec_row: pd.Series) -> None:
    var_95 = float(r.get("var_95", 0))
    es_95  = float(r.get("es_95", 0))
    p_dd   = float(dec_row.get("p_drawdown", 0))
    conf   = float(dec_row.get("confidence", 0))

    st.markdown('<div class="section-label">RISK METRICS</div>', unsafe_allow_html=True)
    st.metric("VaR 95%",      f"{var_95 * 100:.2f}%", help=t("metric_var_help"))
    st.metric("ES 95%",       f"{es_95 * 100:.2f}%",  help=t("metric_es_help"))
    st.metric("P(Drawdown)",  f"{p_dd * 100:.1f}%",   help=t("metric_dd_help"))
    st.metric(t("metric_conf"), f"{conf * 100:.0f}%", help=t("metric_conf_help"))


def _render_shap_bar(top3_shap: list) -> None:
    if not top3_shap:
        return

    features = []
    values   = []
    for item in top3_shap[:3]:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            features.append(feat_label(str(item[0])))
            values.append(float(item[1]))

    if not features:
        return

    colors = ["#00c4b4" if v > 0 else "#e05252" for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=features,
        orientation="h",
        marker_color=colors,
        marker_line_width=0,
        hovertemplate="%{y}: %{x:+.4f}<extra></extra>",
    ))
    fig.add_vline(x=0, line=dict(color="rgba(255,255,255,0.08)", width=1))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=4, b=4, l=0, r=10), height=90,
        showlegend=False,
        xaxis=dict(showgrid=False, color=TEXT_MUTED, zeroline=False,
                   tickfont=dict(size=12, family="IBM Plex Mono")),
        yaxis=dict(color=CHART_AXIS, tickfont=dict(size=12, family="IBM Plex Mono"),
                   autorange="reversed"),
        hoverlabel=dict(bgcolor="#111118", bordercolor="rgba(255,255,255,0.06)",
                        font=dict(size=12, family="IBM Plex Mono", color="#f0f0f5")),
    )
    st.markdown('<div class="section-label">ANOMALY DRIVERS</div>', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _render_correlation_card(corr_row: pd.Series) -> None:
    risk_level = str(corr_row.get("concentration_risk", "—")).upper()
    avg_corr   = float(corr_row.get("avg_correlation", 0))
    count      = int(corr_row.get("high_corr_count", 0))
    peers_raw  = corr_row.get("high_corr_peers", "")
    peers_str  = ", ".join(peers_raw) if isinstance(peers_raw, list) else str(peers_raw).strip()

    risk_color   = {"HIGH": "#e05252", "MEDIUM": "#c49a3c", "LOW": "#4a9e6a"}.get(risk_level, TEXT_MUTED)
    corr_sub_key = {"HIGH": "corr_sub_high", "MEDIUM": "corr_sub_medium"}.get(risk_level, "corr_sub_low")
    _tip = (
        "Stocks with high mutual correlation amplify portfolio risk if they move together. "
        "MEDIUM = 2–4 highly correlated peers detected."
    )

    peers_html = (
        f'<div class="corr-row">High-corr peers: <span class="corr-peers">{peers_str}</span></div>'
        if peers_str else ""
    )

    st.markdown(
        f'<div class="corr-card">'
        f'<div class="corr-hdr">'
        f'<span class="corr-lbl tip" data-tip="{_tip}">Correlation Risk</span>'
        f'<span class="corr-lvl" style="color:{risk_color}">{risk_level}</span>'
        f'</div>'
        f'<div class="corr-sub">{t(corr_sub_key)}</div>'
        f'{peers_html}'
        f'<div class="corr-row">Avg correlation: <span class="corr-peers">{avg_corr:+.2f}</span></div>'
        f'</div>',
        unsafe_allow_html=True,
    )


_DD_CSS = """
<style>
/* ── Deep-Dive header ── */
.dd-header{
    display:flex;align-items:center;gap:16px;
    padding:12px 0 10px;
    border-bottom:1px solid rgba(255,255,255,0.14);
    margin-bottom:8px;
}
.dd-name-block{flex:1}
.dd-company-row{display:flex;align-items:baseline;gap:10px;flex-wrap:wrap}
.dd-company{font-size:20px;font-weight:500;color:#f0f0f5;letter-spacing:-0.2px;line-height:1.2}
.dd-ticker{font-size:13px;color:#4a4a5a;font-family:'IBM Plex Mono',monospace;letter-spacing:0.05em}
.dd-context{font-size:12px;color:#4a4a5a;font-family:'IBM Plex Mono',monospace}
.dd-price-block{text-align:right}
.dd-price{font-size:16px;font-weight:500;color:#f0f0f5;font-family:'IBM Plex Mono',monospace;line-height:1}
.dd-pct{font-size:13px;font-family:'IBM Plex Mono',monospace}
.dd-pct.up{color:#00c4b4}
.dd-pct.dn{color:#e05252}
.dd-badge-block{display:flex;flex-direction:column;gap:6px;align-items:flex-end;min-width:100px}
.badge-labeled{display:flex;flex-direction:column;align-items:flex-end;gap:2px}
.badge-caption{
    font-size:9px;letter-spacing:0.14em;text-transform:uppercase;
    color:#4a4a5a;font-family:'IBM Plex Mono',monospace;
}
.sev-pill{
    font-size:11px;font-weight:500;letter-spacing:0.04em;
    font-family:'IBM Plex Mono',monospace;
    padding:3px 10px;border-radius:12px;border:1px solid;
}
.sig-pill{
    font-size:11px;letter-spacing:0.03em;
    font-family:'IBM Plex Mono',monospace;
    padding:3px 10px;border-radius:10px;border:1px solid;
}

/* ── Regime strip ── */
.dd-regime-strip{
    display:flex;align-items:center;gap:10px;
    padding:5px 0 10px;
    font-family:'IBM Plex Mono',monospace;
    border-bottom:1px solid rgba(255,255,255,0.14);
    margin-bottom:12px;
}
.reg-item{display:flex;align-items:center;gap:5px}
.reg-dot{width:7px;height:7px;border-radius:50%;display:inline-block}
.reg-val{font-size:12px;color:#8b8b9e;font-weight:400}
.reg-sep{color:#4a4a5a}
.reg-date{font-size:12px;color:#4a4a5a;letter-spacing:0.04em}

/* ── Section labels ── */
.section-label{
    font-size:11px;letter-spacing:0.12em;text-transform:uppercase;
    color:#6b6b7e;font-family:'IBM Plex Mono',monospace;font-weight:500;
    padding-bottom:5px;margin-bottom:8px;
    border-bottom:1px solid rgba(255,255,255,0.15);
}

/* ── Metric cards ── */
.metric-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:12px}
.mc{
    background:#111118;
    border:1px solid rgba(255,255,255,0.14);
    border-radius:8px;padding:10px 12px;
}
.mc-lbl{
    font-size:11px;letter-spacing:0.12em;text-transform:uppercase;
    color:#4a4a5a;font-family:'IBM Plex Mono',monospace;margin-bottom:5px;font-weight:500;
}
.mc-val{
    font-size:20px;font-weight:500;color:#f0f0f5;
    font-family:'IBM Plex Mono',monospace;line-height:1;
}

/* ── Radar explanation ── */
.radar-explain{
    background:#16161f;
    border:1px solid rgba(255,255,255,0.16);
    border-radius:8px;
    padding:10px 12px;
    margin-top:8px;
    margin-bottom:10px;
}
.re-key{
    font-size:12px;color:#8b8b9e;
    font-family:'Inter',sans-serif;
    line-height:1.5;
    border-bottom:1px solid rgba(255,255,255,0.15);
    padding-bottom:6px;margin-bottom:7px;
}
.re-legend{display:flex;flex-direction:column;gap:3px;margin-bottom:7px}
.re-row{display:flex;align-items:baseline;gap:6px;line-height:1.4}
.re-axis{
    font-size:12px;font-family:'IBM Plex Mono',monospace;
    color:#a0a0b4;font-weight:500;
    white-space:nowrap;min-width:76px;flex-shrink:0;
}
.re-desc{
    font-size:12px;color:#8b8b9e;
    font-family:'Inter',sans-serif;
}
.re-interp{
    font-size:14px;font-weight:600;
    font-family:'Inter',sans-serif;
    border-top:1px solid rgba(255,255,255,0.15);
    padding:10px 0 2px;line-height:1.5;
    letter-spacing:-0.1px;
}

/* ── Risk Summary ── */
.narrative-box{
    background:#16161f;
    border:1px solid rgba(255,255,255,0.16);
    border-left:2px solid rgba(0,196,180,0.4);
    border-radius:0 8px 8px 0;
    padding:10px 14px;
    font-size:13px;
    font-family:'Inter',sans-serif;
    line-height:1.8;
    margin-bottom:10px;
    word-break:break-word;
}

/* ── No data ── */
.no-data{
    height:160px;display:flex;align-items:center;justify-content:center;
    color:#4a4a5a;font-size:13px;font-family:'Inter',sans-serif;
}

/* ── Period selector ── */
[data-testid="stSegmentedControl"]{margin-bottom:4px!important}

/* ── Chart legend strip ── */
.chart-legend{
    display:flex;align-items:center;gap:18px;flex-wrap:wrap;
    padding:4px 0 6px;
    font-family:'IBM Plex Mono',monospace;
    font-size:12px;color:#8b8b9e;
}
.leg-item{display:flex;align-items:center;gap:6px}
.leg-line{display:inline-block;width:22px;height:2px;border-radius:1px;vertical-align:middle}
.leg-dashed{display:inline-block;width:22px;height:0;border-top:2px dashed;vertical-align:middle}
.leg-dotted{display:inline-block;width:22px;height:0;border-top:2px dotted;vertical-align:middle}
.leg-dot{width:8px;height:8px;border-radius:50%;display:inline-block;vertical-align:middle;flex-shrink:0}

/* ── Outlook section ── */
.outlook-section{
    display:flex;align-items:stretch;
    background:#16161f;
    border:1px solid rgba(255,255,255,0.18);
    border-radius:8px;
    margin-bottom:16px;
    overflow:hidden;
}
.outlook-col{flex:1;padding:16px 20px}
.outlook-sep{width:1px;background:rgba(255,255,255,0.08);margin:14px 0;flex-shrink:0}
.otl-source{
    font-size:11px;letter-spacing:0.12em;text-transform:uppercase;
    color:#8b8b9e;font-family:'IBM Plex Mono',monospace;font-weight:500;
    margin-bottom:6px;
}
.otl-meta{font-size:11px;color:#6b6b7e;text-transform:none;letter-spacing:0}
.otl-verdict{
    font-size:28px;font-weight:500;
    font-family:'IBM Plex Mono',monospace;
    letter-spacing:-0.3px;line-height:1.1;
    margin-bottom:3px;
}
.otl-horizon{
    font-size:12px;color:#6b6b7e;
    font-family:'Inter',sans-serif;
    margin-bottom:12px;
}
.otl-rows{display:flex;flex-direction:column;gap:5px}
.otl-row{
    display:flex;align-items:center;justify-content:space-between;
    font-family:'IBM Plex Mono',monospace;
}
.otl-key{font-size:12px;color:#8b8b9e}
.otl-val{font-size:14px;font-weight:500;color:#f0f0f5}
.otl-bar-wrap{
    display:flex;height:6px;border-radius:4px;overflow:hidden;
    gap:1px;margin-bottom:8px;
}
.ab-seg{height:100%;min-width:1px;cursor:default}
.otl-breakdown{
    font-size:12px;color:#8b8b9e;
    font-family:'Inter',sans-serif;line-height:1.8;
}
.otl-no-data{
    font-size:13px;color:#6b6b7e;
    font-family:'Inter',sans-serif;margin-top:14px;line-height:1.7;
}
.otl-hint{font-size:12px;color:#6b6b7e}

/* ── Earnings badge (regime strip) ── */
.reg-earn{
    font-size:12px;color:#c49a3c;font-weight:500;
    font-family:'IBM Plex Mono',monospace;
    letter-spacing:0.04em;cursor:help;
}

/* ── Correlation card ── */
.corr-card{
    background:#16161f;
    border:1px solid rgba(255,255,255,0.16);
    border-radius:8px;
    padding:10px 12px;
    margin-bottom:12px;
    font-family:'IBM Plex Mono',monospace;
}
.corr-hdr{
    display:flex;align-items:center;justify-content:space-between;
    margin-bottom:6px;
}
.corr-lbl{
    font-size:11px;letter-spacing:0.12em;text-transform:uppercase;
    color:#8b8b9e;cursor:help;font-weight:500;
}
.corr-lvl{font-size:14px;font-weight:500}
.corr-row{font-size:12px;color:#8b8b9e;line-height:1.7}
.corr-peers{color:#c0c0d0}

/* ── st.metric overrides ── */
[data-testid="stMetric"]{
    background:#16161f!important;
    border:1px solid rgba(255,255,255,0.16)!important;
    border-radius:8px!important;
    padding:10px 12px!important;
}
[data-testid="stMetricLabel"] p,
[data-testid="stMetricLabel"] div{
    font-size:11px!important;
    letter-spacing:0.12em!important;
    text-transform:uppercase!important;
    color:#8b8b9e!important;
    font-family:'IBM Plex Mono',monospace!important;
    font-weight:500!important;
}
[data-testid="stMetricValue"] div{
    font-size:20px!important;
    font-weight:500!important;
    color:#f0f0f5!important;
    font-family:'IBM Plex Mono',monospace!important;
    line-height:1.2!important;
}
[data-testid="stMetricLabel"] button svg{width:14px!important;height:14px!important}
[data-testid="stMetricLabel"] button{color:#4a4a5a!important;padding:0 2px!important}
.corr-sub{
    font-size:11px;color:#8b8b9e;
    font-family:'Inter',sans-serif;
    margin-bottom:6px;line-height:1.4;
}

/* ── Stock news flash (below header) ── */
.stock-news-wrap{
    border-top:1px solid rgba(255,255,255,0.08);
    padding:6px 0 8px;margin-bottom:4px;
}
.stock-news-item{
    display:block;text-decoration:none;
}
.stock-news-hl{
    display:block;font-size:12px;color:#f0f0f5;
    line-height:1.4;font-family:'Inter',sans-serif;
    margin-bottom:2px;
    white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
}
.stock-news-item:hover .stock-news-hl{color:#00c4b4;transition:color 0.15s}
.stock-news-meta{
    font-size:11px;color:#7a9ab0;
    font-family:'IBM Plex Mono',monospace;
}
.stock-news-none{
    font-size:12px;color:#8080a4;
    font-family:'Inter',sans-serif;
    border-top:1px solid rgba(255,255,255,0.08);
    padding:6px 0 8px;
}
</style>
"""
_DD_CSS = (
    _DD_CSS
    .replace("#4a4a5a", TEXT_MUTED)
    .replace("#6b6b7e", TEXT_MUTED)
    .replace("#8b8b9e", CHART_AXIS)
)


def _render_risk_summary(
    dec_row: pd.Series,
    top3_shap: list[tuple[str, float]],
    decision_context: str,
    analyst_row: "pd.Series | None",
) -> None:
    """Deterministic 1–2 line risk summary composed from real per-stock data."""
    signal  = str(dec_row.get("trading_signal", "HOLD"))
    posture = SIGNAL_DISPLAY.get(signal, "MONITOR")
    color   = _POSTURE_COLORS.get(posture, "#8b8b9e")

    sep = f' <span style="color:{TEXT_MUTED};font-weight:300"> · </span> '
    sec = f"color:{CHART_AXIS}"
    mono = "font-family:'IBM Plex Mono',monospace;font-size:12px;color:#c8c8d8"

    parts: list[str] = []

    # 1. Posture label
    parts.append(
        f'<span style="color:{color};font-weight:600;letter-spacing:0.06em">{posture}</span>'
    )

    # 2. decision_context
    ctx = str(decision_context or "").strip()
    if ctx and ctx not in ("nan", "None", ""):
        parts.append(f'<span style="{sec}">{html.escape(ctx)}</span>')

    # 3. Top SHAP driver
    if top3_shap:
        feat, val = top3_shap[0]
        label = feat_label(feat)
        sign  = "+" if val >= 0 else ""
        parts.append(
            f'<span style="{sec}">Top driver: '
            f'<span style="{mono}">{html.escape(label)} ({sign}{val:.3f})</span></span>'
        )

    # 4. Analyst consensus
    divergence = False
    if analyst_row is not None:
        consensus = str(analyst_row.get("consensus", "")).strip()
        bp        = analyst_row.get("bullish_pct")
        if consensus and consensus not in ("nan", "None", "N/A", ""):
            bp_str = f"{float(bp) * 100:.0f}%" if bp and str(bp) not in ("nan", "None") else ""
            a_text = html.escape(consensus)
            if bp_str:
                a_text += f' · {bp_str}'
            parts.append(
                f'<span style="{sec}">Analysts: <span style="{mono}">{a_text}</span></span>'
            )
            if posture == "ELEVATED" and consensus.upper() in ("BUY", "STRONG BUY"):
                divergence = True

    summary = sep.join(parts)
    if divergence:
        summary += (
            f'{sep}<span style="color:#00c4b4">↕ short-term risk vs. long-term thesis</span>'
        )

    st.markdown('<div class="section-label">RISK SUMMARY</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="narrative-box">{summary}</div>', unsafe_allow_html=True)


def _render_stock_news(articles: list) -> None:
    """Compact latest-news line shown just below the stock header."""
    if not articles:
        st.markdown(
            f'<div class="stock-news-none">'
            f'<span class="section-label" style="border:none;padding:0;margin:0">'
            f'{t("stock_news_title")}</span> '
            f'<span style="font-size:12px;color:#8080a4">{t("news_no_recent")}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        return

    a   = articles[0]
    hl  = html.escape(str(a.get("headline", "")))
    src = html.escape(str(a.get("source", "")))
    url = str(a.get("url", ""))
    rt  = rel_time(a.get("datetime", 0))
    meta = " · ".join(x for x in [src, rt] if x)
    href = f'href="{url}" target="_blank" rel="noopener"' if url else ""

    st.markdown(
        f'<div class="stock-news-wrap">'
        f'<div class="section-label" style="border:none;padding:0;margin-bottom:5px">'
        f'{t("stock_news_title")}</div>'
        f'<a class="stock-news-item" {href}>'
        f'<span class="stock-news-hl">{hl}</span>'
        f'<span class="stock-news-meta">{meta}</span>'
        f'</a>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_deep_dive(ticker: str | None, decisions: pd.DataFrame, price_data: dict, _news_df=None) -> None:
    st.markdown(_DD_CSS, unsafe_allow_html=True)

    if ticker is None:
        st.markdown('<div class="no-data" style="height:40vh">Select a stock from the Command Center or the sidebar.</div>', unsafe_allow_html=True)
        return

    # ── Load data ───────────────────────────────────────────────────────────────
    det_df = load_detection(ticker)

    dec_df  = decisions[decisions["ticker"] == ticker]
    dec_row = dec_df.iloc[-1] if not dec_df.empty else pd.Series(dtype=object)

    # Analyst ratings
    analyst_row = None
    ar_df = load_analyst_ratings()
    if ar_df is not None and not ar_df.empty:
        ar_filt = ar_df[ar_df["ticker"] == ticker]
        if not ar_filt.empty:
            analyst_row = ar_filt.iloc[-1]

    # Earnings calendar
    earnings_row = None
    earn_df = load_earnings_calendar()
    if earn_df is not None and not earn_df.empty:
        earn_filt = earn_df[(earn_df["ticker"] == ticker) & (earn_df["earnings_soon"] == True)]
        if not earn_filt.empty:
            earnings_row = earn_filt.iloc[-1]

    # Correlation risk
    corr_row = None
    corr_df = load_correlation_risk()
    if corr_df is not None and not corr_df.empty:
        corr_filt = corr_df[corr_df["ticker"] == ticker]
        if not corr_filt.empty:
            corr_row = corr_filt.iloc[-1]

    # Explanations
    top3_shap        = []
    decision_context = ""
    exp_path         = Path("data/explanations/explanations.parquet")
    if exp_path.exists():
        try:
            exp_df  = pd.read_parquet(exp_path)
            exp_row = exp_df[exp_df["ticker"] == ticker]
            if not exp_row.empty:
                last_exp         = exp_row.iloc[-1]
                top3_shap        = _parse_shap(last_exp.get("top3_shap", []))
                decision_context = str(last_exp.get("decision_context", ""))
        except Exception:
            pass

    # ── Header ──────────────────────────────────────────────────────────────────
    _render_header(ticker, det_df, dec_row, price_data, earnings_row)

    # ── Latest news (Finnhub, cached 30 min) ────────────────────────────────────
    _render_stock_news(fetch_stock_news(ticker, 1))

    # ── Outlook: FinWatch AI vs. Analysts ────────────────────────────────────────
    _render_outlook(dec_row, analyst_row)

    # ── Period selector ─────────────────────────────────────────────────────────
    period = st.segmented_control("Period", ["1M", "3M", "6M", "All"], default="1M", key="dd_period")
    if period is None:
        period = "1M"

    # ── 2-column layout ─────────────────────────────────────────────────────────
    left, right = st.columns([5, 4], gap="medium")

    # ─── Left: price chart + RSI + Anomaly Drivers ─────────────────────────────
    with left:
        st.markdown('<div class="section-label">PRICE · MA50 · MA200</div>', unsafe_allow_html=True)
        _render_price_chart(det_df, ticker, period)

        st.markdown('<div style="height:4px"></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="section-label">{term("rsi")}</div>', unsafe_allow_html=True)
        _render_rsi(det_df, period)

        st.markdown('<div style="height:4px"></div>', unsafe_allow_html=True)
        _render_shap_bar(top3_shap)

    # ─── Right: risk panel ──────────────────────────────────────────────────────
    with right:
        if det_df is not None and not det_df.empty:
            r = det_df.iloc[-1]
            keys, vals = _render_radar_chart(r)
            _render_radar_explanation(keys, vals)
            if corr_row is not None:
                _render_correlation_card(corr_row)
        else:
            st.markdown('<div class="no-data">No detection data</div>', unsafe_allow_html=True)

        _render_risk_summary(dec_row, top3_shap, decision_context, analyst_row)

        company = COMPANY_NAMES.get(ticker, ticker)
        if st.button("Full AI risk breakdown →", key="ask_ai_btn", use_container_width=True, type="primary"):
            prefill = f"Give me a full risk breakdown for {ticker} — {company}."
            st.session_state.panel_chat_context  = ticker
            st.session_state.panel_chat_history  = [{"role": "user", "content": prefill}]
            st.rerun()
