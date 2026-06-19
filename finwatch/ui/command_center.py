"""
Command Center — Screen 1
Landing page: regime banner, KPI strip, alert list, sector roll-up bar.
"""

import ast
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

from data.loader import (
    COMPANY_NAMES, SECTORS, SEV_COLOR, SEV_SHORT,
    SIGNAL_DISPLAY, load_detection,
    load_earnings_calendar, load_macro_context,
)
from ui.glossary import term
from ui.i18n import t
from ui.theme import TEXT_MUTED, CHART_AXIS


_SEV_ORDER = {"CRITICAL": 0, "WARNING": 1, "WATCH": 2, "REVIEW": 3, "POSITIVE_MOMENTUM": 4, "NORMAL": 5}
_SEV_CLS   = {
    "CRITICAL": "s-critical", "WARNING": "s-warning", "WATCH": "s-watch",
    "NORMAL": "s-normal", "POSITIVE_MOMENTUM": "s-positive", "REVIEW": "s-review",
}
_SEV_DOT = {
    "CRITICAL":          "#e05252",
    "WARNING":           "#c49a3c",
    "WATCH":             "#5a7ab4",
    "NORMAL":            "#4a9e6a",
    "POSITIVE_MOMENTUM": "#4a9e6a",
    "REVIEW":            "#7a6ab4",
}
_SIG_COLORS = {
    "EXIT":         "#e05252",
    "HOLD":         "#c49a3c",
    "WATCH":        "#5a7ab4",
    "ENTRY":        "#4a9e6a",
    "BUY_SIGNAL":   "#4a9e6a",
    "NEUTRAL":      "#4a9e6a",
    "REDUCE":       "#7a6ab4",
}
# Background fill (15% opacity) for signal pills
_SIG_BG = {
    "EXIT":         "rgba(224,82,82,0.15)",
    "HOLD":         "rgba(196,154,60,0.15)",
    "WATCH":        "rgba(90,122,180,0.15)",
    "ENTRY":        "rgba(74,158,106,0.15)",
    "BUY_SIGNAL":   "rgba(74,158,106,0.15)",
    "NEUTRAL":      "rgba(139,139,158,0.15)",
    "REDUCE":       "rgba(122,106,180,0.15)",
}


def _latest(decisions: pd.DataFrame) -> pd.DataFrame:
    """Return the most recent row per ticker, severity-sorted."""
    if decisions.empty:
        return decisions
    df = decisions.copy()
    df["_date_sort"] = pd.to_datetime(df["date"], errors="coerce")
    latest = df.sort_values("_date_sort").groupby("ticker").last().reset_index()
    latest = latest[~latest["ticker"].str.startswith("^")]
    latest["_sev_ord"] = latest["severity"].map(_SEV_ORDER).fillna(99)
    return latest.sort_values(["_sev_ord", "ticker"]).drop(columns=["_date_sort", "_sev_ord"])


_VOL_LABELS = {
    "low":      "Low Volatility",
    "moderate": "Moderate Volatility",
    "high":     "High Volatility",
}
_REGIME_TIP = "Market regime derived from MA200 and VIX levels. Updated with each pipeline run."


def _regime_banner(decisions: pd.DataFrame, macro_df=None) -> None:
    """Single-line regime + macro context strip."""
    spx_df = load_detection("^SPX")
    regime     = "—"
    vol_regime = "—"

    if spx_df is not None and not spx_df.empty:
        r = spx_df.iloc[-1]
        vol_regime = str(r.get("vol_regime", "—"))
        regime     = str(r.get("regime", "—"))

    _regime_labels = {
        "bull": "BULL", "bear": "BEAR",
        "neutral": "NEUTRAL", "recovery": "RECOVERY",
        "sideways": t("regime_sideways").upper(),
    }
    regime_label = _regime_labels.get(regime.lower(), regime.upper())
    vol_label    = {"low": "Low Vol", "moderate": "Moderate Vol", "high": "High Vol"}.get(
        vol_regime.lower(), f"{vol_regime.title()} Vol"
    )
    regime_color = {"BULL": "#4a9e6a", "BEAR": "#e05252", "RECOVERY": "#4a9e6a"}.get(regime_label, "#8b8b9e")
    vol_color    = {"low": "#4a9e6a", "moderate": "#c49a3c", "high": "#e05252"}.get(vol_regime.lower(), "#8b8b9e")

    # Macro values
    vix_str = "—"; t10y_str = "—"; dxy_str = "—"
    if macro_df is not None and not macro_df.empty:
        mr      = macro_df.iloc[-1]
        vix_val = float(mr.get("vix", 0))
        t10y    = float(mr.get("treasury_10y", 0))
        dxy     = float(mr.get("dollar_index", 0))
        vix_col = "#e05252" if vix_val > 25 else "#00c4b4" if vix_val < 20 else "#8b8b9e"
        vix_str = f'<span style="color:{vix_col}">{vix_val:.2f}</span>'
        t10y_str = f"{t10y:.2f}%"
        dxy_str  = f"{dxy:.1f}"

    st.markdown(
        f'<div class="regime-banner">'
        f'<span class="rb-label">S&amp;P 500</span>'
        f'<span class="rb-sep">·</span>'
        f'<span class="rb-regime" style="color:{regime_color}">{regime_label}</span>'
        f'<span class="rb-sep">·</span>'
        f'<span class="rb-vol" style="color:{vol_color}">{vol_label}</span>'
        f'<span class="rb-sep">·</span>'
        f'<span class="rb-macro">VIX {vix_str}</span>'
        f'<span class="rb-sep">·</span>'
        f'<span class="rb-macro">10Y {t10y_str}</span>'
        f'<span class="rb-sep">·</span>'
        f'<span class="rb-macro">DXY {dxy_str}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _kpi_strip(latest: pd.DataFrame) -> None:
    n_critical = int((latest["severity"] == "CRITICAL").sum())
    n_warning  = int((latest["severity"] == "WARNING").sum())
    n_watch    = int((latest["severity"] == "WATCH").sum())
    avg_pd     = latest["p_drawdown"].mean() if "p_drawdown" in latest.columns else 0.0
    pct        = avg_pd * 100
    pd_col     = "#e05252" if pct > 60 else "#c49a3c" if pct > 35 else "#4a9e6a"
    pd_lbl     = term("p_drawdown", "AVG DRAWDOWN PROB")

    st.markdown(
        f"""
        <div class="kpi-strip">
            <div class="kpi-item">
                <div class="kpi-val" style="color:#e05252">{n_critical}</div>
                <div class="kpi-lbl">CRITICAL</div>
            </div>
            <div class="kpi-sep"></div>
            <div class="kpi-item">
                <div class="kpi-val" style="color:#c49a3c">{n_warning}</div>
                <div class="kpi-lbl">WARNING</div>
            </div>
            <div class="kpi-sep"></div>
            <div class="kpi-item">
                <div class="kpi-val" style="color:#5a7ab4">{n_watch}</div>
                <div class="kpi-lbl">WATCH</div>
            </div>
            <div class="kpi-sep"></div>
            <div class="kpi-item">
                <div class="kpi-val" style="color:{pd_col}">{pct:.1f}%</div>
                <div class="kpi-lbl">{pd_lbl}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _sector_bar(latest: pd.DataFrame) -> None:
    """Plotly stacked horizontal bar — severity counts per sector."""
    sector_rows = []
    for sector, tickers in SECTORS.items():
        grp = latest[latest["ticker"].isin(tickers)]
        if grp.empty:
            continue
        sector_rows.append({
            "sector":   sector,
            "CRITICAL": int((grp["severity"] == "CRITICAL").sum()),
            "WARNING":  int((grp["severity"] == "WARNING").sum()),
            "WATCH":    int((grp["severity"] == "WATCH").sum()),
            "NORMAL":   int(grp["severity"].isin(["NORMAL", "POSITIVE_MOMENTUM"]).sum()),
        })

    if not sector_rows:
        return

    df_sec = pd.DataFrame(sector_rows)
    fig = go.Figure()
    for sev, col in [
        ("CRITICAL", "#e05252"),
        ("WARNING",  "#c49a3c"),
        ("WATCH",    "#5a7ab4"),
        ("NORMAL",   "#4a9e6a"),
    ]:
        fig.add_trace(go.Bar(
            y=df_sec["sector"],
            x=df_sec[sev],
            name=sev,
            orientation="h",
            marker_color=col,
            hovertemplate=f"<b>%{{y}}</b><br>{sev}: %{{x}}<extra></extra>",
        ))

    fig.update_layout(
        barmode="stack",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=4, b=10, l=0, r=10),
        height=300,
        showlegend=False,
        xaxis=dict(
            showgrid=False, color=TEXT_MUTED,
            tickfont=dict(size=12, family="IBM Plex Mono"),
        ),
        yaxis=dict(
            color=CHART_AXIS, tickfont=dict(size=12, family="IBM Plex Mono"),
            gridcolor="rgba(255,255,255,0.09)",
        ),
        hoverlabel=dict(
            bgcolor="#111118", bordercolor="rgba(255,255,255,0.06)",
            font=dict(size=12, family="IBM Plex Mono", color="#f0f0f5"),
        ),
    )
    st.markdown('<div class="section-label">SECTOR SEVERITY</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sev-legend">'
        '<span class="sev-leg-item">'
        '<span class="sev-leg-sw" style="background:#e05252"></span>'
        '<span class="sev-leg-tx" style="color:#e05252">CRITICAL</span></span>'
        '<span class="sev-leg-item">'
        '<span class="sev-leg-sw" style="background:#c49a3c"></span>'
        '<span class="sev-leg-tx" style="color:#c49a3c">WARNING</span></span>'
        '<span class="sev-leg-item">'
        '<span class="sev-leg-sw" style="background:#5a7ab4"></span>'
        '<span class="sev-leg-tx" style="color:#5a7ab4">WATCH</span></span>'
        '<span class="sev-leg-item">'
        '<span class="sev-leg-sw" style="background:#4a9e6a"></span>'
        '<span class="sev-leg-tx" style="color:#4a9e6a">NORMAL</span></span>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown(
        '<div class="sector-caption">Number of stocks by severity level, per sector</div>',
        unsafe_allow_html=True,
    )


def _alert_list(latest: pd.DataFrame, price_data: dict, earnings_df=None) -> None:
    """Render severity-sorted alert rows. Each ticker row navigates to Deep-Dive."""
    st.markdown('<div class="section-label">ACTIVE ALERTS</div>', unsafe_allow_html=True)

    # Earnings lookup: ticker → formatted short date string "Jun 24"
    _earn_soon: set = set()
    _earn_label: dict = {}
    if earnings_df is not None and not earnings_df.empty:
        soon = earnings_df[earnings_df["earnings_soon"] == True]
        for _, er in soon.iterrows():
            tk = str(er["ticker"])
            _earn_soon.add(tk)
            try:
                dt = pd.to_datetime(er["earnings_date"])
                _earn_label[tk] = f"{dt.strftime('%b')} {dt.day}"
            except Exception:
                _earn_label[tk] = "soon"

    # Header row
    _sig_legend = "FAVORABLE — lower-risk profile · MONITOR — mixed/watch · ELEVATED — elevated risk"
    _h0, _h1, _h2, _h3, _h4 = st.columns([0.4, 4.4, 1.6, 1.6, 1.0])
    with _h1:
        st.markdown('<div class="ar-col-hdr">COMPANY</div>', unsafe_allow_html=True)
    with _h2:
        st.markdown('<div class="ar-col-hdr">PRICE</div>', unsafe_allow_html=True)
    with _h3:
        st.markdown(
            f'<div class="ar-col-hdr tip" data-tip="{_sig_legend}">SIGNAL</div>',
            unsafe_allow_html=True,
        )
    with _h4:
        st.markdown(
            f'<div class="ar-col-hdr tip" data-tip="{t("dd_tip")}">RISK</div>',
            unsafe_allow_html=True,
        )
    st.markdown(
        '<div style="height:1px;background:rgba(255,255,255,0.14);margin:0 0 4px 0"></div>',
        unsafe_allow_html=True,
    )

    shown = 0
    for _, row in latest.iterrows():
        ticker  = str(row["ticker"])
        sev     = str(row.get("severity", "NORMAL"))
        signal  = str(row.get("trading_signal", "—"))
        p_dd    = float(row.get("p_drawdown", 0))
        company = COMPANY_NAMES.get(ticker, ticker)

        price, pct = price_data.get(ticker, (None, 0.0))
        price_str = f"${price:,.2f}" if price is not None else "—"
        pct_str   = f"{'+' if pct >= 0 else ''}{pct:.1f}%"
        pct_cls   = "up" if pct >= 0 else "dn"

        dot_col   = _SEV_DOT.get(sev, TEXT_MUTED)
        sig_col   = _SIG_COLORS.get(signal, "#8b8b9e")
        sig_bg    = _SIG_BG.get(signal, "rgba(139,139,158,0.15)")
        sig_label = SIGNAL_DISPLAY.get(signal, signal)
        pd_col    = "#e05252" if p_dd >= 0.70 else "#c49a3c" if p_dd >= 0.40 else "#4a9e6a"

        earn_html = (
            f'&nbsp;<span class="earn-badge">⚡ {_earn_label[ticker]}</span>'
            if ticker in _earn_soon else ""
        )

        st.markdown('<span class="al-row-mark"></span>', unsafe_allow_html=True)
        col_dot, col_main, col_price, col_sig, col_pd = st.columns([0.4, 4.4, 1.6, 1.6, 1.0])
        with col_dot:
            st.markdown(
                f'<div class="ar-dot-wrap"><span class="ar-dot" style="background:{dot_col}"></span></div>',
                unsafe_allow_html=True,
            )
        with col_main:
            if st.button(
                f"{company}",
                key=f"go_{ticker}",
                use_container_width=True,
                help=ticker,
            ):
                st.session_state.selected = ticker
                st.session_state.page = "deep_dive"
                st.rerun()
            st.markdown(
                f'<div class="ar-ticker">{ticker}{earn_html}</div>',
                unsafe_allow_html=True,
            )
        with col_price:
            st.markdown(
                f'<div class="ar-price">{price_str}'
                f'<span class="ar-pct {pct_cls}"> {pct_str}</span></div>',
                unsafe_allow_html=True,
            )
        with col_sig:
            st.markdown(
                f'<span class="ar-sig-pill" style="color:{sig_col};background:{sig_bg}">{sig_label}</span>',
                unsafe_allow_html=True,
            )
        with col_pd:
            st.markdown(
                f'<span class="ar-pd" style="color:{pd_col}">{p_dd*100:.0f}%</span>',
                unsafe_allow_html=True,
            )
        shown += 1

    if shown == 0:
        st.markdown(
            '<div class="ar-empty">No alert data — run the pipeline first.</div>',
            unsafe_allow_html=True,
        )


_CC_CSS = """
<style>
/* ── Regime banner — single line ── */
.regime-banner{
    display:flex;align-items:center;gap:10px;flex-wrap:wrap;
    padding:0 0 14px;
    font-family:'IBM Plex Mono',monospace;
    font-size:13px;
}
.rb-label{color:#4a4a5a;letter-spacing:0.08em;font-size:12px}
.rb-regime{font-weight:500;font-size:13px}
.rb-vol{font-size:13px}
.rb-macro{font-size:13px;color:#8b8b9e}
.rb-sep{color:#4a4a5a}

/* ── KPI strip — flat, no backgrounds ── */
.kpi-strip{
    display:flex;align-items:stretch;
    padding:8px 0 20px;
}
.kpi-item{
    flex:1;text-align:center;padding:0 10px;
}
.kpi-sep{
    width:1px;background:rgba(255,255,255,0.06);margin:2px 0;
}
.kpi-val{
    font-size:32px;font-weight:500;
    font-family:'IBM Plex Mono',monospace;
    line-height:1;margin-bottom:6px;
}
.kpi-lbl{
    font-size:11px;letter-spacing:0.12em;text-transform:uppercase;
    color:#4a4a5a;font-family:'IBM Plex Mono',monospace;font-weight:500;
}

/* ── Section labels ── */
.section-label{
    font-size:11px;letter-spacing:0.12em;text-transform:uppercase;
    color:#4a4a5a;font-family:'IBM Plex Mono',monospace;font-weight:500;
    border-bottom:1px solid rgba(255,255,255,0.14);
    padding-bottom:6px;margin-bottom:8px;
}
.sector-caption{
    font-size:12px;color:#4a4a5a;
    font-family:'Inter',sans-serif;
    text-align:center;margin-top:-4px;margin-bottom:6px;
}

/* ── Alert table header ── */
.ar-col-hdr{
    font-size:11px;letter-spacing:0.12em;text-transform:uppercase;
    color:#4a4a5a;font-family:'IBM Plex Mono',monospace;font-weight:500;
    padding-bottom:4px;white-space:nowrap;
}

/* ── Alert dot ── */
.ar-dot-wrap{
    display:flex;align-items:center;justify-content:center;
    height:40px;
}
.ar-dot{
    width:8px;height:8px;border-radius:50%;
    display:inline-block;flex-shrink:0;
}

/* ── Company name button — plain table-row style ── */
[data-testid="element-container"]:has(.al-row-mark) + [data-testid="stHorizontalBlock"] [data-testid="column"]:nth-child(2) [data-testid="stButton"] button{
    background:transparent!important;
    border:none!important;
    box-shadow:none!important;
    color:#f0f0f5!important;
    font-family:'Inter',sans-serif!important;
    font-size:14px!important;
    font-weight:400!important;
    text-align:left!important;
    justify-content:flex-start!important;
    padding:0!important;
    height:24px!important;
    min-height:24px!important;
    border-radius:0!important;
    letter-spacing:0!important;
    white-space:nowrap!important;
    overflow:hidden!important;
    text-overflow:ellipsis!important;
    width:100%!important;
    transition:color 0.12s!important;
}
[data-testid="element-container"]:has(.al-row-mark) + [data-testid="stHorizontalBlock"] [data-testid="column"]:nth-child(2) [data-testid="stButton"] button:hover{
    color:#00c4b4!important;
}

/* ── Ticker sub-label ── */
.ar-ticker{
    font-size:12px;color:#4a4a5a;
    font-family:'IBM Plex Mono',monospace;
    padding-bottom:4px;line-height:1;
}

/* ── Alert data cells ── */
.ar-price{
    font-size:13px;color:#f0f0f5;
    font-family:'IBM Plex Mono',monospace;
    padding-top:4px;
}
.ar-pct{font-size:12px}
.ar-pct.up{color:#00c4b4}
.ar-pct.dn{color:#e05252}

/* ── Signal pill (opacity fill, no border) ── */
.ar-sig-pill{
    font-family:'IBM Plex Mono',monospace;
    font-size:11px;font-weight:500;letter-spacing:0.04em;
    padding:2px 8px;border-radius:10px;
    display:inline-block;margin-top:5px;
    white-space:nowrap;
}

.ar-pd{
    font-family:'IBM Plex Mono',monospace;
    font-size:13px;display:inline-block;padding-top:5px;
}
.ar-empty{
    color:#4a4a5a;font-size:13px;
    padding:24px 0;font-family:'Inter',sans-serif;
}

/* ── Earnings badge ── */
.earn-badge{
    font-family:'IBM Plex Mono',monospace;
    font-size:11px;color:#c49a3c;
    font-weight:500;
}

/* ── Sector severity legend (horizontal) ── */
.sev-legend{
    display:flex;align-items:center;gap:18px;flex-wrap:nowrap;
    padding:4px 0 6px;
    font-family:'IBM Plex Mono',monospace;
}
.sev-leg-item{display:flex;align-items:center;gap:6px;white-space:nowrap}
.sev-leg-sw{
    width:10px;height:10px;border-radius:2px;
    display:inline-block;flex-shrink:0;
}
.sev-leg-tx{font-size:11px;font-weight:500;letter-spacing:0.06em}

</style>
"""
_CC_CSS = _CC_CSS.replace("#4a4a5a", TEXT_MUTED).replace("#8b8b9e", CHART_AXIS)


def render_command_center(decisions: pd.DataFrame, price_data: dict) -> None:
    st.markdown(_CC_CSS, unsafe_allow_html=True)

    latest      = _latest(decisions)
    macro_df    = load_macro_context()
    earnings_df = load_earnings_calendar()

    _regime_banner(decisions, macro_df)
    _kpi_strip(latest)

    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)

    left, right = st.columns([3, 2], gap="large")

    with left:
        _alert_list(latest, price_data, earnings_df)

    with right:
        _sector_bar(latest)
