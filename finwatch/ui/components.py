"""
FinWatch AI — UI Components
==============================
Sidebar, stock header, risk/news row, anomaly selector,
analysis panel, investor summary, and LLM report renderer.
"""

import ast
import re
import html as _html
import urllib.parse
import streamlit as st
import pandas as pd

from data.loader import (
    COMPANY_NAMES, SECTORS, SEV_CLS,
)
from analytics.analysis import (
    build_analysis, build_investor_summary, explain_anomaly,
)
from llm.translator import translate


def _safe_list(val):
    if isinstance(val, list): return val
    try: return ast.literal_eval(str(val))
    except: return []


# ── Language analysis modal ───────────────────────────────────────────────────

@st.dialog("AI Analyst Report", width="large")
def show_analysis_modal(ticker: str, news_df, lang: str) -> None:
    """Show the AI Analyst Report in a large modal in the selected language."""
    st.session_state.lang_modal = None   # reset so closing X doesn't reopen
    from llm.translator import translate
    if news_df is None or ticker not in news_df["ticker"].values:
        st.markdown(
            '<div style="font-size:11px;color:#5c7080;font-family:IBM Plex Mono">'
            'No LLM report available for this ticker.</div>',
            unsafe_allow_html=True,
        )
        return
    llm_text = news_df[news_df["ticker"] == ticker].iloc[0].get("llm_summary", "")
    if not llm_text or len(llm_text) < 120:
        st.markdown(
            '<div style="font-size:11px;color:#5c7080;font-family:IBM Plex Mono">'
            'LLM report not generated yet. Run the narrator pipeline first.</div>',
            unsafe_allow_html=True,
        )
        return
    lang_labels = {"english": "English", "german": "Deutsch", "arabic": "العربية"}
    st.markdown(
        f'<div style="font-size:9px;letter-spacing:2px;text-transform:uppercase;'
        f'color:#5c7080;font-family:IBM Plex Mono;margin-bottom:14px">'
        f'{ticker} · {lang_labels.get(lang, lang.upper())}</div>',
        unsafe_allow_html=True,
    )
    with st.spinner("Translating…" if lang != "english" else "Loading…"):
        display_text = translate(llm_text, lang, ticker)
    rtl = "direction:rtl;text-align:right;" if lang == "arabic" else ""
    safe_text = _md_to_html(display_text)
    st.markdown(
        f'<div style="font-size:12px;color:#adb5bd;line-height:1.9;'
        f'font-family:IBM Plex Sans,sans-serif;{rtl}">'
        f'{safe_text}</div>',
        unsafe_allow_html=True,
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar(decisions, price_data) -> None:
    """Render the full left sidebar: logo, sector selector, watchlist, language."""
    with st.sidebar:
        st.markdown('<div class="logo">Fin<span>Watch</span> AI</div>', unsafe_allow_html=True)

        if st.button("← Home", key="nav_home", use_container_width=True):
            st.session_state.page = "landing"
            st.rerun()
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

        active_spx = "★ " if st.session_state.get("selected") == "^SPX" else ""
        if st.button(f"{active_spx}S&P 500  ·  Market Overview", key="nav_spx", use_container_width=True):
            st.session_state.selected     = "^SPX"
            st.session_state.clicked_date = None
            st.rerun()
        st.markdown("<hr style='border-color:#1a2332;margin:4px 0 6px'>", unsafe_allow_html=True)

        # ── ESCALATE Alerts ───────────────────────────────────────────────────
        escalate_rows = decisions[decisions["action"] == "ESCALATE"]
        if not escalate_rows.empty:
            st.markdown(
                '<div style="background:rgba(248,81,73,0.07);border:1px solid rgba(248,81,73,0.22);'
                'border-radius:6px;padding:6px 8px 4px;margin-bottom:6px">'
                '<div style="font-size:8px;letter-spacing:2px;color:#f85149;'
                'font-family:\'IBM Plex Mono\',monospace;text-transform:uppercase;margin-bottom:4px">'
                '⚠ Escalate Alerts</div></div>',
                unsafe_allow_html=True,
            )
            for _, er in escalate_rows.iterrows():
                t    = er["ticker"]
                name = COMPANY_NAMES.get(t, t)
                st.markdown('<div class="escalate-btn">', unsafe_allow_html=True)
                if st.button(f"⚠ {name} ({t})", key=f"alert_{t}", use_container_width=True):
                    st.session_state.selected     = t
                    st.session_state.lang_modal   = st.session_state.get("language", "english")
                    st.session_state.anomaly_date = None
                    st.session_state.clicked_date = None
                    if st.session_state.get("page") != "stocks":
                        st.session_state.page = "stocks"
                    # update sector to match ticker
                    for s, ts in SECTORS.items():
                        if t in ts:
                            st.session_state.selected_sector = s
                            break
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
            st.markdown("<hr style='border-color:#1a2332;margin:4px 0 6px'>", unsafe_allow_html=True)

        sector_names = list(SECTORS.keys())
        cur_sector   = st.session_state.selected_sector
        if cur_sector not in sector_names:
            cur_sector = sector_names[0]
        chosen = st.selectbox("", sector_names,
                              index=sector_names.index(cur_sector),
                              label_visibility="collapsed",
                              key="sector_select")
        if chosen != st.session_state.selected_sector:
            st.session_state.selected_sector = chosen

        for t in SECTORS[chosen]:
            row_s = decisions[decisions["ticker"] == t]
            if row_s.empty:
                continue
            name   = COMPANY_NAMES.get(t, t)
            active = "★ " if t == st.session_state.selected else ""
            btn_col, chg_col = st.columns([7, 3])
            with btn_col:
                if st.button(f"{active}{name} ({t})", key=f"stock_{t}", use_container_width=True):
                    st.session_state.selected     = t
                    st.session_state.anomaly_date = None
                    st.session_state.clicked_date = None
                    st.rerun()
            with chg_col:
                if t in price_data:
                    _, pct = price_data[t]
                    arrow  = "▲" if pct > 0 else ("▼" if pct < 0 else "—")
                    css    = "wl-chg-up" if pct > 0 else ("wl-chg-dn" if pct < 0 else "wl-chg-fl")
                    st.markdown(f'<div class="{css}">{arrow}{abs(pct):.1f}%</div>',
                                unsafe_allow_html=True)

        st.markdown("<hr style='border-color:#1a2332;margin:8px 0'>", unsafe_allow_html=True)
        st.markdown('<div class="sb-label">Analysis Language</div>', unsafe_allow_html=True)
        lang_cols = st.columns(3)
        for col, (code, label) in zip(lang_cols, [("english","EN"), ("german","DE"), ("arabic","AR")]):
            with col:
                if st.button(label, key=f"lang_{code}", type="secondary"):
                    st.session_state.lang_modal = code
                    st.rerun()


# ── Stock header ──────────────────────────────────────────────────────────────

def render_stock_header(ticker: str, name: str, det_df) -> tuple:
    """Render the stock name / price header. Returns (last_price, last_chg)."""
    last_price, last_chg = None, None
    if det_df is not None and "Close" in det_df.columns and len(det_df) > 1:
        last_price = det_df["Close"].iloc[-1]
        last_chg   = (det_df["Close"].iloc[-1] - det_df["Close"].iloc[-2]) / det_df["Close"].iloc[-2] * 100

    chg_html   = ""
    price_html = ""
    if last_chg is not None:
        chg_html = (f'<span class="sh-up">▲ +{last_chg:.2f}%</span>'
                    if last_chg >= 0 else
                    f'<span class="sh-dn">▼ {last_chg:.2f}%</span>')
    if last_price:
        price_html = f'<span class="sh-price">${last_price:.2f}</span>'

    st.markdown(f"""
    <div class="stock-header">
      <span class="sh-name">{name}</span>
      <span class="sh-sub">{ticker} · NASDAQ</span>
      {price_html}
      {chg_html}
    </div>""", unsafe_allow_html=True)
    return last_price, last_chg


def render_spx_header(spx_df) -> None:
    """Render the S&P 500 header."""
    spx_last, spx_chg = None, None
    if spx_df is not None and len(spx_df) > 1:
        spx_last = spx_df["Close"].iloc[-1]
        spx_chg  = (spx_df["Close"].iloc[-1] - spx_df["Close"].iloc[-2]) / spx_df["Close"].iloc[-2] * 100

    chg_html   = ""
    price_html = f'<span class="sh-price">{spx_last:,.2f}</span>' if spx_last else ""
    if spx_chg is not None:
        chg_html = (f'<span class="sh-up">▲ +{spx_chg:.2f}%</span>'
                    if spx_chg >= 0 else
                    f'<span class="sh-dn">▼ {spx_chg:.2f}%</span>')

    st.markdown(f"""
    <div class="stock-header">
      <span class="sh-name">S&P 500</span>
      <span class="sh-sub">^GSPC · Market Index</span>
      {price_html}
      {chg_html}
    </div>""", unsafe_allow_html=True)


# ── Risk / Summary / News row ─────────────────────────────────────────────────

def render_risk_news_row(ticker: str, row, det_df, decisions, news_df) -> None:
    """Render the 3-column row: AI Risk Analysis | AI Summary | News Sentiment."""
    sev = row["severity"]
    _rc1, _rc2, _rc3 = st.columns(3, gap="small")

    with _rc1:
        sev_cls = SEV_CLS.get(sev, "s-normal")
        mom     = row.get("momentum_signal", "neutral")
        mom_arr   = "▲" if mom == "rising" else "▼" if mom == "falling" else "—"
        mom_cls   = "rp-up" if mom == "rising" else "rp-dn" if mom == "falling" else "rp-val"
        direction = row.get("direction", "stable") if "direction" in row.index else "stable"
        p_down    = row.get("p_down", 0) if "p_down" in row.index else 0
        dir_arr   = "▼" if direction == "down" else "▲" if direction == "up" else "—"
        dir_cls   = "rp-dn" if direction == "down" else "rp-up" if direction == "up" else "rp-val"
        drawdown, es_ratio = None, None
        if det_df is not None:
            if "max_drawdown_30d" in det_df.columns:
                v = det_df["max_drawdown_30d"].iloc[-1]
                if not pd.isna(v): drawdown = v
            if "es_ratio" in det_df.columns:
                v = det_df["es_ratio"].iloc[-1]
                if not pd.isna(v): es_ratio = v
        dd_html = (f'<span class="rp-val rp-dn">{drawdown*100:.1f}%</span>'
                   if drawdown else '<span class="rp-val">—</span>')
        es_html = (f'<span class="rp-val rp-dn">{es_ratio:.2f}</span>'
                   if es_ratio else '<span class="rp-val">—</span>')

        # Anomaly detection row — only if ≥1 detector fired today
        anom_count = 0
        if det_df is not None and not det_df.empty:
            _last = det_df.iloc[-1]
            for _c in ["z_anomaly", "z_anomaly_60", "if_anomaly", "ae_anomaly"]:
                if _c in _last.index and bool(_last.get(_c)):
                    anom_count += 1
        anom_row_html = ""
        if anom_count > 0:
            _dots = (
                "".join(['<span class="d d-on"></span>' for _ in range(anom_count)])
                + "".join(['<span class="d d-off"></span>' for _ in range(4 - anom_count)])
            )
            anom_row_html = (
                '<div class="rp-row"><span class="rp-key">Anomaly detection</span>'
                f'<div class="dot-row">{_dots}'
                f'<span style="font-size:9px;color:#e3b341;margin-left:4px">{anom_count}/4</span>'
                '</div></div>'
            )

        st.markdown(f"""
        <div class="rp-section">
          <div class="rp-title">AI Risk Analysis</div>
          <div class="sev-big {sev_cls}">{sev.replace('_',' ')}</div>
          <div style="font-size:9px;color:#5c7080;margin-bottom:10px;font-family:IBM Plex Mono">{row.get('action','—')} · {row.get('date','—')}</div>
          {anom_row_html}
          <div class="rp-row"><span class="rp-key">Direction (5D)</span>
            <span class="rp-val {dir_cls}">{dir_arr} {direction} ({int(p_down*100)}%)</span></div>
          <div class="rp-row"><span class="rp-key">Momentum</span>
            <span class="rp-val {mom_cls}">{mom_arr} {mom}</span></div>
          <div class="rp-row"><span class="rp-key">ES Ratio</span>{es_html}</div>
          <div class="rp-row"><span class="rp-key">Drawdown 30D</span>{dd_html}</div>
        </div>""", unsafe_allow_html=True)

    with _rc2:
        summary = row.get("summary", "")
        caution = row.get("caution_flag", None)
        if summary:
            caution_html = (
                f'<div style="color:#e3b341;font-size:10px;margin-top:6px;'
                f'font-family:IBM Plex Mono">⚠ {caution}</div>'
                if caution else ""
            )
            st.markdown(f"""
            <div class="rp-section">
              <div class="rp-title">AI Summary</div>
              <div class="summary-text">{summary}{caution_html}</div>
            </div>""", unsafe_allow_html=True)

    with _rc3:
        if news_df is not None and ticker in news_df["ticker"].values:
            nr         = news_df[news_df["ticker"] == ticker].iloc[0]
            headlines  = _safe_list(nr.get("top_news", []))
            sentiments = _safe_list(nr.get("news_sentiment", []))
            sources    = _safe_list(nr.get("news_sources", nr.get("news_urls", [])))
            if headlines:
                news_html = ""
                for i, h in enumerate(headlines[:3]):
                    sent    = sentiments[i] if i < len(sentiments) else "neutral"
                    s_cls   = {"positive": "s-pos", "negative": "s-neg", "neutral": "s-neu"}.get(sent, "s-neu")
                    source  = sources[i] if i < len(sources) and sources[i] else ""
                    if source.startswith("http"):
                        source = urllib.parse.urlparse(source).netloc.replace("www.", "") or ""
                    source_html = (
                        f'<span style="font-size:9px;color:#5c7080;font-family:IBM Plex Mono;'
                        f'display:inline-block;margin-top:3px">Source: {source}</span>'
                    ) if source else ""
                    news_html += (
                        f'<div class="news-item">'
                        f'<span class="news-sent {s_cls}">{sent.upper()}</span>'
                        f'<div class="news-text">{h}</div>'
                        f'{source_html}</div>'
                    )
                st.markdown(
                    f'<div class="rp-section"><div class="rp-title">News Sentiment</div>'
                    f'{news_html}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<div class="rp-section"><div class="rp-title">News Sentiment</div>'
                '<div style="font-size:10px;color:#5c7080;font-family:IBM Plex Mono">'
                'Run FinBERT pipeline first</div></div>',
                unsafe_allow_html=True,
            )


# ── Anomaly selector ──────────────────────────────────────────────────────────

_ANOMALY_PERIOD_OFFSETS = {
    "1M": pd.DateOffset(months=1),
    "3M": pd.DateOffset(months=3),
    "6M": pd.DateOffset(months=6),
    "1Y": pd.DateOffset(years=1),
    "All": None,
}
_ANOMALY_DET_COLS = ["z_anomaly", "z_anomaly_60", "if_anomaly", "ae_anomaly"]


def render_anomaly_selector(det_df, name: str, period: str = "1M") -> None:
    """Render the anomaly date dropdown filtered to the active chart period."""
    plot_df = det_df.copy()

    # Filter to active period
    available = [c for c in _ANOMALY_DET_COLS if c in plot_df.columns]
    if not available:
        return

    x_end  = plot_df["Date"].iloc[-1]
    offset = _ANOMALY_PERIOD_OFFSETS.get(period)
    if offset is not None:
        plot_df = plot_df[plot_df["Date"] >= (x_end - offset)]

    # Only dates where ≥2 detectors fired
    confirmed = plot_df[available].apply(
        lambda row: sum(bool(v) for v in row), axis=1
    )
    anom_dates = (
        plot_df[confirmed >= 2]["Date"]
        .dt.strftime("%Y-%m-%d")
        .tolist()
    )
    if not anom_dates:
        return

    # Reset selected date if it's outside the current period
    if st.session_state.anomaly_date not in anom_dates:
        st.session_state.anomaly_date = None

    options = ["— select anomaly date —"] + anom_dates
    cur_idx = 0
    if st.session_state.anomaly_date in anom_dates:
        cur_idx = anom_dates.index(st.session_state.anomaly_date) + 1
    chosen = st.selectbox("Anomaly", options, index=cur_idx,
                          label_visibility="collapsed", key="anomaly_select")
    st.session_state.anomaly_date = chosen if chosen != "— select anomaly date —" else None

    if st.session_state.anomaly_date:
        adate   = pd.Timestamp(st.session_state.anomaly_date)
        arow_df = det_df[det_df["Date"].dt.date == adate.date()]
        if not arow_df.empty:
            explanation = explain_anomaly(arow_df.iloc[0], name, st.session_state.anomaly_date)
            if explanation:
                st.markdown(explanation, unsafe_allow_html=True)


# ── Candle detail panel ───────────────────────────────────────────────────────

def render_candle_panel(det_df, clicked_date: str) -> None:
    """Horizontal strip below the chart showing candle details on click."""
    if clicked_date is None or det_df is None:
        return

    date_ts = pd.Timestamp(clicked_date)
    row_df  = det_df[det_df["Date"].dt.date == date_ts.date()]
    if row_df.empty:
        return
    r = row_df.iloc[0]

    close_v = r.get("Close", None)
    open_v  = r.get("Open", None)
    high_v  = r.get("High", None)
    low_v   = r.get("Low", None)
    rsi_v   = r.get("rsi", None)
    vol_v   = r.get("Volume", None)
    ret_v   = r.get("returns", None)

    idx     = row_df.index[0]
    ema20_v = det_df.loc[:idx, "Close"].ewm(span=20).mean().iloc[-1]

    _det_map = [("z_anomaly","Z-30D"),("z_anomaly_60","Z-60D"),
                ("if_anomaly","IsoForest"),("ae_anomaly","LSTM-AE")]
    fired_labels = [lbl for col, lbl in _det_map if bool(r.get(col, False))]

    _TIPS = {
        "O":       "Open — price at market open",
        "H":       "High — highest price of the day",
        "L":       "Low — lowest price of the day",
        "C":       "Close — price at market close",
        "Close":   "Close — price at market close",
        "EMA 20":  "EMA 20 — Exponential Moving Average over 20 days. Smoothed trend line; price above = bullish.",
        "Δ 1D":    "Daily change — % price change vs previous close",
        "RSI":     "RSI — Relative Strength Index (0–100). Above 70 = overbought, below 30 = oversold.",
        "Vol":     "Volume — number of shares traded on this day",
        "ANOMALY": "Anomaly — models that flagged unusual price or volume behavior on this day",
    }

    def _cell(label, value, color="#8b949e"):
        tip = _TIPS.get(label, "")
        tip_attr = f' data-tip="{tip}"' if tip else ""
        return (
            '<div style="display:flex;flex-direction:column;align-items:center;'
            'padding:0 12px;border-right:1px solid #1a2332">'
            f'<span style="font-size:8px;color:#5c7080;font-family:IBM Plex Mono;'
            f'letter-spacing:1px;margin-bottom:2px;cursor:help"{tip_attr}>{label}</span>'
            f'<span style="font-size:11px;color:{color};font-family:IBM Plex Mono;'
            f'font-weight:500">{value}</span>'
            '</div>'
        )

    ret_color = "#1de9b6" if (close_v or 0) >= (open_v or close_v or 0) else "#f85149"

    cells = f'<div style="font-size:10px;color:#cdd9e5;font-family:IBM Plex Mono;font-weight:500;padding:0 12px;border-right:1px solid #1a2332;align-self:center">{clicked_date}</div>'

    if open_v is not None:
        cells += _cell("O", f"${open_v:,.2f}")
        cells += _cell("H", f"${high_v:,.2f}", "#1de9b6")
        cells += _cell("L", f"${low_v:,.2f}", "#f85149")
        cells += _cell("C", f"${close_v:,.2f}", ret_color)
    elif close_v is not None:
        cells += _cell("Close", f"${close_v:,.2f}", ret_color)

    cells += _cell("EMA 20", f"${ema20_v:,.2f}", "#a371f7")

    if ret_v is not None:
        rc  = "#1de9b6" if ret_v >= 0 else "#f85149"
        sym = "▲" if ret_v >= 0 else "▼"
        cells += _cell("Δ 1D", f"{sym}{abs(ret_v)*100:.2f}%", rc)
    if rsi_v is not None:
        rc = "#f85149" if rsi_v > 70 else "#3fb950" if rsi_v < 30 else "#8b949e"
        cells += _cell("RSI", f"{rsi_v:.1f}", rc)
    if vol_v is not None:
        vf = f"{vol_v/1e6:.1f}M" if vol_v >= 1e6 else f"{vol_v/1e3:.0f}K"
        cells += _cell("Vol", vf)

    if fired_labels:
        anom_txt = " · ".join(fired_labels)
        tip = _TIPS["ANOMALY"]
        cells += (
            '<div style="display:flex;flex-direction:column;align-items:center;'
            'padding:0 12px;border-right:1px solid #1a2332">'
            f'<span style="font-size:8px;color:#5c7080;font-family:IBM Plex Mono;'
            f'letter-spacing:1px;margin-bottom:2px;cursor:help" data-tip="{tip}">ANOMALY</span>'
            f'<span style="font-size:10px;color:#e3b341;font-family:IBM Plex Mono">{anom_txt}</span>'
            '</div>'
        )

    _sc, _xc = st.columns([50, 1], gap="small")
    with _sc:
        st.markdown(
            '<div style="background:#0d1117;border:1px solid #1a2332;border-radius:3px;'
            'padding:6px 4px;display:flex;flex-direction:row;align-items:stretch;overflow-x:auto">'
            f'{cells}'
            '</div>',
            unsafe_allow_html=True,
        )
    with _xc:
        if st.button("✕", key="clear_candle", use_container_width=True,
                     help="Clear selection"):
            st.session_state.clicked_date = None
            st.rerun()


# ── Analysis panel + LLM report ───────────────────────────────────────────────

def render_strategy_box(det_df, dec_row) -> None:
    """Rule-based investment strategy: two scenarios (not holding / holding)."""
    if det_df is None or det_df.empty or dec_row is None or dec_row.empty:
        return

    last      = det_df.iloc[-1]
    row       = dec_row.iloc[0]
    close     = float(last.get("Close", 0))
    ema20     = float(det_df["Close"].ewm(span=20).mean().iloc[-1])
    rsi       = float(last.get("rsi", 50))
    mom5      = float(last.get("momentum_5", 0))
    mom10     = float(last.get("momentum_10", 0))
    p_down    = float(row.get("p_down", 0.33))
    p_up      = max(0.0, 1 - p_down - 0.15)
    direction = str(row.get("direction", "stable"))
    mom_sig   = str(row.get("momentum_signal", "neutral"))
    sev       = str(row.get("severity", "NORMAL"))

    above_ema = close > ema20
    confirmed_det = [c for c in ["z_anomaly","z_anomaly_60","if_anomaly","ae_anomaly"]
                     if c in last.index and bool(last.get(c))]
    n_anom = len(confirmed_det)

    # ── NOT HOLDING logic ─────────────────────────────────────────────────────
    buy_signals  = sum([above_ema, rsi > 45, mom_sig == "rising", p_up > 0.50])
    risk_signals = sum([sev in ("CRITICAL","WARNING"), n_anom >= 2, p_down > 0.55])

    if risk_signals >= 2:
        nh_action = "AVOID"
        nh_color  = "#f85149"
        nh_reason = f"Too risky to enter — {sev} severity" + (f", {n_anom} anomaly detectors" if n_anom >= 2 else "")
    elif buy_signals >= 3:
        nh_action = "WATCH FOR ENTRY"
        nh_color  = "#1de9b6"
        nh_reason = "Price above EMA20, momentum positive" + (", RSI strong" if rsi > 50 else "") + ". Wait for next green candle to confirm."
    elif rsi < 35 and mom_sig == "rising":
        nh_action = "POTENTIAL ENTRY"
        nh_color  = "#e3b341"
        nh_reason = f"Oversold (RSI {rsi:.0f}) with rising momentum — possible recovery forming. Wait for close above EMA20."
    elif above_ema and mom_sig != "falling":
        nh_action = "WATCH"
        nh_color  = "#58a6ff"
        nh_reason = f"Above EMA20 but momentum not confirmed. Wait for RSI > 50 and momentum turn."
    else:
        nh_action = "WAIT"
        nh_color  = "#5c7080"
        cond = []
        if not above_ema: cond.append("price above EMA20")
        if rsi < 45:      cond.append(f"RSI > 45 (now {rsi:.0f})")
        if mom_sig == "falling": cond.append("momentum to stop falling")
        nh_reason = "Wait for: " + " + ".join(cond) if cond else "No clear entry signal."

    # ── HOLDING logic ─────────────────────────────────────────────────────────
    sell_signals = sum([
        direction == "down" and p_down > 0.55,
        mom_sig == "falling",
        not above_ema,
        sev in ("CRITICAL", "WARNING"),
        n_anom >= 2,
    ])

    if sell_signals >= 4:
        h_action = "REDUCE / EXIT"
        h_color  = "#f85149"
        h_reason = f"P(down)={p_down*100:.0f}%, below EMA20, falling momentum" + (f", {n_anom} anomaly models triggered" if n_anom >= 2 else "") + ". Cut or trim."
    elif sell_signals >= 2:
        h_action = "TRIM"
        h_color  = "#e3b341"
        h_reason = f"P(down)={p_down*100:.0f}% with {sell_signals} bearish signals. Consider reducing size."
    elif direction == "up" and p_up > 0.50 and above_ema:
        h_action = "HOLD"
        h_color  = "#1de9b6"
        h_reason = f"P(up)={p_up*100:.0f}%, above EMA20, direction positive. Stay in."
    else:
        h_action = "HOLD & MONITOR"
        h_color  = "#58a6ff"
        h_reason = "Mixed signals. Keep position but watch EMA20 and momentum for deterioration."

    st.markdown(f"""
    <div style="background:#0d1117;border:1px solid #1a2332;border-radius:3px;
                padding:10px 16px;margin-top:8px">
      <div style="font-size:9px;letter-spacing:2px;text-transform:uppercase;
                  color:#5c7080;font-family:'IBM Plex Mono',monospace;
                  border-bottom:1px solid #1a2332;padding-bottom:6px;margin-bottom:8px">
        Investment Strategy
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">
        <div>
          <div style="font-size:8px;color:#5c7080;font-family:'IBM Plex Mono',monospace;
                      letter-spacing:1px;margin-bottom:4px">NOT HOLDING</div>
          <div style="font-size:13px;font-weight:600;color:{nh_color};
                      font-family:'IBM Plex Mono',monospace;margin-bottom:4px">{nh_action}</div>
          <div style="font-size:10px;color:#8b949e;line-height:1.5">{nh_reason}</div>
        </div>
        <div style="border-left:1px solid #1a2332;padding-left:12px">
          <div style="font-size:8px;color:#5c7080;font-family:'IBM Plex Mono',monospace;
                      letter-spacing:1px;margin-bottom:4px">HOLDING</div>
          <div style="font-size:13px;font-weight:600;color:{h_color};
                      font-family:'IBM Plex Mono',monospace;margin-bottom:4px">{h_action}</div>
          <div style="font-size:10px;color:#8b949e;line-height:1.5">{h_reason}</div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)


def render_analysis_panel(det_df, dec_row, ticker: str, name: str, lang: str) -> None:
    """Render the 4-column HTML analysis grid + risk drivers as separate elements."""
    result = build_analysis(det_df, dec_row, ticker, name, lang)
    if result is None:
        return
    main_html, drivers_html = result
    st.markdown(main_html, unsafe_allow_html=True)
    if drivers_html:
        st.markdown(drivers_html, unsafe_allow_html=True)


def render_investor_summary(det_df, dec_row, row, news_df, ticker: str) -> None:
    """Render the short investor signal summary."""
    html = build_investor_summary(det_df, dec_row, row, news_df, ticker)
    if html:
        st.markdown(html, unsafe_allow_html=True)


def _md_to_html(text: str) -> str:
    """Safely convert markdown text to HTML for embedding inside a div."""
    escaped = _html.escape(text)
    escaped = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', escaped)
    escaped = re.sub(r'^---\s*$', '<hr style="border:none;border-top:1px solid #1a2332;margin:8px 0">', escaped, flags=re.MULTILINE)
    escaped = escaped.replace('\n', '<br>')
    return escaped


def render_llm_report(ticker: str, news_df, lang: str, dec_row=None) -> None:
    """Render the full LLM analyst report, with translation if needed.
    Falls back to the decision summary when no proper LLM report exists."""
    llm_text  = ""
    fallback  = False

    if news_df is not None and ticker in news_df["ticker"].values:
        llm_text = news_df[news_df["ticker"] == ticker].iloc[0].get("llm_summary", "") or ""

    if len(llm_text) < 120:
        # Try decision summary as fallback
        if dec_row is not None and not dec_row.empty:
            fb = dec_row.iloc[0].get("summary", "") or ""
            if len(fb) > 10:
                llm_text = fb
                fallback = True
        if not fallback:
            return

    with st.spinner("Translating..." if (lang != "english" and not fallback) else ""):
        display_text = translate(llm_text, lang, ticker) if not fallback else llm_text

    rtl_style    = "direction:rtl;text-align:right;" if lang == "arabic" else ""
    report_label = "AI Risk Summary" if fallback else "AI Analyst Report"
    note_html    = (
        '<div style="font-size:9px;color:#5c7080;font-family:IBM Plex Mono;'
        'margin-top:8px">Full LLM report not generated yet — re-run the narrator pipeline.</div>'
    ) if fallback else ""

    safe_text = _md_to_html(display_text)

    st.markdown(f"""
    <div style="background:#0d1117;border:1px solid #1a2332;border-radius:3px;
                padding:14px 18px;margin-top:8px">
      <div style="font-size:9px;letter-spacing:2px;text-transform:uppercase;
                  color:#5c7080;font-family:'IBM Plex Mono',monospace;
                  border-bottom:1px solid #1a2332;padding-bottom:7px;
                  margin-bottom:10px">{report_label}</div>
      <div style="font-size:11px;color:#adb5bd;line-height:1.8;
                  font-family:'IBM Plex Sans',sans-serif;{rtl_style}">{safe_text}</div>
      {note_html}
    </div>""", unsafe_allow_html=True)
