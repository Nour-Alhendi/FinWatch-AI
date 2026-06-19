"""
FinWatch AI — Entry Point (v2)
Run: streamlit run finwatch/app.py  (from project root)

Three screens: Command Center · Stock Deep-Dive · AI Analyst
"""

import sys
from pathlib import Path

_FINWATCH_DIR = Path(__file__).resolve().parent
_ROOT         = _FINWATCH_DIR.parent
for _p in (str(_FINWATCH_DIR), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import streamlit as st
from dotenv import load_dotenv

load_dotenv(_ROOT / ".env")

st.set_page_config(page_title="FinWatch AI", page_icon="📡", layout="wide")

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500&family=IBM+Plex+Mono:wght@400;500&display=swap');

/* ── Base ── */
html,body,[class*="css"]{
    font-family:'Inter',sans-serif;
    background:#0a0a0f;
    color:#f0f0f5;
    font-size:14px;
}
[data-testid="stAppViewContainer"]{background:#0a0a0f}

/* ── Hide Streamlit chrome ── */
header[data-testid="stHeader"]{display:none!important}
#MainMenu{visibility:hidden!important}
[data-testid="stToolbar"]{display:none!important}
footer{visibility:hidden!important}
[data-testid="stDecoration"]{display:none!important}
[data-testid="stSidebarCollapseButton"]{display:none!important}

/* ── Main container ── */
[data-testid="stMainBlockContainer"],.block-container{
    padding-top:1.5rem!important;
    padding-bottom:2rem!important;
    padding-left:1.5rem!important;
    padding-right:1.5rem!important;
    max-width:1400px!important;
}
section.main > div{padding-top:0!important}
[data-testid="stVerticalBlock"]{gap:0.5rem!important}
[data-testid="element-container"]{margin-bottom:0!important}

/* ── Sidebar ── */
[data-testid="stSidebar"]{
    background:#0a0a0f!important;
    border-right:1px solid rgba(255,255,255,0.14)!important;
    width:220px!important;min-width:220px!important;
}
[data-testid="stSidebar"] > div:first-child{width:220px!important;min-width:220px!important}
[data-testid="stSidebar"] .block-container{padding-top:0!important}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"]{gap:0!important}
[data-testid="stSidebar"] .stButton{margin:0!important;padding:0!important}
[data-testid="stSidebar"] button{
    background:transparent!important;
    border:none!important;
    border-left:2px solid transparent!important;
    border-radius:0!important;
    color:#8b8b9e!important;
    font-family:'Inter',sans-serif!important;
    font-size:14px!important;
    font-weight:400!important;
    text-align:left!important;
    padding:0 16px!important;
    width:100%!important;
    height:40px!important;
    min-height:40px!important;
    line-height:40px!important;
    transition:background 0.12s,color 0.12s,border-color 0.12s!important;
    letter-spacing:0!important;
}
[data-testid="stSidebar"] button:hover{
    background:rgba(255,255,255,0.03)!important;
    color:#f0f0f5!important;
    border-left:2px solid transparent!important;
}

/* Active nav — 2px teal bar + subtle teal background */
[data-testid="stSidebar"] [data-testid="element-container"]:has(.nav-active-mark) + [data-testid="element-container"] button{
    color:#f0f0f5!important;
    background:rgba(0,196,180,0.06)!important;
    border-left:2px solid #00c4b4!important;
    font-weight:500!important;
}

/* ── Logo ── */
.logo{
    font-family:'IBM Plex Mono',monospace;
    font-size:22px;font-weight:700;
    display:block;
    padding:28px 20px 12px;
    letter-spacing:-0.5px;
    color:#f0f0f5;
    line-height:1;
    white-space:nowrap;
}
.logo .logo-ai{color:#00c4b4}
.sb-label{
    display:block;
    font-size:10px;letter-spacing:0.14em;color:#8080a4;
    text-transform:uppercase;
    padding:12px 20px 6px;
    font-family:'IBM Plex Mono',monospace;
    font-weight:500;
}
.sb-divider{height:1px;background:rgba(255,255,255,0.14);margin:8px 0}
.sb-stock-label{
    display:block;
    font-size:10px;letter-spacing:0.12em;color:#8080a4;
    text-transform:uppercase;padding:8px 20px 8px;
    font-family:'IBM Plex Mono',monospace;font-weight:500;
}
/* Selectbox label in sidebar — styled to match sb-stock-label */
[data-testid="stSidebar"] [data-testid="stSelectbox"] label,
[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-testid="stWidgetLabel"] p{
    font-size:10px!important;letter-spacing:0.12em!important;
    color:#8080a4!important;text-transform:uppercase!important;
    font-family:'IBM Plex Mono',monospace!important;font-weight:500!important;
    padding:8px 0 6px 20px!important;margin:0!important;display:block!important;
}

/* ── Tooltips ── */
.tip,[data-tip]{border-bottom:1px dotted rgba(90,122,180,0.45);cursor:help;position:relative}
[data-tip]::after{
    content:attr(data-tip);position:absolute;left:0;top:120%;
    background:#111118;color:#8b8b9e;font-size:12px;line-height:1.6;
    padding:8px 14px;border-radius:8px;white-space:normal;
    max-width:280px;min-width:160px;
    border:1px solid rgba(255,255,255,0.14);
    z-index:9999;visibility:hidden;opacity:0;
    transition:opacity 0.15s;pointer-events:none;
    font-family:'Inter',sans-serif;font-weight:400;
}
[data-tip]:hover::after{visibility:visible;opacity:1}

/* ── Page title bar ── */
.page-title-bar{
    padding:8px 0 12px;
    border-bottom:1px solid rgba(255,255,255,0.14);
    margin-bottom:16px;
}
.page-title{
    font-size:20px;font-weight:500;color:#f0f0f5;
    font-family:'IBM Plex Mono',monospace;letter-spacing:-0.2px;line-height:1.2;
}
.page-sub{
    font-size:11px;letter-spacing:0.12em;color:#8080a4;
    font-family:'IBM Plex Mono',monospace;text-transform:uppercase;margin-top:4px;
    font-weight:500;
}

/* ── Severity colors ── */
.s-critical{color:#e05252}
.s-warning{color:#c49a3c}
.s-watch{color:#5a7ab4}
.s-normal{color:#4a9e6a}
.s-positive{color:#4a9e6a}
.s-review{color:#7a6ab4}

/* ── Card surface ── */
.fw-card{
    background:#111118;
    border:1px solid rgba(255,255,255,0.14);
    border-radius:8px;
    padding:16px;
}

/* ── Buttons (main content area — NOT sidebar) ── */
[data-testid="stMain"] [data-testid="stButton"] button,
[data-testid="stMainBlockContainer"] [data-testid="stButton"] button{
    background:transparent!important;
    border:1px solid rgba(255,255,255,0.18)!important;
    border-radius:6px!important;
    color:#8b8b9e!important;
    font-family:'Inter',sans-serif!important;
    font-size:13px!important;
    font-weight:400!important;
    transition:border-color 0.12s,color 0.12s!important;
}
[data-testid="stMain"] [data-testid="stButton"] button:hover,
[data-testid="stMainBlockContainer"] [data-testid="stButton"] button:hover{
    border-color:#00c4b4!important;
    color:#f0f0f5!important;
    background:transparent!important;
}

/* ── Inputs / selectbox ── */
[data-testid="stSelectbox"] > div > div{
    background:#111118!important;
    border:1px solid rgba(255,255,255,0.18)!important;
    color:#f0f0f5!important;
    font-size:13px!important;
    font-family:'Inter',sans-serif!important;
}
[data-testid="stSelectbox"] > div > div:focus-within{
    border-color:#00c4b4!important;
    box-shadow:none!important;
}
[data-testid="stTextInput"] input{
    background:#16161f!important;
    border:1px solid rgba(255,255,255,0.18)!important;
    color:#f0f0f5!important;
    font-size:13px!important;
    font-family:'Inter',sans-serif!important;
    border-radius:8px!important;
}
[data-testid="stTextInput"] input:focus{
    border-color:#00c4b4!important;
    box-shadow:none!important;
    outline:none!important;
}

/* ── Segmented control ── */
[data-testid="stSegmentedControl"] > div{
    background:#111118!important;
    border:1px solid rgba(255,255,255,0.18)!important;
    border-radius:6px!important;
}
[data-testid="stSegmentedControl"]{margin-bottom:6px!important}

/* ── Remove red focus rings ── */
*:focus{outline:none!important;box-shadow:none!important}
*:focus-visible{outline:1px solid rgba(0,196,180,0.4)!important;box-shadow:none!important}

/* ── Scrollbar ── */
::-webkit-scrollbar{width:4px;height:4px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:rgba(255,255,255,0.1);border-radius:2px}
::-webkit-scrollbar-thumb:hover{background:rgba(255,255,255,0.15)}

/* ── Language toggle (top-right) ── */
[data-testid="element-container"]:has(.lang-top-mark) + [data-testid="element-container"] [data-testid="stSegmentedControl"]{
    margin-top:10px!important;
}
[data-testid="element-container"]:has(.lang-top-mark) + [data-testid="element-container"] [data-testid="stSegmentedControl"] > div{
    background:transparent!important;
    border:1px solid rgba(255,255,255,0.18)!important;
    border-radius:4px!important;
    padding:1px!important;
    height:26px!important;
    min-height:26px!important;
}
[data-testid="element-container"]:has(.lang-top-mark) + [data-testid="element-container"] [data-testid="stSegmentedControl"] label{
    font-size:11px!important;
    font-family:'IBM Plex Mono',monospace!important;
    letter-spacing:0.08em!important;
    padding:2px 8px!important;
    min-height:22px!important;
    line-height:22px!important;
}

/* ── Responsive: narrow viewport ── */
@media(max-width:1100px){
    .kpi-val{font-size:24px!important}
    .regime-banner{font-size:12px!important;gap:7px!important}
}
@media(max-width:900px){
    .kpi-val{font-size:20px!important}
    .ar-price{font-size:12px!important}
    .ar-col-hdr{font-size:10px!important}
    .ar-sig-pill{font-size:10px!important}
    .regime-banner{font-size:11px!important;gap:5px!important}
}
/* Alert cells: prevent hard overflow, allow ellipsis */
.ar-price,.ar-pd{min-width:0;overflow:hidden;text-overflow:ellipsis}
.ar-pct{display:block}
.kpi-item{min-width:70px}
</style>
""", unsafe_allow_html=True)

# ── Imports ────────────────────────────────────────────────────────────────────
from data.loader import (
    COMPANY_NAMES, load_decisions, load_news, load_price_summary,
)
from ui.i18n           import t as _t
from ui.command_center import render_command_center
from ui.deep_dive      import render_deep_dive, render_metric_cards
from ui.portfolio_page import render_portfolio_page
from ui.chat_panel     import render_chat_panel
from data.portfolio    import load_portfolios

# ── Session state ──────────────────────────────────────────────────────────────
_defaults = {
    "page":               "command_center",
    "selected":           None,
    "ai_agent":           None,
    "agent_error":        None,
    "panel_chat_history": [],
    "panel_chat_context": None,
    "portfolios":         None,
    "active_portfolio":   None,
    "dd_period":          "1M",
    "lang":               "en",
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

if st.session_state.portfolios is None:
    st.session_state.portfolios = load_portfolios()

# ── Load shared data ───────────────────────────────────────────────────────────
decisions  = load_decisions()
price_data = load_price_summary()
news_df    = load_news()

all_tickers = [t for t in decisions["ticker"].tolist() if not t.startswith("^")]
if st.session_state.selected is None and all_tickers:
    st.session_state.selected = all_tickers[0]

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="logo">FinWatch<span class="logo-ai"> AI</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-divider" style="margin:0 20px 2px"></div>', unsafe_allow_html=True)


    _page = st.session_state.page
    for _key, _label, _page_id in [
        ("nav_cc",  "Command Center",  "command_center"),
        ("nav_dd",  "Stock Deep-Dive", "deep_dive"),
        ("nav_pf",  "My Portfolio",    "portfolio"),
    ]:
        if _page == _page_id:
            st.markdown('<span class="nav-active-mark"></span>', unsafe_allow_html=True)
        if st.button(_label, key=_key, use_container_width=True):
            st.session_state.page = _page_id
            st.rerun()

    # Stock selector (only on deep-dive)
    if st.session_state.page == "deep_dive":
        st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)
        # Sort: non-SPX tickers, severity-sorted
        dec_latest = decisions.copy()
        dec_latest["_so"] = dec_latest["severity"].map(
            {"CRITICAL": 0, "WARNING": 1, "WATCH": 2, "REVIEW": 3, "POSITIVE_MOMENTUM": 4, "NORMAL": 5}
        ).fillna(99)
        dec_latest = (dec_latest[~dec_latest["ticker"].str.startswith("^")]
                      .sort_values("_so")
                      .drop_duplicates("ticker"))
        ordered_tickers = dec_latest["ticker"].tolist()

        current = st.session_state.selected
        idx = ordered_tickers.index(current) if current in ordered_tickers else 0
        selected = st.selectbox(
            _t("select_stock_label"),
            ordered_tickers,
            index=idx,
            key="ticker_select",
            label_visibility="visible",
            format_func=lambda t: f"{COMPANY_NAMES.get(t, t)} ({t})",
        )
        if selected != st.session_state.selected:
            st.session_state.selected = selected
            st.rerun()

# ── JS tooltip engine ──────────────────────────────────────────────────────────
import streamlit.components.v1 as _comp
_comp.html("""<script>
(function(){
    var par=window.parent;
    if(par._fwTip)return;
    par._fwTip=true;
    var doc=par.document;
    doc.querySelectorAll('#fw-tip').forEach(function(e){e.remove()});
    var tip=doc.createElement('div');
    tip.id='fw-tip';
    tip.style.cssText='position:fixed;background:#1c2a3a;border:1px solid #3a5068;border-radius:6px;padding:6px 12px;font-size:11px;font-family:IBM Plex Mono,monospace;color:#fff;max-width:260px;pointer-events:none;opacity:0;transition:opacity 0.12s;z-index:999999;line-height:1.5;box-shadow:0 4px 20px rgba(0,0,0,0.8)';
    doc.body.appendChild(tip);
    function attach(){
        doc.querySelectorAll('.tip[data-tip]').forEach(function(el){
            if(el._fw)return;el._fw=true;
            el.addEventListener('mouseover',function(e){tip.textContent=el.getAttribute('data-tip');tip.style.opacity='1';move(e)});
            el.addEventListener('mousemove',move);
            el.addEventListener('mouseout',function(){tip.style.opacity='0'});
        });
    }
    function move(e){
        var x=e.clientX+14,y=e.clientY-44;
        if(x+280>par.innerWidth)x=e.clientX-294;
        if(y<8)y=e.clientY+18;
        tip.style.left=x+'px';tip.style.top=y+'px';
    }
    attach();
    new MutationObserver(attach).observe(doc.body,{childList:true,subtree:true});
})();
</script>""", height=0)

# ── Page title bar ─────────────────────────────────────────────────────────────
_page_titles = {
    "command_center": (_t("page_cc_title"), _t("page_cc_sub")),
    "deep_dive":      (_t("page_dd_title"), f"{st.session_state.selected or '—'} · {_t('page_dd_sub')}"),
    "portfolio":      (_t("page_pf_title"), _t("page_pf_sub")),
}
_title, _sub = _page_titles.get(st.session_state.page, ("FinWatch AI", ""))
_ptc, _plc = st.columns([8, 1], gap="small")
with _ptc:
    st.markdown(
        f'<div class="page-title-bar"><div class="page-title">{_title}</div>'
        f'<div class="page-sub">{_sub}</div></div>',
        unsafe_allow_html=True,
    )
with _plc:
    st.markdown('<span class="lang-top-mark"></span>', unsafe_allow_html=True)
    _lang_top = st.segmented_control(
        "Language",
        ["EN", "DE"],
        default="EN" if st.session_state.lang == "en" else "DE",
        key="lang_seg_top",
        label_visibility="collapsed",
    )
    if _lang_top and _lang_top.lower() != st.session_state.lang:
        st.session_state.lang = _lang_top.lower()
        st.rerun()

# ── Router ─────────────────────────────────────────────────────────────────────
if st.session_state.page == "command_center":
    _cc_col, _cc_chat = st.columns([7, 3], gap="large")
    with _cc_col:
        render_command_center(decisions, price_data)
    with _cc_chat:
        render_chat_panel(ticker=None)

elif st.session_state.page == "deep_dive":
    _dd_col, _dd_chat = st.columns([7, 3], gap="large")
    with _dd_col:
        render_deep_dive(st.session_state.selected, decisions, price_data, news_df)
    with _dd_chat:
        render_chat_panel(ticker=st.session_state.selected)
        _ticker = st.session_state.selected
        if _ticker:
            from data.loader import load_detection
            _det = load_detection(_ticker)
            _dec_df  = decisions[decisions["ticker"] == _ticker]
            _dec_row = _dec_df.iloc[-1] if not _dec_df.empty else __import__("pandas").Series(dtype=object)
            if _det is not None and not _det.empty:
                render_metric_cards(_det.iloc[-1], _dec_row)

elif st.session_state.page == "portfolio":
    render_portfolio_page()
