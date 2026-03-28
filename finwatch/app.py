"""
FinWatch AI — Main Entry Point
================================
Run:  streamlit run finwatch/app.py

This file wires all sub-modules together.
dashboard.py is left untouched; this is the modular equivalent.
"""

import sys
from pathlib import Path

# Allow imports like `from data.loader import ...` when running from project root
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import pandas as pd
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

st.set_page_config(page_title="FinWatch AI", page_icon="📡", layout="wide")

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

/* ── Base ── */
html,body,[class*="css"]{
    font-family:'Inter',sans-serif;
    background:#060a0f;
    color:#e2e8f0;
}
/* Dot-grid background for depth */
[data-testid="stAppViewContainer"]{
    background:
        radial-gradient(ellipse 80% 50% at 10% 0%, rgba(29,233,182,0.04) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 90% 100%, rgba(88,166,255,0.03) 0%, transparent 60%),
        radial-gradient(rgba(26,35,50,0.55) 1px, transparent 1px)
        #060a0f;
    background-size:auto, auto, 26px 26px;
}

/* ── Sidebar ── */
[data-testid="stSidebar"]{
    background:linear-gradient(180deg,#080d15 0%,#060a0f 100%)!important;
    border-right:1px solid rgba(30,45,65,0.8)!important;
    width:220px!important;min-width:220px!important;
}
[data-testid="stSidebar"] > div:first-child{width:220px!important;min-width:220px!important}
[data-testid="stSidebar"] .block-container{padding-top:0.6rem!important}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"]{gap:0px!important}
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"]{gap:2px!important;margin-bottom:0!important}
[data-testid="stSidebar"] .stButton{margin:0!important;padding:0!important}

[data-testid="stSidebar"] button{
    background:transparent!important;
    border:none!important;
    color:#637a91!important;
    font-family:'IBM Plex Mono',monospace!important;
    font-size:11px!important;
    text-align:left!important;
    padding:2px 10px!important;
    border-radius:0!important;
    border-left:2px solid transparent!important;
    width:100%!important;
    min-height:24px!important;
    line-height:1.4!important;
    transition:all 0.15s ease!important;
}
[data-testid="stSidebar"] button:hover{
    background:rgba(29,233,182,0.05)!important;
    color:#cdd9e5!important;
    border-left:2px solid rgba(29,233,182,0.3)!important;
}

/* ── Main layout ── */
.main .block-container{padding:0 1.2rem 2rem!important;padding-top:0!important;max-width:100%!important}
#MainMenu,footer,header{display:none!important}
[data-testid="stToolbar"]{display:none!important}
[data-testid="stHeader"]{display:none!important}
[data-testid="stDecoration"]{display:none!important}
section.main > div{padding-top:0!important}

/* ── Logo ── */
.logo{
    font-family:'IBM Plex Mono',monospace;
    font-size:15px;font-weight:500;
    color:#e2e8f0;
    padding:14px 0 10px;
    letter-spacing:0.5px;
}
.logo span{
    color:#1de9b6;
    text-shadow:0 0 20px rgba(29,233,182,0.5);
}
.sb-label{
    font-size:8px;letter-spacing:2.5px;color:#3d5266;
    text-transform:uppercase;padding:8px 0 4px;
    font-family:'IBM Plex Mono',monospace;
}

/* ── Stock Header ── */
.stock-header{
    padding:12px 0 10px;
    display:flex;align-items:center;gap:12px;
    border-bottom:1px solid rgba(30,45,65,0.8);
    margin-bottom:2px;
}
[data-testid="stVerticalBlock"]>[data-testid="element-container"]{margin-bottom:0!important}
div[data-testid="element-container"]>.stMarkdown{margin-bottom:0!important}
.sh-name{font-size:17px;font-weight:600;color:#e2e8f0;letter-spacing:-0.3px}
.sh-sub{font-size:10px;color:#3d5266;font-family:'IBM Plex Mono',monospace;letter-spacing:0.5px}
.sh-price{font-size:22px;font-weight:500;color:#e2e8f0;margin-left:auto;font-family:'IBM Plex Mono',monospace}
.sh-up{font-size:12px;color:#1de9b6;font-family:'IBM Plex Mono',monospace;text-shadow:0 0 12px rgba(29,233,182,0.4)}
.sh-dn{font-size:12px;color:#f85149;font-family:'IBM Plex Mono',monospace;text-shadow:0 0 12px rgba(248,81,73,0.4)}

/* ── Watchlist changes ── */
.wl-chg-up{color:#1de9b6;font-family:'IBM Plex Mono',monospace;font-size:10px;text-align:right;line-height:1.8;padding-top:3px}
.wl-chg-dn{color:#f85149;font-family:'IBM Plex Mono',monospace;font-size:10px;text-align:right;line-height:1.8;padding-top:3px}
.wl-chg-fl{color:#3d5266;font-family:'IBM Plex Mono',monospace;font-size:10px;text-align:right;line-height:1.8;padding-top:3px}

/* ── Chart label ── */
.chart-label{
    font-size:9px;letter-spacing:2px;color:#3d5266;
    text-transform:uppercase;margin-bottom:6px;
    font-family:'IBM Plex Mono',monospace;
}

/* ── Cards / Panels ── */
.rp-section{
    padding:12px 0;
    border-bottom:1px solid rgba(30,45,65,0.6);
}
.rp-title{
    font-size:8px;letter-spacing:2.5px;color:#3d5266;
    text-transform:uppercase;margin-bottom:10px;
    font-family:'IBM Plex Mono',monospace;
}
.sev-big{font-size:17px;font-weight:600;margin-bottom:2px;font-family:'IBM Plex Mono',monospace;letter-spacing:-0.5px}

/* Severity colors + glow */
.s-critical{color:#f85149;text-shadow:0 0 16px rgba(248,81,73,0.35)}
.s-warning{color:#e3b341;text-shadow:0 0 14px rgba(227,179,65,0.3)}
.s-watch{color:#58a6ff;text-shadow:0 0 12px rgba(88,166,255,0.25)}
.s-normal{color:#2ea043}.s-positive{color:#2ea043}.s-review{color:#a371f7}

.rp-row{display:flex;justify-content:space-between;align-items:center;padding:3px 0}
.rp-key{font-size:10px;color:#3d5266;font-family:'IBM Plex Mono',monospace}
.rp-val{font-size:10px;color:#c9d1d9;font-family:'IBM Plex Mono',monospace}
.rp-up{color:#1de9b6!important}.rp-dn{color:#f85149!important}
.dot-row{display:flex;gap:4px;align-items:center}
.d{width:7px;height:7px;border-radius:50%;display:inline-block}
.d-on{background:#2ea043;box-shadow:0 0 6px rgba(46,160,67,0.6)}
.d-off{background:#1a2332}

/* ── News ── */
.news-item{display:block;width:100%;padding:7px 0;border-bottom:1px solid rgba(30,45,65,0.5)}
.news-item:last-child{border-bottom:none}
.news-sent{font-size:8px;padding:1px 6px;border-radius:3px;display:inline-block;margin-bottom:4px;font-family:'IBM Plex Mono',monospace;letter-spacing:0.5px}
.s-neg{background:rgba(248,81,73,0.12);color:#f85149;border:1px solid rgba(248,81,73,0.2)}
.s-pos{background:rgba(46,160,67,0.12);color:#2ea043;border:1px solid rgba(46,160,67,0.2)}
.s-neu{background:rgba(30,45,65,0.6);color:#637a91;border:1px solid rgba(30,45,65,0.8)}
.news-text{font-size:10px;color:#637a91;line-height:1.5}
.summary-text{font-size:10px;color:#637a91;line-height:1.7}

/* ── Analysis Panel ── */
.analysis-panel{
    background:rgba(11,17,26,0.7);
    border:1px solid rgba(30,45,65,0.6);
    border-radius:10px;
    padding:14px 18px;
    margin-top:10px;
    box-shadow:0 4px 20px rgba(0,0,0,0.35),inset 0 1px 0 rgba(255,255,255,0.03);
    backdrop-filter:blur(6px);
}
.an-header{
    font-size:8px;letter-spacing:2.5px;text-transform:uppercase;
    color:#3d5266;border-bottom:1px solid rgba(30,45,65,0.6);
    padding-bottom:8px;margin-bottom:12px;font-family:'IBM Plex Mono',monospace;
}
.an-date{color:#3d5266;margin-left:10px;font-size:8px}
.an-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}
.an-title{font-size:7px;letter-spacing:2px;text-transform:uppercase;color:#3d5266;margin-bottom:5px;border-bottom:1px solid rgba(30,45,65,0.4);padding-bottom:3px;font-family:'IBM Plex Mono',monospace}
.an-row{display:flex;align-items:flex-start;gap:5px;padding:2px 0}
.an-pos{color:#1de9b6;font-size:9px;flex-shrink:0;line-height:15px;font-family:'IBM Plex Mono',monospace}
.an-risk{color:#f85149;font-size:9px;flex-shrink:0;line-height:15px;font-family:'IBM Plex Mono',monospace}
.an-warn{color:#e3b341;font-size:9px;flex-shrink:0;line-height:15px;font-family:'IBM Plex Mono',monospace}
.an-neu{color:#3d5266;font-size:9px;flex-shrink:0;line-height:15px;font-family:'IBM Plex Mono',monospace}
.an-text{font-size:10px;color:#8b9aab;line-height:1.5;font-family:'IBM Plex Mono',monospace}

/* ── Tooltip ── */
[data-tip]{position:relative;cursor:help;border-bottom:1px dotted rgba(99,122,145,0.4)}
[data-tip]::after{
    content:attr(data-tip);position:absolute;left:0;top:120%;
    background:#0f1923;color:#c9d1d9;font-size:10px;line-height:1.6;
    padding:7px 12px;border-radius:6px;white-space:normal;
    max-width:240px;min-width:140px;
    border:1px solid rgba(30,45,65,0.8);
    box-shadow:0 8px 24px rgba(0,0,0,0.5);
    z-index:9999;visibility:hidden;opacity:0;
    transition:opacity 0.15s ease;pointer-events:none;
    font-family:'Inter',sans-serif;font-weight:400;
}
[data-tip]:hover::after{visibility:visible;opacity:1}

/* ── Segmented control ── */
[data-testid="stSegmentedControl"] > div{
    background:rgba(11,17,26,0.8)!important;
    border:1px solid rgba(30,45,65,0.7)!important;
    border-radius:6px!important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar{width:4px;height:4px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:rgba(30,45,65,0.8);border-radius:2px}
::-webkit-scrollbar-thumb:hover{background:rgba(48,73,104,0.9)}

/* ── Escalate alert buttons ── */
.escalate-btn button{
    color:#f85149!important;
    border-left:2px solid rgba(248,81,73,0.5)!important;
    background:rgba(248,81,73,0.05)!important;
}
.escalate-btn button:hover{
    background:rgba(248,81,73,0.12)!important;
    border-left:2px solid rgba(248,81,73,0.8)!important;
    color:#ff6b6b!important;
}
</style>
""", unsafe_allow_html=True)

# ── Imports (after sys.path is set) ───────────────────────────────────────────
from data.loader import (
    COMPANY_NAMES, SECTORS, ETF_STOCKS, SEV_CLS,
    load_decisions, load_detection, load_news, load_spx, load_price_summary,
)
from ui.components import (
    render_sidebar, render_stock_header, render_spx_header,
    render_risk_news_row, render_anomaly_selector,
    render_analysis_panel, render_investor_summary, render_llm_report,
    render_strategy_box, show_analysis_modal, render_candle_panel,
)
from ui.charts import render_price_chart, render_rsi_chart, render_spx_chart

# ── Session State ─────────────────────────────────────────────────────────────
decisions   = load_decisions()
news_df     = load_news()
price_data  = load_price_summary()
all_tickers = decisions["ticker"].tolist()


def _find_sector(t):
    for s, ts in SECTORS.items():
        if t in ts: return s
    return list(SECTORS.keys())[0]


if "selected"        not in st.session_state: st.session_state.selected        = all_tickers[0]
if "period"          not in st.session_state: st.session_state.period          = "1M"
if "spx_period"      not in st.session_state: st.session_state.spx_period      = "1M"
if "category"        not in st.session_state: st.session_state.category        = "Stocks"
if "anomaly_date"    not in st.session_state: st.session_state.anomaly_date    = None
if "language"        not in st.session_state: st.session_state.language        = "english"
if "llm_cache"       not in st.session_state: st.session_state.llm_cache       = {}
if "lang_modal"      not in st.session_state: st.session_state.lang_modal      = None
if "clicked_date"    not in st.session_state: st.session_state.clicked_date    = None
if "selected_sector" not in st.session_state: st.session_state.selected_sector = _find_sector(st.session_state.selected)
if "page"            not in st.session_state: st.session_state.page            = "landing"


# ── Landing Page ───────────────────────────────────────────────────────────────
def render_landing():
    decisions_l  = load_decisions()
    total        = len(decisions_l)
    critical     = (decisions_l["severity"] == "CRITICAL").sum()
    warning      = (decisions_l["severity"] == "WARNING").sum()

    st.markdown("""
    <style>
    /* Hide sidebar on landing */
    [data-testid="stSidebar"]{display:none!important}
    .main .block-container{padding:0 2rem 2rem!important;padding-top:0!important}

    /* Landing background — layered radial glows */
    [data-testid="stAppViewContainer"]{
        background:
            radial-gradient(ellipse 70% 55% at 15% 10%,  rgba(29,233,182,0.08) 0%, transparent 55%),
            radial-gradient(ellipse 50% 40% at 85% 80%,  rgba(88,166,255,0.07) 0%, transparent 55%),
            radial-gradient(ellipse 40% 35% at 80% 10%,  rgba(163,113,247,0.05) 0%, transparent 50%),
            radial-gradient(rgba(22,33,48,0.5) 1px, transparent 1px)
            #060a0f;
        background-size:auto,auto,auto,24px 24px;
    }

    @keyframes float {
        0%,100%{transform:translateY(0px)} 50%{transform:translateY(-6px)}
    }
    @keyframes glow-pulse {
        0%,100%{opacity:0.5} 50%{opacity:1}
    }

    .hero-wrap{
        display:flex;flex-direction:column;align-items:center;
        justify-content:center;text-align:center;
        padding:10vh 0 5vh;
    }
    .hero-badge{
        display:inline-block;
        font-family:'IBM Plex Mono',monospace;font-size:10px;
        letter-spacing:3px;text-transform:uppercase;color:#1de9b6;
        border:1px solid rgba(29,233,182,0.3);border-radius:20px;
        padding:4px 16px;margin-bottom:28px;
        background:rgba(29,233,182,0.05);
    }
    .hero-title{
        font-family:'Inter',sans-serif;
        font-size:68px;font-weight:700;letter-spacing:-3px;line-height:1;
        background:linear-gradient(135deg,#e2e8f0 0%,#1de9b6 40%,#58a6ff 75%,#a371f7 100%);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;
        background-clip:text;margin-bottom:20px;
    }
    .hero-sub{
        font-family:'Inter',sans-serif;font-size:15px;font-weight:300;
        color:#637a91;max-width:560px;line-height:1.8;margin-bottom:14px;
    }
    .hero-stats{
        display:flex;gap:32px;justify-content:center;margin-top:16px;
        font-family:'IBM Plex Mono',monospace;
    }
    .hs-item{text-align:center}
    .hs-val{font-size:22px;font-weight:500;color:#e2e8f0;display:block}
    .hs-val.red{color:#f85149;text-shadow:0 0 16px rgba(248,81,73,0.4)}
    .hs-val.yellow{color:#e3b341;text-shadow:0 0 14px rgba(227,179,65,0.3)}
    .hs-val.teal{color:#1de9b6;text-shadow:0 0 14px rgba(29,233,182,0.4)}
    .hs-label{font-size:9px;letter-spacing:2px;text-transform:uppercase;color:#3d5266;margin-top:2px}

    .divider{
        width:120px;height:1px;
        background:linear-gradient(90deg,transparent,rgba(29,233,182,0.4),transparent);
        margin:40px auto;
    }

    .cards-title{
        text-align:center;font-family:'IBM Plex Mono',monospace;
        font-size:9px;letter-spacing:3px;text-transform:uppercase;
        color:#3d5266;margin-bottom:24px;
    }

    .lcard{
        background:rgba(11,17,26,0.75);
        border:1px solid rgba(30,45,65,0.6);
        border-radius:16px;
        padding:32px 24px 24px;
        text-align:center;
        backdrop-filter:blur(12px);
        box-shadow:0 8px 32px rgba(0,0,0,0.4),inset 0 1px 0 rgba(255,255,255,0.03);
        transition:all 0.25s ease;
        margin-bottom:8px;
        height:100%;
    }
    .lcard:hover{
        border-color:rgba(29,233,182,0.35);
        box-shadow:0 12px 40px rgba(0,0,0,0.5),0 0 30px rgba(29,233,182,0.08),inset 0 1px 0 rgba(255,255,255,0.04);
        transform:translateY(-2px);
    }
    .lcard-icon{font-size:32px;margin-bottom:14px;display:block;animation:float 4s ease-in-out infinite}
    .lcard-title{font-family:'Inter',sans-serif;font-size:17px;font-weight:600;color:#e2e8f0;margin-bottom:10px;letter-spacing:-0.3px}
    .lcard-desc{font-family:'Inter',sans-serif;font-size:12px;color:#637a91;line-height:1.7;margin-bottom:0}
    .lcard-tag{
        display:inline-block;margin-bottom:16px;
        font-family:'IBM Plex Mono',monospace;font-size:8px;
        letter-spacing:2px;text-transform:uppercase;
        padding:2px 10px;border-radius:10px;
    }
    .tag-live{background:rgba(29,233,182,0.1);color:#1de9b6;border:1px solid rgba(29,233,182,0.2)}
    .tag-soon{background:rgba(30,45,65,0.5);color:#3d5266;border:1px solid rgba(30,45,65,0.6)}

    /* Landing buttons */
    div[data-testid="stButton"] > button{
        background:rgba(29,233,182,0.08)!important;
        border:1px solid rgba(29,233,182,0.25)!important;
        border-radius:8px!important;
        color:#1de9b6!important;
        font-family:'IBM Plex Mono',monospace!important;
        font-size:11px!important;
        letter-spacing:1.5px!important;
        text-transform:uppercase!important;
        padding:10px!important;
        margin-top:16px!important;
        transition:all 0.2s ease!important;
    }
    div[data-testid="stButton"] > button:hover{
        background:rgba(29,233,182,0.15)!important;
        border-color:rgba(29,233,182,0.5)!important;
        box-shadow:0 0 20px rgba(29,233,182,0.15)!important;
    }
    div[data-testid="stButton"] > button:disabled,
    div[data-testid="stButton"] > button[disabled]{
        background:rgba(30,45,65,0.3)!important;
        border-color:rgba(30,45,65,0.4)!important;
        color:#3d5266!important;cursor:not-allowed!important;
    }

    .footer-note{
        text-align:center;margin-top:40px;
        font-family:'IBM Plex Mono',monospace;font-size:9px;
        letter-spacing:1.5px;color:#1e2d3d;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Hero ──
    st.markdown(f"""
    <div class="hero-wrap">
        <div class="hero-badge">AI · FINANCE · MONITORING</div>
        <div class="hero-title">FinWatch AI</div>
        <div class="hero-sub">
            An end-to-end AI monitoring system for equity markets —<br>
            detecting anomalies, predicting risk, and generating analyst reports automatically.
        </div>
        <div class="hero-stats">
            <div class="hs-item">
                <span class="hs-val teal">{total}</span>
                <div class="hs-label">Stocks Monitored</div>
            </div>
            <div class="hs-item">
                <span class="hs-val red">{int(critical)}</span>
                <div class="hs-label">Critical Today</div>
            </div>
            <div class="hs-item">
                <span class="hs-val yellow">{int(warning)}</span>
                <div class="hs-label">Warnings Today</div>
            </div>
        </div>
    </div>
    <div class="divider"></div>
    <div class="cards-title">Select a view to continue</div>
    """, unsafe_allow_html=True)

    # ── Cards ──
    c1, c2, c3 = st.columns(3, gap="large")

    with c1:
        st.markdown("""
        <div class="lcard">
            <span class="lcard-icon">📈</span>
            <div class="lcard-tag tag-live">Live</div>
            <div class="lcard-title">Equities</div>
            <div class="lcard-desc">
                Monitor 45 stocks across 9 sectors.<br>
                Anomaly detection, risk scoring,<br>
                AI-generated analyst reports.
            </div>
        </div>""", unsafe_allow_html=True)
        if st.button("Open Equities →", key="go_stocks", use_container_width=True):
            st.session_state.page = "stocks"
            st.rerun()

    with c2:
        st.markdown("""
        <div class="lcard">
            <span class="lcard-icon">🌐</span>
            <div class="lcard-tag tag-live">Live</div>
            <div class="lcard-title">Market Index</div>
            <div class="lcard-desc">
                S&P 500 overview and sector<br>
                alert summary. Understand<br>
                the macro context at a glance.
            </div>
        </div>""", unsafe_allow_html=True)
        if st.button("Open Index →", key="go_index", use_container_width=True):
            st.session_state.page     = "stocks"
            st.session_state.selected = "^SPX"
            st.rerun()

    with c3:
        st.markdown("""
        <div class="lcard">
            <span class="lcard-icon">🗂️</span>
            <div class="lcard-tag tag-live">Live</div>
            <div class="lcard-title">Sector ETFs</div>
            <div class="lcard-desc">
                Risk overview across 9 sector ETFs.<br>
                Top performers, worst drawdowns,<br>
                and severity distribution per sector.
            </div>
        </div>""", unsafe_allow_html=True)
        if st.button("Open ETFs →", key="go_etfs", use_container_width=True):
            st.session_state.page = "etfs"
            st.rerun()

    st.markdown('<div class="footer-note">FINWATCH AI · AI ENGINEERING PROJECT · 2026</div>', unsafe_allow_html=True)


if st.session_state.page == "landing":
    render_landing()
    st.stop()


# ── ETF Page ──────────────────────────────────────────────────────────────────
def render_etf_page():
    SEV_ORDER  = {"CRITICAL":0,"WARNING":1,"WATCH":2,"REVIEW":3,"POSITIVE_MOMENTUM":4,"NORMAL":5}
    SEV_COLOR  = {"CRITICAL":"#f85149","WARNING":"#e3b341","WATCH":"#58a6ff","NORMAL":"#3fb950","POSITIVE_MOMENTUM":"#3fb950","REVIEW":"#a371f7"}

    with st.sidebar:
        st.markdown('<div class="logo">Fin<span>Watch</span> AI</div>', unsafe_allow_html=True)
        if st.button("← Home", key="etf_home", use_container_width=True):
            st.session_state.page = "landing"
            st.rerun()

    st.markdown("""
    <div style="padding:14px 0 10px;border-bottom:1px solid rgba(30,45,65,0.8);margin-bottom:16px;display:flex;align-items:center;gap:14px">
        <div>
            <div style="font-size:20px;font-weight:700;color:#e2e8f0;letter-spacing:-0.5px">Sector ETF Overview</div>
            <div style="font-size:10px;color:#3d5266;font-family:'IBM Plex Mono',monospace;margin-top:2px">
                9 SECTORS · RISK AGGREGATION · AI MONITORING
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    for etf_name, tickers in ETF_STOCKS.items():
        sec_dec = decisions[decisions["ticker"].isin(tickers)]
        if sec_dec.empty:
            continue

        # Aggregate stats
        worst_sev = (sec_dec.assign(_o=sec_dec["severity"].map(SEV_ORDER).fillna(99))
                     .sort_values("_o")["severity"].iloc[0])
        sev_counts = sec_dec["severity"].value_counts()
        critical   = int(sev_counts.get("CRITICAL", 0))
        warning    = int(sev_counts.get("WARNING", 0))
        watch      = int(sev_counts.get("WATCH", 0))
        normal     = int(sev_counts.get("NORMAL", 0) + sev_counts.get("POSITIVE_MOMENTUM", 0))

        # Price changes from price_data
        changes = []
        for t in tickers:
            if t in price_data:
                _, pct = price_data[t]
                changes.append((t, pct))
        changes.sort(key=lambda x: x[1])
        worst_t = changes[0]  if changes else None
        best_t  = changes[-1] if changes else None

        sev_color = SEV_COLOR.get(worst_sev, "#637a91")
        sev_css   = SEV_CLS.get(worst_sev, "")

        # Severity bar (proportional)
        n = len(tickers)
        bar_html = ""
        for sev, cnt, col in [
            ("CRITICAL", critical, "#f85149"),
            ("WARNING",  warning,  "#e3b341"),
            ("WATCH",    watch,    "#58a6ff"),
            ("NORMAL",   normal,   "#2ea043"),
        ]:
            if cnt > 0:
                w = round(cnt / n * 100)
                bar_html += f'<div style="width:{w}%;background:{col};height:4px;border-radius:2px;display:inline-block" title="{cnt} {sev}"></div>'

        worst_html = ""
        best_html  = ""
        if worst_t:
            arrow = "▼" if worst_t[1] < 0 else "▲"
            worst_html = f'<span style="color:#f85149;font-family:IBM Plex Mono;font-size:10px">{worst_t[0]} {arrow}{abs(worst_t[1]):.1f}%</span>'
        if best_t and best_t != worst_t:
            arrow = "▲" if best_t[1] >= 0 else "▼"
            best_html = f'<span style="color:#1de9b6;font-family:IBM Plex Mono;font-size:10px">{best_t[0]} {arrow}{abs(best_t[1]):.1f}%</span>'

        st.markdown(f"""
        <div style="background:rgba(11,17,26,0.6);border:1px solid rgba(30,45,65,0.5);
                    border-radius:10px;padding:14px 18px;margin-bottom:10px;
                    box-shadow:0 4px 16px rgba(0,0,0,0.3)">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
                <div>
                    <span style="font-size:13px;font-weight:600;color:#e2e8f0;letter-spacing:-0.2px">{etf_name}</span>
                </div>
                <div style="display:flex;align-items:center;gap:20px">
                    <div style="text-align:right">
                        <div style="font-size:8px;letter-spacing:2px;text-transform:uppercase;color:#3d5266;font-family:IBM Plex Mono;margin-bottom:2px">Best 1D</div>
                        {best_html}
                    </div>
                    <div style="text-align:right">
                        <div style="font-size:8px;letter-spacing:2px;text-transform:uppercase;color:#3d5266;font-family:IBM Plex Mono;margin-bottom:2px">Worst 1D</div>
                        {worst_html}
                    </div>
                    <div style="text-align:right;min-width:80px">
                        <div style="font-size:8px;letter-spacing:2px;text-transform:uppercase;color:#3d5266;font-family:IBM Plex Mono;margin-bottom:2px">Sector Risk</div>
                        <span style="font-size:12px;font-weight:600;color:{sev_color};font-family:IBM Plex Mono">{worst_sev}</span>
                    </div>
                </div>
            </div>
            <div style="display:flex;gap:3px;width:100%;margin-bottom:8px">{bar_html}</div>
            <div style="display:flex;gap:16px">
                {'<span style="font-size:9px;color:#f85149;font-family:IBM Plex Mono">'+str(critical)+'C</span>' if critical else ''}
                {'<span style="font-size:9px;color:#e3b341;font-family:IBM Plex Mono">'+str(warning)+'W</span>' if warning else ''}
                {'<span style="font-size:9px;color:#58a6ff;font-family:IBM Plex Mono">'+str(watch)+'Wt</span>' if watch else ''}
                {'<span style="font-size:9px;color:#2ea043;font-family:IBM Plex Mono">'+str(normal)+'N</span>' if normal else ''}
                <span style="font-size:9px;color:#3d5266;font-family:IBM Plex Mono;margin-left:auto">{" · ".join(tickers)}</span>
            </div>
        </div>""", unsafe_allow_html=True)


if st.session_state.page == "etfs":
    render_etf_page()
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────
render_sidebar(decisions, price_data)

# ── Main ──────────────────────────────────────────────────────────────────────
ticker = st.session_state.selected

# ── Market Overview (^SPX) ────────────────────────────────────────────────────
if ticker == "^SPX":
    spx_df = load_spx()
    render_spx_header(spx_df)

    _sl, _sr = st.columns([6, 4], gap="small")
    with _sl:
        st.markdown('<div class="chart-label" style="padding-top:6px">S&P 500 Price History</div>', unsafe_allow_html=True)
    with _sr:
        _new_spx_period = st.segmented_control(
            "SPX Period", ["1M", "3M", "6M", "1Y", "All"],
            default=st.session_state.spx_period,
            label_visibility="collapsed",
            key="spx_period_ctrl",
        )
        if _new_spx_period and _new_spx_period != st.session_state.spx_period:
            st.session_state.spx_period = _new_spx_period
            st.rerun()
    render_spx_chart(spx_df, st.session_state.spx_period)

    # Sector Alert Overview
    st.markdown('<div class="chart-label">Sector Alert Overview</div>', unsafe_allow_html=True)
    sev_order = {"CRITICAL": 0, "WARNING": 1, "WATCH": 2, "REVIEW": 3, "POSITIVE_MOMENTUM": 4, "NORMAL": 5}
    sector_rows_html = ""
    for sector, etf_tickers in ETF_STOCKS.items():
        sector_dec = decisions[decisions["ticker"].isin(etf_tickers)]
        if sector_dec.empty:
            continue
        critical = (sector_dec["severity"] == "CRITICAL").sum()
        warning  = (sector_dec["severity"] == "WARNING").sum()
        watch    = (sector_dec["severity"] == "WATCH").sum()
        worst    = (sector_dec.assign(_o=sector_dec["severity"].map(sev_order).fillna(99))
                               .sort_values("_o").iloc[0]["severity"])
        sev_c    = SEV_CLS.get(worst, "s-normal")
        sector_rows_html += f"""
        <div class="rp-row">
          <span class="rp-key" style="width:180px">{sector}</span>
          <span class="rp-val {sev_c}" style="min-width:60px">{worst.replace("_"," ")}</span>
          <span style="font-size:9px;color:#f85149;font-family:IBM Plex Mono;min-width:30px">{f'{critical}C' if critical else ''}</span>
          <span style="font-size:9px;color:#e3b341;font-family:IBM Plex Mono;min-width:30px">{f'{warning}W' if warning else ''}</span>
          <span style="font-size:9px;color:#58a6ff;font-family:IBM Plex Mono">{f'{watch}Wt' if watch else ''}</span>
        </div>"""
    st.markdown(f'<div class="rp-section">{sector_rows_html}</div>', unsafe_allow_html=True)

    # Market Pulse + Top Critical
    _spx_c1, _spx_c2 = st.columns(2, gap="small")
    with _spx_c1:
        total    = len(decisions)
        by_sev   = decisions["severity"].value_counts()
        critical = by_sev.get("CRITICAL", 0)
        warning  = by_sev.get("WARNING", 0)
        watch    = by_sev.get("WATCH", 0)
        normal   = by_sev.get("NORMAL", 0) + by_sev.get("POSITIVE_MOMENTUM", 0)
        st.markdown(f"""
        <div class="rp-section">
          <div class="rp-title">Market Pulse</div>
          <div class="rp-row"><span class="rp-key">Total Stocks</span><span class="rp-val">{total}</span></div>
          <div class="rp-row"><span class="rp-key s-critical">Critical</span><span class="rp-val s-critical">{critical}</span></div>
          <div class="rp-row"><span class="rp-key s-warning">Warning</span><span class="rp-val s-warning">{warning}</span></div>
          <div class="rp-row"><span class="rp-key s-watch">Watch</span><span class="rp-val s-watch">{watch}</span></div>
          <div class="rp-row"><span class="rp-key s-normal">Normal</span><span class="rp-val s-normal">{normal}</span></div>
        </div>""", unsafe_allow_html=True)
    with _spx_c2:
        top_critical = decisions[decisions["severity"] == "CRITICAL"].head(8)
        if not top_critical.empty:
            crit_html = ""
            for _, r in top_critical.iterrows():
                t = r["ticker"]
                n = COMPANY_NAMES.get(t, t)
                crit_html += f'<div class="rp-row"><span class="rp-key">{n}</span><span class="rp-val s-critical">{t}</span></div>'
            st.markdown(f'<div class="rp-section"><div class="rp-title">Top Critical</div>{crit_html}</div>', unsafe_allow_html=True)

    st.stop()

# ── Normal Stock View ─────────────────────────────────────────────────────────
dec_row = decisions[decisions["ticker"] == ticker]
if dec_row.empty:
    st.warning(f"No decision data for {ticker}")
    st.stop()
row = dec_row.iloc[0]

det_df = load_detection(ticker)
name   = COMPANY_NAMES.get(ticker, ticker)
lang   = st.session_state.language

render_stock_header(ticker, name, det_df)

if det_df is not None and "Close" in det_df.columns and len(det_df) > 1:
    render_risk_news_row(ticker, row, det_df, decisions, news_df)

    _pl, _pr = st.columns([6, 4], gap="small")
    with _pl:
        st.markdown('<div class="chart-label" style="padding-top:6px">Price  ·  EMA 20</div>', unsafe_allow_html=True)
    with _pr:
        _new_period = st.segmented_control(
            "Period", ["1M", "3M", "6M", "1Y", "All"],
            default=st.session_state.period,
            label_visibility="collapsed",
            key="period_ctrl",
        )
        if _new_period and _new_period != st.session_state.period:
            st.session_state.period = _new_period
            st.rerun()
    _ev = render_price_chart(
        det_df, ticker, st.session_state.period, st.session_state.clicked_date
    )
    try:
        _pts = _ev.selection.points if _ev else []
        if _pts:
            _cd = str(_pts[0].get("x", ""))[:10]
            if _cd:
                st.session_state.clicked_date = _cd
    except (AttributeError, TypeError, KeyError):
        pass

    render_candle_panel(det_df, st.session_state.clicked_date)
    render_anomaly_selector(det_df, name, st.session_state.period)

    st.markdown('<div class="chart-label">RSI  ·  RSI MA 14</div>', unsafe_allow_html=True)
    render_rsi_chart(det_df, st.session_state.period)

    render_analysis_panel(det_df, dec_row, ticker, name, lang)
    render_investor_summary(det_df, dec_row, row, news_df, ticker)
    render_strategy_box(det_df, dec_row)
    render_llm_report(ticker, news_df, lang, dec_row)

    if st.session_state.lang_modal:
        show_analysis_modal(ticker, news_df, st.session_state.lang_modal)
