"""
FinWatch AI — TradingView-Style Dashboard
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import ast
import os
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")

st.set_page_config(page_title="FinWatch AI", page_icon="📡", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500&display=swap');
html,body,[class*="css"]{font-family:'IBM Plex Sans',sans-serif;background:#080c10;color:#cdd9e5}
[data-testid="stSidebar"]{background:#0d1117!important;border-right:1px solid #1a2332!important;width:210px!important;min-width:210px!important}
[data-testid="stSidebar"] > div:first-child{width:210px!important;min-width:210px!important}
[data-testid="stSidebar"] .block-container{padding-top:0.5rem!important}
.main .block-container{padding:0 1rem 1rem!important;padding-top:0!important;max-width:100%!important}
#MainMenu,footer,header{visibility:hidden}
[data-testid="stToolbar"]{display:none}

/* Sidebar buttons */
[data-testid="stSidebar"] button{
    background:#0d1117!important;border:none!important;
    color:#8b949e!important;font-family:'IBM Plex Mono',monospace!important;
    font-size:11px!important;text-align:left!important;padding:5px 10px!important;
    border-radius:0!important;border-left:2px solid transparent!important;
    width:100%!important;
}
[data-testid="stSidebar"] button:hover{background:#161b22!important;color:#cdd9e5!important}

.logo{font-family:'IBM Plex Mono',monospace;font-size:14px;font-weight:500;color:#cdd9e5;padding:10px 0 8px}
.logo span{color:#1de9b6}
.sb-label{font-size:8px;letter-spacing:2px;color:#5c7080;text-transform:uppercase;padding:6px 0 4px;font-family:'IBM Plex Mono',monospace}

.stock-header{padding:10px 0;display:flex;align-items:center;gap:10px;border-bottom:1px solid #1a2332;margin-bottom:8px}
.sh-name{font-size:16px;font-weight:500;color:#cdd9e5}
.sh-sub{font-size:10px;color:#5c7080;font-family:'IBM Plex Mono',monospace}
.sh-price{font-size:20px;font-weight:500;color:#cdd9e5;margin-left:auto;font-family:'IBM Plex Mono',monospace}
.sh-up{font-size:12px;color:#1de9b6;font-family:'IBM Plex Mono',monospace}
.sh-dn{font-size:12px;color:#f85149;font-family:'IBM Plex Mono',monospace}

/* Sidebar watchlist price column */
.wl-chg-up{color:#1de9b6;font-family:'IBM Plex Mono',monospace;font-size:10px;text-align:right;line-height:1.6;padding-top:3px}
.wl-chg-dn{color:#f85149;font-family:'IBM Plex Mono',monospace;font-size:10px;text-align:right;line-height:1.6;padding-top:3px}
.wl-chg-fl{color:#5c7080;font-family:'IBM Plex Mono',monospace;font-size:10px;text-align:right;line-height:1.6;padding-top:3px}

.chart-label{font-size:8px;letter-spacing:1.5px;color:#5c7080;text-transform:uppercase;margin-bottom:4px;font-family:'IBM Plex Mono',monospace}

.rp-section{padding:10px 0;border-bottom:1px solid #1a2332}
.rp-title{font-size:8px;letter-spacing:2px;color:#5c7080;text-transform:uppercase;margin-bottom:8px;font-family:'IBM Plex Mono',monospace}
.sev-big{font-size:16px;font-weight:500;margin-bottom:2px;font-family:'IBM Plex Mono',monospace}
.s-critical{color:#f85149}.s-warning{color:#e3b341}.s-watch{color:#58a6ff}
.s-normal{color:#3fb950}.s-positive{color:#3fb950}.s-review{color:#a371f7}
.rp-row{display:flex;justify-content:space-between;align-items:center;padding:3px 0}
.rp-key{font-size:10px;color:#5c7080;font-family:'IBM Plex Mono',monospace}
.rp-val{font-size:10px;color:#cdd9e5;font-family:'IBM Plex Mono',monospace}
.rp-up{color:#1de9b6!important}.rp-dn{color:#f85149!important}
.dot-row{display:flex;gap:3px;align-items:center}
.d{width:7px;height:7px;border-radius:50%;display:inline-block}
.d-on{background:#3fb950}.d-off{background:#2d3748}

.news-item{padding:6px 0;border-bottom:1px solid #0a0f15}
.news-item:last-child{border-bottom:none}
.news-sent{font-size:8px;padding:1px 5px;border-radius:2px;display:inline-block;margin-bottom:3px;font-family:'IBM Plex Mono',monospace}
.s-neg{background:#2d0f0f;color:#f85149}
.s-pos{background:#0a1f0d;color:#3fb950}
.s-neu{background:#161b22;color:#8b949e}
.news-text{font-size:10px;color:#8b949e;line-height:1.4}
.summary-text{font-size:10px;color:#8b949e;line-height:1.6}

/* Analysis Panel */
.analysis-panel{background:#0d1117;border:1px solid #1a2332;border-radius:3px;padding:12px 16px;margin-top:10px}
.an-header{font-size:9px;letter-spacing:2px;text-transform:uppercase;color:#cdd9e5;border-bottom:1px solid #1a2332;padding-bottom:7px;margin-bottom:10px;font-family:'IBM Plex Mono',monospace}
.an-date{color:#5c7080;margin-left:10px;font-size:8px}
.an-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}
.an-title{font-size:7px;letter-spacing:2px;text-transform:uppercase;color:#5c7080;margin-bottom:5px;border-bottom:1px solid #161b22;padding-bottom:3px;font-family:'IBM Plex Mono',monospace}
.an-row{display:flex;align-items:flex-start;gap:5px;padding:2px 0}
.an-pos{color:#1de9b6;font-size:9px;flex-shrink:0;line-height:15px;font-family:'IBM Plex Mono',monospace}
.an-risk{color:#f85149;font-size:9px;flex-shrink:0;line-height:15px;font-family:'IBM Plex Mono',monospace}
.an-warn{color:#e3b341;font-size:9px;flex-shrink:0;line-height:15px;font-family:'IBM Plex Mono',monospace}
.an-neu{color:#5c7080;font-size:9px;flex-shrink:0;line-height:15px;font-family:'IBM Plex Mono',monospace}
.an-text{font-size:10px;color:#adb5bd;line-height:1.5;font-family:'IBM Plex Mono',monospace}

/* Hover tooltips */
[data-tip]{position:relative;cursor:help;border-bottom:1px dotted #314158}
[data-tip]::after{
    content:attr(data-tip);position:absolute;left:0;top:120%;
    background:#1a2332;color:#cdd9e5;font-size:10px;line-height:1.5;
    padding:6px 10px;border-radius:4px;white-space:normal;max-width:240px;min-width:140px;
    border:1px solid #314158;z-index:9999;
    visibility:hidden;opacity:0;transition:opacity 0.15s;pointer-events:none;
    font-family:'IBM Plex Sans',sans-serif;font-weight:300;
}
[data-tip]:hover::after{visibility:visible;opacity:1}
</style>
""", unsafe_allow_html=True)

# ── Universe ──────────────────────────────────────────────────────────────────
COMPANY_NAMES = {
    "AAPL":"Apple","MSFT":"Microsoft","NVDA":"Nvidia","AMD":"AMD","GOOG":"Alphabet",
    "PLTR":"Palantir","META":"Meta","CRM":"Salesforce","SNOW":"Snowflake","AI":"C3.ai",
    "TSLA":"Tesla","AMZN":"Amazon","NKE":"Nike","MCD":"McDonald's","SBUX":"Starbucks",
    "JPM":"JPMorgan","BAC":"Bank of America","GS":"Goldman Sachs","MS":"Morgan Stanley","BLK":"BlackRock",
    "JNJ":"J&J","PFE":"Pfizer","UNH":"UnitedHealth","ABBV":"AbbVie","MRK":"Merck",
    "PG":"P&G","KO":"Coca-Cola","COST":"Costco","WMT":"Walmart","CL":"Colgate",
    "XOM":"ExxonMobil","CVX":"Chevron","COP":"ConocoPhillips","SLB":"Schlumberger","EOG":"EOG Resources",
    "CAT":"Caterpillar","HON":"Honeywell","BA":"Boeing","GE":"GE Aerospace","RTX":"Raytheon",
    "BE":"Bloom Energy","ENPH":"Enphase","PLUG":"Plug Power","NEE":"NextEra","FSLR":"First Solar",
}

ETF_STOCKS = {
    "Technology (XLK)":             ["AAPL","MSFT","NVDA","AMD","GOOG"],
    "AI & Robotics (BOTZ)":         ["PLTR","META","CRM","SNOW","AI"],
    "Financials (XLF)":             ["JPM","BAC","GS","MS","BLK"],
    "Healthcare (XLV)":             ["JNJ","PFE","UNH","ABBV","MRK"],
    "Consumer Staples (XLP)":       ["PG","KO","COST","WMT","CL"],
    "Energy (XLE)":                 ["XOM","CVX","COP","SLB","EOG"],
    "Consumer Discret. (XLY)":      ["TSLA","AMZN","NKE","MCD","SBUX"],
    "Industrials (XLI)":            ["CAT","HON","BA","GE","RTX"],
    "Clean Energy (ICLN)":          ["BE","ENPH","PLUG","NEE","FSLR"],
}

PERIOD_MAP = {"5D":5,"1M":21,"Q":63,"1Y":252,"5Y":1260,"10Y":2520}
SEV_COLOR  = {"CRITICAL":"#f85149","WARNING":"#e3b341","WATCH":"#58a6ff","NORMAL":"#3fb950","POSITIVE_MOMENTUM":"#3fb950","REVIEW":"#a371f7"}
SEV_SHORT  = {"CRITICAL":"CRIT","WARNING":"WARN","WATCH":"WTCH","NORMAL":"NORM","POSITIVE_MOMENTUM":"POS+","REVIEW":"REV"}
SEV_CLS    = {"CRITICAL":"s-critical","WARNING":"s-warning","WATCH":"s-watch","NORMAL":"s-normal","POSITIVE_MOMENTUM":"s-positive","REVIEW":"s-review"}

# ── Data ──────────────────────────────────────────────────────────────────────
@st.cache_data
def load_decisions():
    p = Path("data/decisions/decisions.parquet")
    if p.exists():
        df = pd.read_parquet(p)
        df["_s"] = df["severity"].map({"CRITICAL":0,"WARNING":1,"WATCH":2,"REVIEW":3,"POSITIVE_MOMENTUM":4,"NORMAL":5}).fillna(99)
        return df.sort_values("_s").drop(columns=["_s"])
    tickers = list(COMPANY_NAMES.keys())
    np.random.seed(42)
    sevs = ["CRITICAL"]*35+["WARNING"]*5+["WATCH"]*3+["NORMAL"]*2
    np.random.shuffle(sevs)
    return pd.DataFrame({
        "ticker":tickers,"date":["2026-03-20"]*len(tickers),
        "severity":sevs[:len(tickers)],"action":["ESCALATE"]*len(tickers),
        "confidence":np.random.uniform(0,1,len(tickers)).round(2),
        "context":["idiosyncratic"]*len(tickers),
        "momentum_signal":np.random.choice(["falling","neutral","rising"],len(tickers)),
        "summary":["Multiple risk indicators triggered."]*len(tickers),
        "caution_flag":[None]*len(tickers),
        "direction":["down"]*len(tickers),"p_down":[0.7]*len(tickers),
    })

@st.cache_data
def load_detection(ticker):
    p = Path(f"data/detection/{ticker}.parquet")
    if p.exists():
        df = pd.read_parquet(p)
        df["Date"] = pd.to_datetime(df["Date"])
        return df.sort_values("Date").reset_index(drop=True)
    return None

@st.cache_data
def load_news():
    p = Path("data/explanations/llm_news_enriched.parquet")
    if p.exists(): return pd.read_parquet(p)
    return None

@st.cache_data
def load_spx():
    p = Path("data/raw/references/^SPX.parquet")
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df.columns = [c.capitalize() if c.lower() in ("open","high","low","close","volume") else c for c in df.columns]
    if "Date" not in df.columns and df.index.name == "Date":
        df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").reset_index(drop=True)

@st.cache_data(ttl=300)
def load_price_summary():
    """Last close price + 1-day % change for every ticker with detection data."""
    result = {}
    det_dir = Path("data/detection")
    if not det_dir.exists():
        return result
    all_t = set(COMPANY_NAMES)
    for ticker in all_t:
        f = det_dir / f"{ticker}.parquet"
        if not f.exists():
            continue
        try:
            tmp = pd.read_parquet(f, columns=["Date", "Close"])
            tmp = tmp.dropna(subset=["Close"]).sort_values("Date").tail(2)
            if len(tmp) >= 2:
                prev = float(tmp.iloc[-2]["Close"])
                last = float(tmp.iloc[-1]["Close"])
                result[ticker] = (last, (last - prev) / prev * 100)
            elif len(tmp) == 1:
                result[ticker] = (float(tmp.iloc[-1]["Close"]), 0.0)
        except Exception:
            pass
    return result

def safe_list(val):
    if isinstance(val, list): return val
    try: return ast.literal_eval(str(val))
    except: return []

def build_analysis(det_df, dec_row, ticker, name, lang="english"):
    """Generate a styled HTML analysis panel from existing pipeline data."""
    if det_df is None or det_df.empty:
        return None
    r   = det_df.iloc[-1]
    row = dec_row.iloc[0]

    rsi      = r.get("rsi", 50)
    mom5     = r.get("momentum_5", 0)
    mom10    = r.get("momentum_10", 0)
    vol      = r.get("volatility", 0)
    ret_1d   = r.get("returns", 0)
    drawdown = r.get("max_drawdown_30d", 0)
    regime   = str(r.get("regime", "unknown"))
    excess   = r.get("excess_return", 0)
    obv      = r.get("obv_signal", 0)
    sev      = row.get("severity", "NORMAL")
    direction= row.get("direction", "stable")
    p_down   = float(row.get("p_down", 0.33))
    p_up     = max(0.0, 1 - p_down - 0.15)
    mom_sig  = row.get("momentum_signal", "neutral")
    vol_ma20 = r.get("volume_ma20", 1)
    volume   = r.get("Volume", 0)
    date_str = str(r.get("Date", ""))[:10]
    action   = row.get("action", "—")
    caution  = row.get("caution_flag", None)

    # Icon helpers
    def pos(icon="▲"): return f'<span class="an-pos">{icon}</span>'
    def risk(icon="●"): return f'<span class="an-risk">{icon}</span>'
    def warn(icon="▲"): return f'<span class="an-warn">{icon}</span>'
    def neu(icon="→"):  return f'<span class="an-neu">{icon}</span>'

    _TIPS_ALL = {
        "english": {
            "Trend: bullish":   "Bullish: Price is rising — positive market direction",
            "Trend: bearish":   "Bearish: Price is falling — negative market direction",
            "Trend: sideways":  "Sideways: No clear trend — price moves without direction",
            "Momentum: rising": "Rising: Price momentum is accelerating upward",
            "Momentum: falling":"Falling: Price momentum is accelerating downward",
            "Momentum: neutral":"Neutral: No clear directional force in the market",
            "overbought":       "Overbought (RSI >70): Too many buyers — correction possible",
            "oversold":         "Oversold (RSI <30): Too many sellers — recovery possible",
            "RSI:":             "RSI (0–100): Measures if a stock is overbought or oversold. >70 = overbought, <30 = oversold",
            "Volume:":          "Trading volume compared to the 20-day average",
            "elevated":         "Elevated volume: Unusually high trading — often on news or breakouts",
            "thin":             "Thin volume: Low trading activity — price direction less reliable",
            "Drawdown":         "Maximum loss from the last peak over 30 days",
            "Regime:":          "Market regime: Statistical environment of the stock (e.g. calm, volatile, trending)",
            "outperforming":    "Outperformance: Stock rising more than the overall market",
            "underperforming":  "Underperformance: Stock falling more than the overall market",
            "in line":          "In line: Stock moving similarly to the overall market",
            "OBV:":             "OBV (On-Balance Volume): Measures buying vs. selling pressure via volume",
            "buying pressure":  "Buying pressure: More buyers than sellers — bullish signal",
            "selling pressure": "Selling pressure: More sellers than buyers — bearish signal",
            "Return:":          "Daily return: Price change on the last trading day in %",
            "P(down)":          "AI probability: Price will fall in the next 5 days",
            "P(up)":            "AI probability: Price will rise in the next 5 days",
            "Ann. vol":         "Annualised volatility: Estimated yearly price swing in %",
            "Action:":          "Recommended action from the AI system based on all signals",
            "Risk level:":      "Overall risk level of the stock according to AI analysis",
            "CRITICAL":         "CRITICAL: Multiple risk models triggered simultaneously — immediate attention needed",
            "WARNING":          "WARNING: Elevated risk detected — close monitoring recommended",
            "WATCH":            "WATCH: Slightly elevated risk — keep situation under observation",
            "NORMAL":           "NORMAL: No elevated risk — stock behaving within normal range",
        },
        "german": {
            "Trend: bullish":   "Bullish: Der Kurs steigt – positive Marktrichtung",
            "Trend: bearish":   "Bearish: Der Kurs fällt – negative Marktrichtung",
            "Trend: sideways":  "Sideways: Kein klarer Trend – Kurs bewegt sich seitwärts",
            "Momentum: rising": "Rising: Kursdynamik beschleunigt sich nach oben",
            "Momentum: falling":"Falling: Kursdynamik beschleunigt sich nach unten",
            "Momentum: neutral":"Neutral: Keine klare Kraft in eine Richtung",
            "overbought":       "Überkauft (RSI >70): Zu viele Käufe – Korrektur möglich",
            "oversold":         "Überverkauft (RSI <30): Zu viele Verkäufe – Erholung möglich",
            "RSI:":             "RSI (0–100): Misst ob die Aktie über- oder unterkauft ist. >70 = überkauft, <30 = überverkauft",
            "Volume:":          "Handelsvolumen im Vergleich zum 20-Tage-Durchschnitt",
            "elevated":         "Erhöhtes Volumen: Ungewöhnlich viel Handel – oft bei News oder Ausbrüchen",
            "thin":             "Dünnes Volumen: Wenig Handel – Kursrichtung weniger verlässlich",
            "Drawdown":         "Maximaler Verlust vom letzten Höchststand in 30 Tagen",
            "Regime:":          "Marktregime: Statistisches Umfeld der Aktie (z.B. ruhig, volatil, trendend)",
            "outperforming":    "Outperformance: Aktie steigt stärker als der Gesamtmarkt",
            "underperforming":  "Underperformance: Aktie fällt stärker als der Gesamtmarkt",
            "in line":          "In line: Aktie bewegt sich ähnlich wie der Gesamtmarkt",
            "OBV:":             "OBV (On-Balance-Volume): Misst Kauf- vs. Verkaufsdruck anhand des Volumens",
            "buying pressure":  "Kaufdruck: Mehr Käufer als Verkäufer – bullishes Signal",
            "selling pressure": "Verkaufsdruck: Mehr Verkäufer als Käufer – bearishes Signal",
            "Return:":          "Tagesrendite: Kursveränderung am letzten Handelstag in %",
            "P(down)":          "KI-Wahrscheinlichkeit: Kurs fällt in den nächsten 5 Tagen",
            "P(up)":            "KI-Wahrscheinlichkeit: Kurs steigt in den nächsten 5 Tagen",
            "Ann. vol":         "Annualisierte Volatilität: Geschätzte jährliche Kursschwankung in %",
            "Action:":          "Empfohlene Maßnahme des KI-Systems basierend auf allen Signalen",
            "Risk level:":      "Gesamtes Risikoniveau der Aktie laut KI-Analyse",
            "CRITICAL":         "CRITICAL: Mehrere Risikomodelle schlagen an – sofortige Beobachtung nötig",
            "WARNING":          "WARNING: Erhöhtes Risiko erkannt – aufmerksame Beobachtung empfohlen",
            "WATCH":            "WATCH: Leicht erhöhtes Risiko – Situation im Auge behalten",
            "NORMAL":           "NORMAL: Kein erhöhtes Risiko – Aktie verhält sich im Rahmen",
        },
        "arabic": {
            "Trend: bullish":   "صاعد: السعر يرتفع — اتجاه إيجابي في السوق",
            "Trend: bearish":   "هابط: السعر ينخفض — اتجاه سلبي في السوق",
            "Trend: sideways":  "جانبي: لا اتجاه واضح — السعر يتحرك أفقياً",
            "Momentum: rising": "صاعد: زخم السعر يتسارع للأعلى",
            "Momentum: falling":"هابط: زخم السعر يتسارع للأسفل",
            "Momentum: neutral":"محايد: لا قوة اتجاهية واضحة في السوق",
            "overbought":       "مشتراة بإفراط (RSI >70): عمليات شراء مفرطة — تصحيح محتمل",
            "oversold":         "مباعة بإفراط (RSI <30): عمليات بيع مفرطة — ارتداد محتمل",
            "RSI:":             "RSI (0–100): يقيس ما إذا كانت الأسهم مشتراة أو مباعة بإفراط. >70 مشتراة بإفراط، <30 مباعة بإفراط",
            "Volume:":          "حجم التداول مقارنةً بالمتوسط لـ20 يوماً",
            "elevated":         "حجم مرتفع: تداول غير معتاد — غالباً عند الأخبار أو الاختراقات",
            "thin":             "حجم ضعيف: نشاط تداول منخفض — اتجاه السعر أقل موثوقية",
            "Drawdown":         "أقصى خسارة من آخر ذروة خلال 30 يوماً",
            "Regime:":          "نظام السوق: البيئة الإحصائية للسهم (مثل: هادئ، متقلب، في اتجاه)",
            "outperforming":    "أداء متفوق: السهم يرتفع أكثر من السوق العام",
            "underperforming":  "أداء متأخر: السهم ينخفض أكثر من السوق العام",
            "in line":          "متوافق: السهم يتحرك بشكل مشابه للسوق العام",
            "OBV:":             "OBV: يقيس ضغط الشراء مقابل البيع من خلال حجم التداول",
            "buying pressure":  "ضغط شراء: المشترون أكثر من البائعين — إشارة صاعدة",
            "selling pressure": "ضغط بيع: البائعون أكثر من المشترين — إشارة هابطة",
            "Return:":          "العائد اليومي: تغير السعر في آخر يوم تداول بالنسبة المئوية",
            "P(down)":          "احتمال الذكاء الاصطناعي: انخفاض السعر خلال 5 أيام القادمة",
            "P(up)":            "احتمال الذكاء الاصطناعي: ارتفاع السعر خلال 5 أيام القادمة",
            "Ann. vol":         "التذبذب السنوي: تقدير التأرجح السعري السنوي بالنسبة المئوية",
            "Action:":          "الإجراء الموصى به من نظام الذكاء الاصطناعي بناءً على جميع الإشارات",
            "Risk level:":      "مستوى المخاطرة الإجمالي للسهم وفق تحليل الذكاء الاصطناعي",
            "CRITICAL":         "حرج: نماذج مخاطر متعددة تنبّه في آن واحد — يحتاج انتباهاً فورياً",
            "WARNING":          "تحذير: مخاطرة مرتفعة — مراقبة دقيقة موصى بها",
            "WATCH":            "مراقبة: مخاطرة مرتفعة قليلاً — ابقِ الوضع تحت المراقبة",
            "NORMAL":           "طبيعي: لا مخاطرة مرتفعة — السهم يتصرف ضمن النطاق الطبيعي",
        },
    }
    _TIPS = _TIPS_ALL.get(lang, _TIPS_ALL["english"])
    def r_row(ic, txt):
        tip = next((v for k, v in _TIPS.items() if k.lower() in txt.lower()), "")
        tip_attr = f' data-tip="{tip}"' if tip else ""
        return f'<div class="an-row">{ic}<span class="an-text"{tip_attr}>{txt}</span></div>'

    # Trend
    if "up" in regime or "bull" in regime or (mom5 > 0 and mom10 > 0):
        trend_ic, trend_str = pos(), "bullish"
    elif "down" in regime or "bear" in regime or (mom5 < 0 and mom10 < 0):
        trend_ic, trend_str = risk(), "bearish"
    else:
        trend_ic, trend_str = neu(), "sideways"

    # Momentum
    mom_ic = pos() if mom_sig == "rising" else risk() if mom_sig == "falling" else neu()

    # Severity
    sev_ic = pos() if sev in ("NORMAL", "POSITIVE_MOMENTUM") else risk() if sev in ("CRITICAL", "WARNING") else warn()

    # RSI
    if rsi > 70:    rsi_ic, rsi_str = risk(), f"overbought ({rsi:.1f})"
    elif rsi < 30:  rsi_ic, rsi_str = pos(), f"oversold ({rsi:.1f}) — recovery signal"
    elif rsi > 55:  rsi_ic, rsi_str = pos(), f"strong ({rsi:.1f})"
    elif rsi < 45:  rsi_ic, rsi_str = risk(), f"weak ({rsi:.1f})"
    else:           rsi_ic, rsi_str = neu(), f"neutral ({rsi:.1f})"

    # Volume
    vol_ratio = (volume / vol_ma20) if vol_ma20 > 0 else 1
    if vol_ratio > 1.5:    vol_ic, vol_str = warn("▲"), f"{vol_ratio:.1f}x avg (elevated)"
    elif vol_ratio < 0.6:  vol_ic, vol_str = risk(), f"{vol_ratio:.1f}x avg (thin)"
    elif vol_ratio >= 0.9: vol_ic, vol_str = pos(), f"{vol_ratio:.1f}x avg (normal)"
    else:                  vol_ic, vol_str = neu(), f"{vol_ratio:.1f}x avg (below avg)"

    # Drawdown
    dd_str = f"{drawdown*100:.1f}%" if drawdown else "N/A"
    if drawdown and abs(drawdown) > 0.15:   dd_ic = risk("▼")
    elif drawdown and abs(drawdown) > 0.08: dd_ic = warn("▼")
    else:                                   dd_ic = neu()

    # Market context
    if excess > 0.01:    ctx_ic, ctx_str = pos(), f"outperforming +{excess*100:.1f}%"
    elif excess < -0.01: ctx_ic, ctx_str = risk(), f"underperforming {excess*100:.1f}%"
    else:                ctx_ic, ctx_str = neu(), "in line w/ market"

    # OBV + return
    obv_ic  = pos() if obv > 0 else risk()
    obv_str = "buying pressure" if obv > 0 else "selling pressure"
    ret_ic  = pos() if ret_1d > 0.005 else risk() if ret_1d < -0.005 else neu()

    # Direction forecast
    d_arrow = "▲" if direction == "up" else "▼" if direction == "down" else "→"
    d_conf  = int((1 - p_down) * 100) if direction == "up" else int(p_down * 100)
    dir_ic  = pos(d_arrow) if direction == "up" else risk(d_arrow) if direction == "down" else neu(d_arrow)
    pd_ic   = risk() if p_down > 0.5 else pos()

    vol_ann = vol * 100 * 16
    va_ic   = risk() if vol_ann > 40 else warn() if vol_ann > 25 else pos()
    act_ic  = risk() if sev in ("CRITICAL", "WARNING") else pos()

    # Build sections
    exec_html = (
        r_row(trend_ic, f"Trend: {trend_str}") +
        r_row(mom_ic,   f"Momentum: {mom_sig}") +
        r_row(sev_ic,   f"Risk level: {sev.replace('_',' ')}")
    )
    tech_html = (
        r_row(rsi_ic,  f"RSI: {rsi_str}") +
        r_row(vol_ic,  f"Volume: {vol_str}") +
        r_row(dd_ic,   f"Drawdown 30D: {dd_str}") +
        r_row(neu(),   f"Regime: {regime}")
    )
    ctx_html = (
        r_row(ctx_ic, ctx_str) +
        r_row(obv_ic, f"OBV: {obv_str}") +
        r_row(ret_ic, f"Return: {ret_1d*100:+.2f}%")
    )
    fcast_html = (
        r_row(dir_ic,  f"{direction.upper()} — {d_conf}% confidence") +
        r_row(pd_ic,   f"P(down) {p_down*100:.0f}% · P(up) {p_up*100:.0f}%") +
        r_row(va_ic,   f"Ann. vol: {vol_ann:.1f}%") +
        r_row(act_ic,  f"Action: {action}")
    )
    if caution:
        fcast_html += r_row(warn("⚠"), caution)

    return f"""
    <div class="analysis-panel">
      <div class="an-header">Market Analysis — {name} ({ticker})<span class="an-date">{date_str}</span></div>
      <div class="an-grid">
        <div class="an-section"><div class="an-title">Executive Summary</div>{exec_html}</div>
        <div class="an-section"><div class="an-title">Technical Picture</div>{tech_html}</div>
        <div class="an-section"><div class="an-title">Market Context</div>{ctx_html}</div>
        <div class="an-section"><div class="an-title">AI Forecast (5D)</div>{fcast_html}</div>
      </div>
    </div>"""


def explain_anomaly(r, ticker_name, date_str):
    detectors = []
    if r.get("z_anomaly", False):
        detectors.append(f"**Z-Score (30d)**: Return deviated strongly from 30-day norm (z = {r.get('z_score', 0):.2f})")
    if r.get("z_anomaly_60", False):
        detectors.append(f"**Z-Score (60d)**: Deviation confirmed over 60-day window (z = {r.get('z_score_60', 0):.2f})")
    if r.get("if_anomaly", False):
        detectors.append("**Isolation Forest**: Multivariate pattern inconsistent with historical calm periods")
    if r.get("ae_anomaly", False):
        detectors.append(f"**LSTM Autoencoder**: High reconstruction error — sequence breaks learned pattern (error = {r.get('ae_error', 0):.4f})")

    if not detectors:
        return None

    ctx = []
    ret = r.get("returns", None)
    if ret is not None: ctx.append(f"Daily return: {ret*100:.2f}%")
    vol = r.get("volatility", None)
    if vol is not None: ctx.append(f"Volatility: {vol*100:.2f}%")
    rsi = r.get("rsi", None)
    if rsi is not None: ctx.append(f"RSI: {rsi:.1f}")
    vz = r.get("volume_zscore", None)
    if vz is not None and abs(vz) > 1.5: ctx.append(f"Volume spike (z = {vz:.1f})")
    if r.get("market_anomaly", False): ctx.append("Market-wide event")
    if r.get("sector_anomaly", False): ctx.append("Sector-wide event")
    if r.get("is_high_volume", False): ctx.append("Unusually high volume")

    score = int(r.get("anomaly_score", len(detectors)))
    lines = [
        f"#### {ticker_name} — {date_str}",
        f"**{score} of 4 detectors triggered**",
        "",
        *[f"- {d}" for d in detectors],
    ]
    if ctx:
        lines += ["", "**Context:** " + " · ".join(ctx)]
    return "\n".join(lines)

def build_investor_summary(det_df, dec_row, row, news_df, ticker) -> str:
    """Build a short investor-friendly signal summary shown above the full LLM report."""
    if det_df is None or det_df.empty:
        return ""

    last = det_df.iloc[-1]
    signals = []

    # ── Risk level ────────────────────────────────────────────────
    sev    = str(row.get("severity", "NORMAL"))
    action = str(row.get("action", "MONITOR"))
    conf   = float(row.get("confidence", 0))
    sev_color = {"CRITICAL": "#e05252", "WARNING": "#e0a030", "NORMAL": "#52b788"}.get(sev, "#adb5bd")
    signals.append(
        f'<span style="color:{sev_color};font-weight:600">{sev}</span>'
        f'<span style="color:#5c7080"> — Action: </span>'
        f'<span style="color:#adb5bd">{action}</span>'
        f'<span style="color:#5c7080"> · Confidence: </span>'
        f'<span style="color:#adb5bd">{int(conf*100)}%</span>'
    )

    # ── Anomaly detection ─────────────────────────────────────────
    detectors = []
    if last.get("z_anomaly"):    detectors.append("Z-Score (30D)")
    if last.get("z_anomaly_60"): detectors.append("Z-Score (60D)")
    if last.get("if_anomaly"):   detectors.append("Isolation Forest")
    if last.get("ae_anomaly"):   detectors.append("LSTM Autoencoder")

    if detectors:
        det_str = ", ".join(detectors)
        signals.append(
            f'<span style="color:#e05252">⬤</span>'
            f'<span style="color:#adb5bd"> Anomaly detected [{det_str}] — unusual price or volume pattern identified</span>'
        )
    else:
        signals.append(
            f'<span style="color:#52b788">▲</span>'
            f'<span style="color:#adb5bd"> No anomalies detected — behavior within normal range</span>'
        )

    # ── AI direction forecast ─────────────────────────────────────
    direction = "stable"
    p_down    = 0.33
    if dec_row is not None and not dec_row.empty:
        direction = str(dec_row.get("direction", "stable"))
        p_down    = float(dec_row.get("p_down", 0.33))
    p_up = max(0.0, 1 - p_down - 0.15)

    dir_icon  = {"up": "▲", "down": "▼", "stable": "→"}.get(direction, "→")
    dir_color = {"up": "#1de9b6", "down": "#e05252", "stable": "#adb5bd"}.get(direction, "#adb5bd")
    signals.append(
        f'<span style="color:{dir_color}">{dir_icon}</span>'
        f'<span style="color:#adb5bd"> AI 5-day forecast: <strong style="color:{dir_color}">{direction.upper()}</strong>'
        f' — P(up) {int(p_up*100)}% · P(down) {int(p_down*100)}%</span>'
    )

    # ── Momentum ──────────────────────────────────────────────────
    mom = "neutral"
    if dec_row is not None and not dec_row.empty:
        mom = str(dec_row.get("momentum_signal", "neutral"))
    elif "momentum_signal" in last.index:
        mom = str(last.get("momentum_signal", "neutral"))
    mom_icon  = {"rising": "▲", "falling": "▼", "neutral": "→"}.get(mom, "→")
    mom_color = {"rising": "#1de9b6", "falling": "#e05252", "neutral": "#adb5bd"}.get(mom, "#adb5bd")
    signals.append(
        f'<span style="color:{mom_color}">{mom_icon}</span>'
        f'<span style="color:#adb5bd"> Momentum: <strong style="color:{mom_color}">{mom.capitalize()}</strong></span>'
    )

    # ── News sentiment ────────────────────────────────────────────
    if news_df is not None and ticker in news_df["ticker"].values:
        nr         = news_df[news_df["ticker"] == ticker].iloc[0]
        sentiments = nr.get("news_sentiment", [])
        if isinstance(sentiments, list) and sentiments:
            counts  = {"positive": 0, "negative": 0, "neutral": 0}
            for s in sentiments:
                counts[s] = counts.get(s, 0) + 1
            dominant = max(counts, key=counts.get)
            sent_icon  = {"positive": "▲", "negative": "⬤", "neutral": "→"}.get(dominant, "→")
            sent_color = {"positive": "#52b788", "negative": "#e05252", "neutral": "#adb5bd"}.get(dominant, "#adb5bd")
            signals.append(
                f'<span style="color:{sent_color}">{sent_icon}</span>'
                f'<span style="color:#adb5bd"> News sentiment: <strong style="color:{sent_color}">{dominant.capitalize()}</strong></span>'
            )

    rows_html = "".join(
        f'<div style="padding:4px 0;border-bottom:1px solid #1a2332;font-size:11px;'
        f'font-family:\'IBM Plex Sans\',sans-serif;line-height:1.5">{s}</div>'
        for s in signals
    )

    return f"""
    <div style="background:#0d1117;border:1px solid #1a2332;border-radius:3px;
                padding:12px 18px;margin-top:8px;margin-bottom:4px">
      <div style="font-size:9px;letter-spacing:2px;text-transform:uppercase;
                  color:#5c7080;font-family:'IBM Plex Mono',monospace;
                  border-bottom:1px solid #1a2332;padding-bottom:7px;margin-bottom:8px">
        Investor Summary
      </div>
      {rows_html}
    </div>"""


def _translate(text: str, language: str, ticker: str) -> str:
    """Translate llm_summary to target language via Groq. Cached in session state."""
    if language == "english":
        return text
    if "llm_cache" not in st.session_state:
        st.session_state.llm_cache = {}
    cache_key = f"{ticker}_{language}"
    if cache_key in st.session_state.llm_cache:
        return st.session_state.llm_cache[cache_key]
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return text
    lang_map = {"german": "German", "arabic": "Arabic"}
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content":
                f"Translate the following financial analysis to {lang_map[language]}. "
                f"Keep all section headers (**bold**), numbers, and formatting exactly. "
                f"Only translate the text, nothing else.\n\n{text}"}],
            temperature=0.1,
            max_tokens=1800,
        )
        result = resp.choices[0].message.content.strip()
        st.session_state.llm_cache[cache_key] = result
        return result
    except Exception as e:
        st.warning(f"Translation failed: {e}")
        return text

# ── Session State ─────────────────────────────────────────────────────────────
decisions  = load_decisions()
news_df    = load_news()
price_data = load_price_summary()
all_tickers = decisions["ticker"].tolist()

if "selected"       not in st.session_state: st.session_state.selected       = all_tickers[0]
if "period"         not in st.session_state: st.session_state.period         = "1M"
if "category"       not in st.session_state: st.session_state.category       = "Stocks"
if "anomaly_date"   not in st.session_state: st.session_state.anomaly_date   = None
if "language"       not in st.session_state: st.session_state.language       = "english"
if "llm_cache"      not in st.session_state: st.session_state.llm_cache      = {}

SECTORS = {
    "Technology":        ["AAPL","MSFT","NVDA","AMD","GOOG"],
    "AI & Software":     ["PLTR","META","CRM","SNOW","AI"],
    "Consumer":          ["TSLA","AMZN","NKE","MCD","SBUX"],
    "Financials":        ["JPM","BAC","GS","MS","BLK"],
    "Healthcare":        ["JNJ","PFE","UNH","ABBV","MRK"],
    "Consumer Staples":  ["PG","KO","COST","WMT","CL"],
    "Energy":            ["XOM","CVX","COP","SLB","EOG"],
    "Industrials":       ["CAT","HON","BA","GE","RTX"],
    "Clean Energy":      ["BE","ENPH","PLUG","NEE","FSLR"],
}

def _find_sector(ticker):
    for s, ts in SECTORS.items():
        if ticker in ts: return s
    return list(SECTORS.keys())[0]

if "selected_sector" not in st.session_state:
    st.session_state.selected_sector = _find_sector(st.session_state.selected)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="logo">Fin<span>Watch</span> AI</div>', unsafe_allow_html=True)

    # Sector selectbox
    sector_names = list(SECTORS.keys())
    cur_sector = st.session_state.selected_sector
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
        if row_s.empty: continue
        name   = COMPANY_NAMES.get(t, t)
        active = "★ " if t == st.session_state.selected else ""
        btn_col, chg_col = st.columns([7, 3])
        with btn_col:
            if st.button(f"{active}{name} ({t})", key=f"stock_{t}", use_container_width=True):
                st.session_state.selected = t
                st.session_state.anomaly_date = None
                st.rerun()
        with chg_col:
            if t in price_data:
                _, pct = price_data[t]
                arrow = "▲" if pct > 0 else ("▼" if pct < 0 else "—")
                css   = "wl-chg-up" if pct > 0 else ("wl-chg-dn" if pct < 0 else "wl-chg-fl")
                st.markdown(
                    f'<div class="{css}">{arrow}{abs(pct):.1f}%</div>',
                    unsafe_allow_html=True,
                )

    st.markdown("<hr style='border-color:#1a2332;margin:8px 0'>", unsafe_allow_html=True)
    st.markdown('<div class="sb-label">Language</div>', unsafe_allow_html=True)
    lang_cols = st.columns(3)
    for col, (code, label) in zip(lang_cols, [("english","EN"), ("german","DE"), ("arabic","AR")]):
        with col:
            active = st.session_state.language == code
            if st.button(label, key=f"lang_{code}",
                         type="primary" if active else "secondary"):
                st.session_state.language = code
                st.session_state.llm_cache = {}   # clear cache so translation reruns
                st.rerun()

# ── Main ──────────────────────────────────────────────────────────────────────
ticker  = st.session_state.selected

# ── Market Overview (^SPX) ────────────────────────────────────────────────────
if ticker == "^SPX":
    spx_df = load_spx()

    # Header
    spx_last, spx_chg = None, None
    if spx_df is not None and len(spx_df) > 1:
        spx_last = spx_df["Close"].iloc[-1]
        spx_chg  = (spx_df["Close"].iloc[-1] - spx_df["Close"].iloc[-2]) / spx_df["Close"].iloc[-2] * 100

    chg_html  = ""
    if spx_chg is not None:
        if spx_chg >= 0:
            chg_html = f'<span class="sh-up">▲ +{spx_chg:.2f}%</span>'
        else:
            chg_html = f'<span class="sh-dn">▼ {spx_chg:.2f}%</span>'
    price_html = f'<span class="sh-price">{spx_last:,.2f}</span>' if spx_last else ""

    st.markdown(f"""
    <div class="stock-header">
      <span class="sh-name">S&P 500</span>
      <span class="sh-sub">^GSPC · Market Index</span>
      {price_html}
      {chg_html}
    </div>""", unsafe_allow_html=True)

    # SPX Chart
    st.markdown('<div class="chart-label">S&P 500 Price History</div>', unsafe_allow_html=True)
    if spx_df is not None:
        plot_spx = spx_df.copy()
        spx_prev   = plot_spx["Close"].shift(1).fillna(plot_spx["Close"])
        spx_colors = ["#1de9b6" if c >= p else "#f85149"
                      for c, p in zip(plot_spx["Close"], spx_prev)]
        fig = go.Figure()
        has_ohlc_spx = all(c in plot_spx.columns for c in ["Open", "High", "Low"])
        if has_ohlc_spx:
            fig.add_trace(go.Candlestick(
                x=plot_spx["Date"],
                open=plot_spx["Open"], high=plot_spx["High"],
                low=plot_spx["Low"],   close=plot_spx["Close"],
                increasing_line_color="#1de9b6", decreasing_line_color="#f85149",
                increasing_fillcolor="#1de9b6",  decreasing_fillcolor="#f85149",
                name="SPX", whiskerwidth=0.4,
            ))
        else:
            fig.add_trace(go.Scatter(
                x=plot_spx["Date"], y=plot_spx["Close"].round(2),
                mode="markers",
                marker=dict(symbol="line-ew-open", color=spx_colors,
                            size=6, line=dict(width=2)),
                hovertemplate="%{x|%d %b %Y}<br><b>%{y:,.2f}</b><extra></extra>",
            ))
        spx_x_end   = plot_spx["Date"].iloc[-1]
        spx_x_start = spx_x_end - pd.DateOffset(months=2)
        _spx_rs = [
            dict(count=1,  label="1M", step="month", stepmode="backward"),
            dict(count=3,  label="3M", step="month", stepmode="backward"),
            dict(count=6,  label="6M", step="month", stepmode="backward"),
            dict(count=1,  label="1Y", step="year",  stepmode="backward"),
            dict(step="all", label="All"),
        ]
        _spx_tickformatstops = [
            dict(dtickrange=[None,    86400000], value="%d %b"),
            dict(dtickrange=[86400000, "M1"],   value="%d %b '%y"),
            dict(dtickrange=["M1",    "M12"],   value="%b '%y"),
            dict(dtickrange=["M12",   None],    value="%Y"),
        ]
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d1117",
            xaxis=dict(
                showgrid=True, gridcolor="#1a2332", color="#5c7080",
                tickfont=dict(size=10, family="IBM Plex Mono"),
                tickformatstops=_spx_tickformatstops,
                range=[spx_x_start, spx_x_end],
                rangeselector=dict(
                    buttons=_spx_rs,
                    bgcolor="#0d1117", activecolor="#1a2332",
                    bordercolor="#314158", borderwidth=1,
                    font=dict(color="#8b949e", size=9, family="IBM Plex Mono"),
                    x=0, y=1.04,
                ),
                rangeslider=dict(visible=True, bgcolor="#080c10",
                                 bordercolor="#1a2332", thickness=0.05),
            ),
            yaxis=dict(showgrid=True, gridcolor="#1a2332", color="#5c7080",
                       tickfont=dict(size=10, family="IBM Plex Mono"),
                       autorange=True),
            hoverlabel=dict(
                bgcolor="#1a2332", bordercolor="#314158",
                font=dict(size=10, family="IBM Plex Sans", color="#cdd9e5"),
            ),
            margin=dict(t=30, b=10, l=10, r=10), height=400,
            showlegend=False, hovermode="x",
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.markdown('<div style="height:300px;display:flex;align-items:center;justify-content:center;color:#5c7080;font-size:12px;font-family:IBM Plex Mono">No SPX data available</div>', unsafe_allow_html=True)

    # Sector Performance Table
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
        worst    = sector_dec.assign(_o=sector_dec["severity"].map(sev_order).fillna(99)).sort_values("_o").iloc[0]["severity"]
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

    # Market pulse + top critical below, side by side
    _spx_c1, _spx_c2 = st.columns(2, gap="small")
    with _spx_c1:
        total     = len(decisions)
        by_sev    = decisions["severity"].value_counts()
        critical  = by_sev.get("CRITICAL", 0)
        warning   = by_sev.get("WARNING", 0)
        watch     = by_sev.get("WATCH", 0)
        normal    = by_sev.get("NORMAL", 0) + by_sev.get("POSITIVE_MOMENTUM", 0)
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
sev = row["severity"]

det_df = load_detection(ticker)
name   = COMPANY_NAMES.get(ticker, ticker)

last_price, last_chg = None, None
if det_df is not None and "Close" in det_df.columns and len(det_df) > 1:
    last_price = det_df["Close"].iloc[-1]
    last_chg   = (det_df["Close"].iloc[-1] - det_df["Close"].iloc[-2]) / det_df["Close"].iloc[-2] * 100

    # Header
    chg_html  = ""
    if last_chg is not None:
        if last_chg >= 0:
            chg_html = f'<span class="sh-up">▲ +{last_chg:.2f}%</span>'
        else:
            chg_html = f'<span class="sh-dn">▼ {last_chg:.2f}%</span>'
    price_html = f'<span class="sh-price">${last_price:.2f}</span>' if last_price else ""

    st.markdown(f"""
    <div class="stock-header">
      <span class="sh-name">{name}</span>
      <span class="sh-sub">{ticker} · NASDAQ</span>
      {price_html}
      {chg_html}
    </div>""", unsafe_allow_html=True)

    # ── Top row: Risk | Summary | News ────────────────────────────────────────
    _rc1, _rc2, _rc3 = st.columns(3, gap="small")

    with _rc1:
        sev_cls   = SEV_CLS.get(sev, "s-normal")
        dot_count = int(row["confidence"] * 4)
        dots_html = ("".join([f'<span class="d d-on"></span>' for _ in range(dot_count)]) +
                     "".join([f'<span class="d d-off"></span>' for _ in range(4-dot_count)]))
        mom     = row.get("momentum_signal", "neutral")
        mom_arr = "▲" if mom=="rising" else "▼" if mom=="falling" else "—"
        mom_cls = "rp-up" if mom=="rising" else "rp-dn" if mom=="falling" else "rp-val"
        direction = row.get("direction", "stable") if "direction" in row.index else "stable"
        p_down    = row.get("p_down", 0) if "p_down" in row.index else 0
        dir_arr   = "▼" if direction=="down" else "▲" if direction=="up" else "—"
        dir_cls   = "rp-dn" if direction=="down" else "rp-up" if direction=="up" else "rp-val"
        drawdown, es_ratio = None, None
        if det_df is not None:
            if "max_drawdown_30d" in det_df.columns:
                v = det_df["max_drawdown_30d"].iloc[-1]
                if not pd.isna(v): drawdown = v
            if "es_ratio" in det_df.columns:
                v = det_df["es_ratio"].iloc[-1]
                if not pd.isna(v): es_ratio = v
        dd_html = f'<span class="rp-val rp-dn">{drawdown*100:.1f}%</span>' if drawdown else '<span class="rp-val">—</span>'
        es_html = f'<span class="rp-val rp-dn">{es_ratio:.2f}</span>' if es_ratio else '<span class="rp-val">—</span>'
        st.markdown(f"""
        <div class="rp-section">
          <div class="rp-title">AI Risk Analysis</div>
          <div class="sev-big {sev_cls}">{sev.replace('_',' ')}</div>
          <div style="font-size:9px;color:#5c7080;margin-bottom:10px;font-family:IBM Plex Mono">{row.get('action','—')} · {row.get('date','—')}</div>
          <div class="rp-row"><span class="rp-key">Model agreement</span>
            <div class="dot-row">{dots_html}<span style="font-size:9px;color:#5c7080;margin-left:4px">{dot_count}/4</span></div>
          </div>
          <div class="rp-row"><span class="rp-key">Direction (5D)</span>
            <span class="rp-val {dir_cls}">{dir_arr} {direction} ({int(p_down*100)}%)</span></div>
          <div class="rp-row"><span class="rp-key">Momentum</span>
            <span class="rp-val {mom_cls}">{mom_arr} {mom}</span></div>
          <div class="rp-row"><span class="rp-key">ES Ratio</span>{es_html}</div>
          <div class="rp-row"><span class="rp-key">Drawdown 30D</span>{dd_html}</div>
        </div>""", unsafe_allow_html=True)

    with _rc2:
        summary = row.get("summary","")
        caution = row.get("caution_flag", None)
        if summary:
            caution_html = f'<div style="color:#e3b341;font-size:10px;margin-top:6px;font-family:IBM Plex Mono">⚠ {caution}</div>' if caution else ""
            st.markdown(f"""
            <div class="rp-section">
              <div class="rp-title">AI Summary</div>
              <div class="summary-text">{summary}{caution_html}</div>
            </div>""", unsafe_allow_html=True)

    with _rc3:
        if news_df is not None and ticker in news_df["ticker"].values:
            nr         = news_df[news_df["ticker"]==ticker].iloc[0]
            headlines  = safe_list(nr.get("top_news", []))
            sentiments = safe_list(nr.get("news_sentiment", []))
            sources    = safe_list(nr.get("news_sources", nr.get("news_urls", [])))
            if headlines:
                news_html = ""
                for i, h in enumerate(headlines[:3]):
                    sent    = sentiments[i] if i < len(sentiments) else "neutral"
                    s_cls   = {"positive":"s-pos","negative":"s-neg","neutral":"s-neu"}.get(sent,"s-neu")
                    source  = sources[i] if i < len(sources) and sources[i] else ""
                    if source.startswith("http"):
                        import urllib.parse
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
                st.markdown(f'<div class="rp-section"><div class="rp-title">News Sentiment</div>{news_html}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="rp-section"><div class="rp-title">News Sentiment</div><div style="font-size:10px;color:#5c7080;font-family:IBM Plex Mono">Run FinBERT pipeline first</div></div>', unsafe_allow_html=True)

    # Price chart — Close + EMA20
    st.markdown('<div class="chart-label">Price  ·  EMA 20</div>', unsafe_allow_html=True)

    if det_df is not None and "Close" in det_df.columns:
        plot_df  = det_df.copy()
        ema20    = plot_df["Close"].ewm(span=20).mean()
        close_last = plot_df["Close"].iloc[-1]
        ema20_last = ema20.iloc[-1]
        has_ohlc   = all(c in plot_df.columns for c in ["Open", "High", "Low"])
        price_min  = min((plot_df["Low"] if has_ohlc else plot_df["Close"]).min(), ema20.min()) * 0.98
        price_max  = max((plot_df["High"] if has_ohlc else plot_df["Close"]).max(), ema20.max()) * 1.02

        fig = go.Figure()
        # Price — candlestick if OHLC available, else colored daily ticks
        if has_ohlc:
            fig.add_trace(go.Candlestick(
                x=plot_df["Date"],
                open=plot_df["Open"], high=plot_df["High"],
                low=plot_df["Low"],   close=plot_df["Close"],
                increasing_line_color="#1de9b6", decreasing_line_color="#f85149",
                increasing_fillcolor="#1de9b6",  decreasing_fillcolor="#f85149",
                name="Price", whiskerwidth=0.4,
                hoverinfo="skip",
            ))
            # Ghost hover — only non-skip trace, so hovermode="closest" always picks it
            _hover_cd = np.column_stack([
                plot_df["Open"].round(2), plot_df["High"].round(2),
                plot_df["Low"].round(2),  plot_df["Close"].round(2),
                ema20.round(2),
            ])
            fig.add_trace(go.Scatter(
                x=plot_df["Date"], y=plot_df["Close"],
                mode="markers", marker=dict(opacity=0, size=8),
                customdata=_hover_cd, showlegend=False, name="",
                hovertemplate=(
                    "<b>%{x|%d %b '%y}</b><br>"
                    "Open   $%{customdata[0]:,.2f}<br>"
                    "High    $%{customdata[1]:,.2f}<br>"
                    "Low     $%{customdata[2]:,.2f}<br>"
                    "Close  $%{customdata[3]:,.2f}<br>"
                    "EMA20 $%{customdata[4]:,.2f}"
                    "<extra></extra>"
                ),
            ))
        else:
            prev_c = plot_df["Close"].shift(1).fillna(plot_df["Close"])
            day_colors = ["#1de9b6" if c >= p else "#f85149"
                          for c, p in zip(plot_df["Close"], prev_c)]
            fig.add_trace(go.Scatter(
                x=plot_df["Date"], y=plot_df["Close"].round(2),
                mode="markers",
                marker=dict(symbol="line-ew-open", color=day_colors,
                            size=8, line=dict(width=2.5)),
                name="Close",
                hovertemplate="<b>%{x|%d %b '%y}</b><br>Close $%{y:,.2f}<extra></extra>",
            ))
        # EMA20 on top — hover shown in candlestick template via customdata
        fig.add_trace(go.Scatter(
            x=plot_df["Date"], y=ema20.round(2),
            mode="lines", name="EMA20",
            line=dict(color="#a371f7", width=1.2, dash="dot"),
            hoverinfo="skip",
        ))

        # Anomaly dots — orange circle-open to distinguish from red candles
        if "combined_anomaly" in plot_df.columns:
            anom_combined = plot_df[plot_df["combined_anomaly"] == True]
            if len(anom_combined) > 0:
                score_labels = anom_combined["anomaly_score"].astype(int).astype(str) + "/4"
                fig.add_trace(go.Scatter(
                    x=anom_combined["Date"], y=anom_combined["Close"],
                    mode="markers",
                    marker=dict(color="rgba(0,0,0,0)", size=12, symbol="circle-open",
                                line=dict(color="#ff9800", width=2.5)),
                    name="Anomaly", hoverinfo="skip",
                ))

        # End-of-line labels — outside chart (xref=paper)
        label_gap = abs(close_last - ema20_last)
        close_yshift = (+8 if close_last >= ema20_last else -8) if label_gap / (price_max - price_min + 1e-9) < 0.04 else 0
        ema20_yshift = -close_yshift

        fig.add_annotation(x=1.01, xref="paper", y=close_last, yref="y",
                           text=f"<b>${close_last:.2f}</b>",
                           xanchor="left", showarrow=False,
                           font=dict(size=9, color="#1de9b6", family="IBM Plex Mono"),
                           yshift=close_yshift)
        fig.add_annotation(x=1.01, xref="paper", y=ema20_last, yref="y",
                           text=f"EMA20 ${ema20_last:.2f}",
                           xanchor="left", showarrow=False,
                           font=dict(size=9, color="#a371f7", family="IBM Plex Mono"),
                           yshift=ema20_yshift)

        x_end   = plot_df["Date"].iloc[-1]
        x_start = x_end - pd.DateOffset(months=2)
        # Y range from visible window only (not full history)
        vis_w = plot_df[plot_df["Date"] >= x_start]
        if has_ohlc and len(vis_w) > 0:
            y_lo = vis_w["Low"].min() * 0.98
            y_hi = vis_w["High"].max() * 1.02
        elif len(vis_w) > 0:
            y_lo = vis_w["Close"].min() * 0.98
            y_hi = vis_w["Close"].max() * 1.02
        else:
            y_lo, y_hi = price_min, price_max
        _rs_buttons = [
            dict(count=1,  label="1M", step="month", stepmode="backward"),
            dict(count=3,  label="3M", step="month", stepmode="backward"),
            dict(count=6,  label="6M", step="month", stepmode="backward"),
            dict(count=1,  label="1Y", step="year",  stepmode="backward"),
            dict(step="all", label="All"),
        ]
        _tickformatstops = [
            dict(dtickrange=[None,    86400000], value="%d %b"),
            dict(dtickrange=[86400000, "M1"],   value="%d %b '%y"),
            dict(dtickrange=["M1",    "M12"],   value="%b '%y"),
            dict(dtickrange=["M12",   None],    value="%Y"),
        ]
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d1117",
            xaxis=dict(
                showgrid=True, gridcolor="#1a2332", color="#5c7080",
                tickfont=dict(size=10, family="IBM Plex Mono"),
                tickformatstops=_tickformatstops,
                range=[x_start, x_end],
                rangeselector=dict(
                    buttons=_rs_buttons,
                    bgcolor="#0d1117", activecolor="#1a2332",
                    bordercolor="#314158", borderwidth=1,
                    font=dict(color="#8b949e", size=9, family="IBM Plex Mono"),
                    x=0, y=1.04,
                ),
                rangeslider=dict(visible=True, bgcolor="#080c10",
                                 bordercolor="#1a2332", thickness=0.05),
            ),
            yaxis=dict(showgrid=True, gridcolor="#1a2332", color="#5c7080",
                       tickfont=dict(size=10, family="IBM Plex Mono"), tickprefix="$",
                       range=[y_lo, y_hi]),
            hoverlabel=dict(
                bgcolor="#1a2332", bordercolor="#314158",
                font=dict(size=10, family="IBM Plex Sans", color="#cdd9e5"),
            ),
            margin=dict(t=30, b=10, l=10, r=90), height=400,
            showlegend=False, hovermode="closest",
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False},
                        key="price_chart")

        # Anomaly date selector — reliable alternative to clicking tiny dots
        if "combined_anomaly" in plot_df.columns:
            anom_dates = plot_df[plot_df["combined_anomaly"] == True]["Date"].dt.strftime("%Y-%m-%d").tolist()
            if anom_dates:
                options = ["— select anomaly date —"] + anom_dates
                cur_idx = 0
                if st.session_state.anomaly_date in anom_dates:
                    cur_idx = anom_dates.index(st.session_state.anomaly_date) + 1
                chosen = st.selectbox("Anomaly", options, index=cur_idx,
                                      label_visibility="collapsed", key="anomaly_select")
                if chosen != "— select anomaly date —":
                    st.session_state.anomaly_date = chosen
                else:
                    st.session_state.anomaly_date = None

        # Anomaly explanation box
        if st.session_state.anomaly_date:
            adate = pd.Timestamp(st.session_state.anomaly_date)
            arow_df = det_df[det_df["Date"].dt.date == adate.date()]
            if not arow_df.empty:
                arow = arow_df.iloc[0]
                explanation = explain_anomaly(arow, name, st.session_state.anomaly_date)
                if explanation:
                    st.markdown(explanation)
    else:
        st.markdown('<div style="height:300px;display:flex;align-items:center;justify-content:center;color:#5c7080;font-size:12px;font-family:IBM Plex Mono">No price data</div>', unsafe_allow_html=True)

    # RSI + RSI MA(14)
    st.markdown('<div class="chart-label">RSI  ·  RSI MA 14</div>', unsafe_allow_html=True)
    if det_df is not None and "rsi" in det_df.columns:
        rsi_df    = det_df.copy()
        rsi_last  = rsi_df["rsi"].iloc[-1]
        rsi_ma    = rsi_df["rsi"].rolling(window=14, min_periods=1).mean()
        rsima_last = rsi_ma.iloc[-1]

        fig2 = go.Figure()
        fig2.add_hline(y=70, line=dict(color="#f85149", width=0.6, dash="dot"))
        fig2.add_hline(y=30, line=dict(color="#3fb950", width=0.6, dash="dot"))
        fig2.add_hline(y=50, line=dict(color="#2d3748", width=0.5, dash="dot"))

        # RSI MA behind RSI
        fig2.add_trace(go.Scatter(
            x=rsi_df["Date"], y=rsi_ma.round(2),
            mode="lines", name="RSI MA14",
            line=dict(color="#58a6ff", width=1.2, dash="dot"),
            hovertemplate="RSI MA14: %{y:.1f}<extra></extra>",
        ))
        # RSI
        fig2.add_trace(go.Scatter(
            x=rsi_df["Date"], y=rsi_df["rsi"].round(2),
            mode="lines", name="RSI",
            line=dict(color="#e3b341", width=1.5),
            hovertemplate="RSI: %{y:.1f}<extra></extra>",
        ))

        # End-of-line labels — outside chart (xref=paper)
        gap = abs(rsi_last - rsima_last)
        rsi_yshift   = (+6 if rsi_last >= rsima_last else -6)   if gap < 5 else 0
        rsima_yshift = (-6 if rsi_last >= rsima_last else +6)   if gap < 5 else 0

        fig2.add_annotation(x=1.01, xref="paper", y=rsi_last, yref="y",
                            text=f"<b>RSI {rsi_last:.1f}</b>",
                            xanchor="left", showarrow=False,
                            font=dict(size=8, color="#e3b341", family="IBM Plex Mono"),
                            yshift=rsi_yshift)
        fig2.add_annotation(x=1.01, xref="paper", y=rsima_last, yref="y",
                            text=f"MA {rsima_last:.1f}",
                            xanchor="left", showarrow=False,
                            font=dict(size=8, color="#58a6ff", family="IBM Plex Mono"),
                            yshift=rsima_yshift)

        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d1117",
            xaxis=dict(showgrid=False, color="#5c7080", tickfont=dict(size=9, family="IBM Plex Mono")),
            yaxis=dict(showgrid=False, color="#5c7080", tickfont=dict(size=9, family="IBM Plex Mono"),
                       range=[0, 100]),
            margin=dict(t=5, b=5, l=10, r=70), height=130,
            showlegend=False, hovermode="x unified",
        )
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    # Visual analysis grid — full width below lower chart
    analysis_html = build_analysis(det_df, dec_row, ticker, name, st.session_state.language)
    if analysis_html:
        st.markdown(analysis_html, unsafe_allow_html=True)

    # Investor Summary — short signal overview above full report
    inv_summary_html = build_investor_summary(det_df, dec_row, row, news_df, ticker)
    if inv_summary_html:
        st.markdown(inv_summary_html, unsafe_allow_html=True)

    # LLM Analyst Report — full narrative text
    if news_df is not None and ticker in news_df["ticker"].values:
        llm_text = news_df[news_df["ticker"] == ticker].iloc[0].get("llm_summary", "")
        if llm_text and len(llm_text) > 120:
            lang = st.session_state.language
            with st.spinner("Translating..." if lang != "english" else ""):
                display_text = _translate(llm_text, lang, ticker)
            st.markdown(f"""
            <div style="background:#0d1117;border:1px solid #1a2332;border-radius:3px;
                        padding:14px 18px;margin-top:8px">
              <div style="font-size:9px;letter-spacing:2px;text-transform:uppercase;
                          color:#5c7080;font-family:'IBM Plex Mono',monospace;
                          border-bottom:1px solid #1a2332;padding-bottom:7px;
                          margin-bottom:10px">AI Analyst Report</div>
              <div style="font-size:11px;color:#adb5bd;line-height:1.8;
                          font-family:'IBM Plex Sans',sans-serif;white-space:pre-wrap;
                          {'direction:rtl;text-align:right' if lang == 'arabic' else ''}">{display_text}</div>
            </div>""", unsafe_allow_html=True)


