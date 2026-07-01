# FinWatch AI — Project State Document

> Last updated: 2026-06-20
> This file is the single source of truth for onboarding AI assistants.
> Show this file at the start of any new conversation to restore full context.

---

## What is this system?

**FinWatch AI** is a personal AI-powered stock monitoring and risk assessment system.

It does NOT predict stock prices. It detects anomalies and risk, then produces:
- Risk severity labels: `CRITICAL / WARNING / WATCH / NORMAL / POSITIVE_SIGNAL / REVIEW`
- Risk posture signals (UI display): `FAVORABLE / MONITOR / ELEVATED`
  - Internal codes (ENTRY/HOLD/EXIT/NEUTRAL) are mapped to display labels via `SIGNAL_DISPLAY` in `finwatch/data/loader.py`
  - The mapping is render-time only — no pipeline logic was changed
- Human-readable explanations (via Groq LLM + SHAP)

Run via: `python src/pipeline.py`
Dashboard via: `streamlit run finwatch/app.py` (from the `finwatch/` directory)

---

## Architecture — 8-Layer Pipeline

```
Layer 1  — Data Ingestion          src/ingestion/download_historical.py
Layer 2  — Data Quality            src/quality/quality_pipeline.py
Layer 3  — Feature Engineering     src/features/feature_pipeline.py
Layer 4  — Anomaly Detection       src/detection/detection_pipeline.py
Layer 5  — Prediction Models       src/prediction/prediction_pipeline.py
Layer 6  — Decision Engine         src/decision/decision_engine.py + decision_pipeline.py
Layer 7  — Explainability          src/explainability/explainability_pipeline.py
Layer 8  — Reporting + Dashboard   src/reporting/ + finwatch/app.py
```

Entry point: `src/pipeline.py`

---

## Detection Layer (Layer 4) — The Reliable Core

Four complementary anomaly detectors, combined into a **weighted anomaly score (0–1)**:

| Model | File | Weight | What it detects |
|---|---|---|---|
| LSTM Autoencoder | `src/detection/lstm_autoencoder.py` | 0.30 | Sequence anomalies (32 models: 16 sector groups × 2 vol regimes) |
| Isolation Forest | `src/detection/isolation_forest.py` | 0.30 | Multivariate outliers (16 models, one per sector group) |
| Return Z-Score | `src/detection/statistical.py` | 0.20 | Return distribution outliers (20d + 60d window) |
| Sector Z-Score | `src/detection/statistical.py` | 0.20 | Stock vs sector peers |

`anomaly_score_weighted` = weighted sum, 0–1 continuous.
`anomaly_score` = integer 0–4 (legacy, still used for backcompat).

Saved to: `data/detection/{ticker}.parquet`

---

## Prediction Layer (Layer 5)

### Drawdown Probability Model
- **File**: `src/prediction/models/drawdown_probability.py`
- **Type**: XGBoost + LightGBM (best model selected automatically at training time)
- **Target**: P(max drawdown > 5% in next 10 days)
- **Performance**: AUC 0.715 on holdout set (2024–2026, unseen during training)
- **Model file**: `models/xgboost_drawdown.pkl`

### Meta-Model (Stacking Layer)
- **File**: `src/prediction/models/meta_model.py`
- **Type**: Logistic Regression stacking
- **Purpose**: Combines `p_drawdown` + anomaly signals + VIX → refined `p_drawdown_meta`
- **Model file**: `models/meta_model.pkl`

### XGBoost Direction Model (NOT USED IN PRODUCTION)
- **File**: `src/prediction/models/xgboost_direction.py`
- **Status**: DISABLED — ~40% accuracy (barely above 33% random for 3 classes)
- **Reason**: EMH makes direction prediction near-impossible from technical data alone

### XGBoost Risk Model (DEPRECATED)
- **File**: `src/prediction/models/xgboost_risk.py`
- **Status**: Replaced by drawdown probability model. pkl kept for backward compat.

---

## Fundamental Data Collectors

All collectors save to `data/fundamental/` as parquet files.
**1-day lag**: collectors run after the decision pipeline, so the first-ever run has no fundamentals. Accepted design tradeoff.

| Collector | File | Output | What it collects |
|---|---|---|---|
| Earnings | `src/ingestion/earnings_collector.py` | `data/fundamental/earnings.parquet` | `days_to_next_earnings` |
| Insider | `src/ingestion/insider_collector.py` | `data/fundamental/insider.parquet` | `insider_sentiment` (-1 to +1) |
| Options | `src/ingestion/options_collector.py` | `data/fundamental/options.parquet` | `put_call_ratio`, `options_fear` |
| Valuation | `src/ingestion/valuation_collector.py` | `data/fundamental/valuation.parquet` | `pe_ratio`, `pe_forward`, `pb_ratio`, `revenue_growth` |
| Sentiment | `src/ingestion/sentiment_collector.py` | `data/fundamental/sentiment.parquet` | historical news for future training |

---

## Decision Engine (Layer 6)

### Files
- `src/decision/decision_engine.py` — core logic
- `src/decision/decision_pipeline.py` — loads all data sources, calls engine, saves output

### AnomalyInput dataclass fields
```python
# Core
ticker, date, p_drawdown, drawdown_risk, anomaly_score, anomaly_score_weighted
market_anomaly, sector_anomaly
# Technical
rsi, momentum_5, momentum_10, drawdown, obv_signal, volatility, excess_return, es_ratio, vix_level
# MA / Regime context
price_vs_ma200, price_vs_ma50   # stock-specific (not SPX MAs)
regime                           # "bull" / "bear" / "transition_down" / "transition_up" / "unknown"
volume_trend, trend_strength
# Sentiment
vader_score, finbert_score, news_sentiment_score
# Fundamental
days_to_next_earnings, insider_sentiment, put_call_ratio, options_fear
# Valuation
pe_ratio, pe_forward, pb_ratio, revenue_growth
```

### DecisionOutput dataclass fields
```python
ticker, date, severity, action, confidence, context
p_drawdown, anomaly_score, anomaly_score_weighted, drawdown_risk
momentum_signal, caution_flag, override_reason, summary, sentiment_note
trading_signal   # ENTRY / HOLD / EXIT / NEUTRAL  (internal code; UI maps to FAVORABLE/MONITOR/ELEVATED)
anomaly_type
```

### Severity logic (priority-ordered)
1. `CRITICAL` — p_drawdown ≥ p_crit AND (anomaly_w ≥ 0.30 AND bearish confirmation)
2. `CRITICAL` — actual 30d drawdown ≤ −15%
3. `WARNING` — p_drawdown ≥ p_warn
4. `WARNING` — actual drawdown ≤ −8%
5. `WATCH` — anomaly_w ≥ 0.20 OR moderate p_drawdown
6. `POSITIVE_SIGNAL` — p_dd < 30% AND RSI < 70 AND positive momentum AND anomaly_w < 0.20
7. `NORMAL` — none of the above
8. `REVIEW` — conflicting: high anomaly but low p_drawdown and stock outperforming

VIX-aware: thresholds raised at low VIX (< 20) to reduce false positives in calm markets.
Regime-aware: bear market upgrades WATCH→WARNING; blocks ENTRY signals.

### Risk signal (display) logic
For WARNING severity — momentum-recovery aware:
- Strong ML signal (`p_dd ≥ 0.50` or `anomaly_w ≥ 0.35`) + **no recovery** → **ELEVATED**
- Strong ML signal + **recovering** (`momentum_5 > 0.03` or RSI bullish divergence) → **MONITOR**
- Weak ML signal → **MONITOR** regardless of momentum

Valuation gates: blocks FAVORABLE on negative or extreme P/E (> 50); strengthens on cheap fundamentals.

### Output
Saved to: `data/decisions/decisions.parquet`

---

## Backtesting

### Main Backtest
- **File**: `src/backtesting/backtest.py`
- Walk-forward: 4-year rolling train, 6-month test, 11 windows, no lookahead
- **Results**: 683 decisions

| Signal | n | Avg 20d Return | Drawdown Rate |
|---|---|---|---|
| FAVORABLE | 2 | +2.96% | 0.0% |
| MONITOR | 190 | +2.56% | 32.1% |
| NEUTRAL | 62 | +3.49% | 21.0% |
| ELEVATED | 429 | +4.90% | 44.3% |

Overall (CRITICAL + WARNING vs. no-event): **Precision 41.4%, Recall 92.4%, F1 0.571**
System is tuned for high recall — missed drawdowns cost more than false alarms.

Outputs:
- `data/backtesting/backtest_results.parquet` — 683 rows
- `data/backtesting/signal_precision.parquet`
- `data/backtesting/anomaly_precision.parquet`
- `data/backtesting/summary.txt`

### Baseline Comparison
- **File**: `src/backtesting/baseline_comparison.py` (added 2026-06-20)
- Tests 5 naive heuristics against the same 683 decisions and same target

| Method | AUC | F1 |
|---|---|---|
| LogReg (vol + momentum) | 0.621 | 0.496 |
| Volatility (60d realized) | 0.604 | 0.489 |
| **FinWatch (full ML stack)** | **0.597** | **0.571** |
| Momentum (neg 5d + MA50) | 0.540 | 0.432 |
| RSI (overbought proxy) | 0.525 | 0.420 |

**Honest read**: FinWatch trails naive baselines on pure AUC ranking (0.597 vs 0.621) but wins on F1 (+0.075) because it operates at 86% flag rate for 92.4% recall. The ML edge is recall + signal composition, not ranking.

Outputs:
- `data/backtesting/baseline_comparison.parquet`
- `data/backtesting/baseline_comparison_summary.txt`

---

## Dashboard — 4-Screen Streamlit App

**Entry point**: `finwatch/app.py`
**Theme**: Dark, custom CSS, EN/DE language toggle (`finwatch/ui/i18n.py`)
**Colors**: Centralized in `finwatch/ui/theme.py` — `TEXT_MUTED`, `CHART_AXIS`, `COLOR_NEUTRAL` are single source of truth; all files import from there
**Run from**: `finwatch/` directory (`streamlit run app.py`)

### Screens

| Screen | File | What it shows |
|---|---|---|
| Command Center | `finwatch/ui/command_center.py` | Regime banner, KPI strip, alert list, sector severity bar |
| Stock Deep-Dive | `finwatch/ui/deep_dive.py` | Finnhub news, price chart, SHAP, radar, LLM narrative, AI chat |
| AI Analyst | `finwatch/ui/ai_analyst.py` | Standalone AI Analyst chat page |
| My Portfolio | `finwatch/ui/portfolio_page.py` | Positions, P&L, risk signals |

### Signal display mapping (render-time only, no pipeline change)
```python
# finwatch/data/loader.py
SIGNAL_DISPLAY = {
    "ENTRY":      "FAVORABLE",
    "BUY_SIGNAL": "FAVORABLE",
    "HOLD":       "MONITOR",
    "NEUTRAL":    "NEUTRAL",
    "WATCH":      "MONITOR",
    "EXIT":       "ELEVATED",
    "REDUCE":     "ELEVATED",
}
```

### AI Analyst
- **SDK**: Groq Python SDK, direct tool-call loop (NOT LlamaIndex)
- **Model**: `llama-3.3-70b-versatile`
- **Tools**: 11 tools in `src/agent/tools.py`
- **Loop**: Synchronous, up to 8 iterations, shows tool-call chips in UI
- **Init error handling**: `_ensure_agent()` in `chat_panel.py` logs full errors to Python logger, shows only a clean UI message (no API keys leaked)

### Finnhub news (per-stock)
- Added to Deep-Dive: `_render_stock_news()` calls `fetch_stock_news(ticker, limit=1)` from `finwatch/data/loader.py`
- Cached 30 min (`@st.cache_data(ttl=1800)`)
- Shown just below the stock header, above the anomaly profile

---

## Data Flow

```
data/raw/{ticker}.parquet           ← download_historical
data/features/{ticker}.parquet      ← feature_pipeline
data/detection/{ticker}.parquet     ← detection_pipeline
data/decisions/decisions.parquet    ← decision_pipeline
data/fundamental/                   ← collectors (1-day lag)
data/backtesting/                   ← backtest + baseline_comparison
data/explanations/                  ← explainability_pipeline (SHAP + LLM)
```

---

## Models Directory

```
models/xgboost_drawdown.pkl         ← Drawdown Probability (ACTIVE)
models/meta_model.pkl               ← Meta-Model stacking (ACTIVE)
models/ae_{sector}_{vol}.keras      ← LSTM Autoencoders per bucket (ACTIVE)
models/if_{sector}.pkl              ← Isolation Forests per sector (ACTIVE)
models/xgboost_direction.pkl        ← Direction model (NOT USED)
models/xgboost_risk.pkl             ← Old risk model (DEPRECATED)
models/risk_label_encoder.pkl       ← For old risk model (DEPRECATED)
```

---

## Environment / Config

- **API keys**: `FINNHUB_API_KEY`, `GROQ_API_KEY`, `TWELVE_DATA_API_KEY` — in `.env` at project root
- **Config**: `config/assets.yaml` — ticker list + sector assignments
- **Universe**: 58 stocks, 13 sectors, 13 sector ETF groups
- **Requirements**: `requirements.txt`

Key pinned versions: `numpy==1.26.4`, `pandas==2.2.1`, `tensorflow-macos==2.16.2`, `shap==0.49.1`

---

## Known Issues / Technical Debt

1. **Fundamentals 1-day lag**: Collectors run after the decision pipeline. First-ever run has no fundamentals. Accepted design tradeoff.

2. **XGBoost direction model trained but not used**: Exists at `models/xgboost_direction.pkl`, not called anywhere in production. Research only.

3. **Groq daily token limit (100k TPD)**: LLM narrations (Layer 7) degrade to template fallbacks when daily quota exhausted. Resets daily.

4. **`market_news_title` i18n key unused**: Key exists in `finwatch/ui/i18n.py` (EN + DE) but the market news box was removed from Command Center. Key is harmless but stale.

5. **AUC on walk-forward (0.597) vs. model holdout (0.715)**: The two numbers measure different things. 0.715 = drawdown model AUC on its training holdout. 0.597 = AUC of p_drawdown scores on the 683 walk-forward production decisions (different population, aggressive high-recall flag rate). Both are correct; document both contexts.

---

## What Was Done — Recent Sessions (2026-06)

### UI Redesign (branch: feature/ui-redesign)
1. **Full dark-theme dashboard redesign** — 4 screens (Command Center, Deep-Dive, AI Analyst, My Portfolio)
2. **Signal relabeling** — internal ENTRY/HOLD/EXIT codes mapped to FAVORABLE/MONITOR/ELEVATED at render time via `SIGNAL_DISPLAY` dict; no pipeline changes
3. **WCAG contrast fixes** — all secondary/muted text raised to ≥ 4.5:1 contrast; colors centralized in `theme.py` (`TEXT_MUTED=#8080a4`, `CHART_AXIS=#7a9ab0`, `COLOR_NEUTRAL=#7a95ab`)
4. **Per-stock Finnhub news** — `fetch_stock_news()` + `rel_time()` added to `finwatch/data/loader.py`; `_render_stock_news()` added to Deep-Dive below stock header
5. **Sidebar stock selector fix** — removed custom `.sb-stock-label` div, switched to native selectbox `label_visibility="visible"` with CSS override; fixes label clipping
6. **Chat panel init security** — `_ensure_agent()` now logs full error to Python logger, stores only clean error codes (`"missing_key"` / `"init_failed"`) in session state, never exposes API key content to UI
7. **Market news box removed** — was added then removed (off-topic content from Finnhub `category=general`); Command Center right column now shows only Sector Severity bar
8. **EN/DE i18n** — 4 new keys added: `stock_news_title`, `news_no_recent`, `select_stock_label`, `market_news_title` (last one unused but present)

### README Updates (2026-06-19 / 2026-06-20)
- Real backtest numbers (683 decisions, 11 windows)
- NEUTRAL signal row added to backtest table
- Overall detection metrics (Precision 41.4%, Recall 92.4%, F1 0.571)
- ETF count corrected: 11 → 13
- Screenshots placed under correct section headers
- "trading signal" → "risk signal" throughout; external analyst data clearly labelled
- Key Design Decisions moved above Backtesting Results
- Baseline comparison subsection added (honest AUC comparison vs. heuristics)

### Baseline Comparison Script (2026-06-20)
- **New file**: `src/backtesting/baseline_comparison.py`
- Tests Random, Volatility, Momentum, RSI, LogReg baselines against same 683 decisions
- Computes AUC + Precision/Recall/F1 per method, prints comparison table, reports AUC/F1 edge
- Outputs: `data/backtesting/baseline_comparison.parquet` + `baseline_comparison_summary.txt`

---

## What Still Needs To Be Done

- [ ] Create `lstm_inference.py` for standalone LSTM-AE inference / live demo
- [ ] Fix dashboard to show when valuation data was last updated (staleness warning)
- [ ] Rerun baseline comparison after next model retrain to track edge over time
- [ ] Consider `AVOID` signal for stocks with negative P/E (never enter under any condition)
