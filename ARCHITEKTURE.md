# FinWatch AI
––––––––––––––

Produktname:     FinWatch AI
GitHub Repo:     finwatch-ai
Beschreibung:    AI-Driven Financial Anomaly Detection
                 and Monitoring System



# LAYER 1 – DATA INGESTION
├── Stooq API (historical, 10 years, daily)
├── 11 Stocks, OHLCV
└── Output: raw_clean parquet files

# LAYER 2 – DATA QUALITY CHECKS
├── Missing values
├── Duplicates
├── OHLC violations
└── Output: validated data

# LAYER 3 – FEATURE ENGINEERING
├── Daily Returns
├── Volatility
├── Rolling Stats (mean, std)
├── Beta
├── Correlation
└── Output: feature parquet files

# LAYER 4 – ANOMALY DETECTION
├── Statistical:    Z-Score (threshold ±3σ)
├── ML:             Isolation Forest (contamination=0.05)
├── Deep Learning:  Autoencoder (reconstruction error)
├── Combine:        anomaly_score (0–3), combined_anomaly
├── Severity:       MINOR / WARNING / CRITICAL  (script within detection)
└── Output: anomaly flags + severity label per model

# LAYER 5 – CONTEXTUAL VALIDATION
├── Compare stock vs S&P 500
├── Compare stock vs VIX
├── News Sentiment (FinBERT)
└── Output: context enriched anomaly

# LAYER 6 – PREDICTION CORRIDOR
├── LSTM (last 30 days OHLCV) → predicted_close
├── prediction_error = |actual - predicted| / predicted
├── XGBoost Classifier:
│   ├── Input: anomaly_score + severity + z_score + prediction_error
│   ├── P(up)     %
│   ├── P(stable) %
│   └── P(down)   %
└── Output: predicted price + direction probabilities

# LAYER 7 – GUARDRAILS + AI AGENT
├── FIX:       auto-correct data issues
├── KEEP:      flag but keep
├── ESCALATE:  alert human
└── Output: autonomous decision

# LAYER 8 – REPORTING + XAI
├── Automatic report per ticker
├── Explanation WHY anomaly
├── News context
└── Output: PDF/HTML report

# LAYER 9 – LOGGING & AUDIT
├── All decisions logged
├── Timestamp + model + reason
└── Output: audit trail
