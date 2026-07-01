"""FinWatch UI — internationalisation helper.

Usage:
    from ui.i18n import t
    label = t("metric_conf")                            # "Confidence" or "Konfidenz"
    msg   = t("radar_main_driver", phrase="elevated volatility")
"""
from __future__ import annotations

_STRINGS: dict[str, dict[str, str]] = {
    "en": {
        # ── Price chart ────────────────────────────────────────────────────────
        "price":             "Price",
        "anomaly":           "Anomaly",
        "anomaly_confirmed": "Anomaly confirmed",
        # ── Market regime ──────────────────────────────────────────────────────
        "regime_sideways":   "Sideways",
        # ── Drawdown tooltip ───────────────────────────────────────────────────
        "dd_tip": (
            "Probability of a significant drawdown in the next 10 days, "
            "as predicted by the XGBoost model (0 = low risk, 1 = high risk)"
        ),
        # ── Radar explanation block ────────────────────────────────────────────
        "radar_key": (
            "Larger diamond = more unusual behavior = higher warning signal. "
            "Small & centered = calm / normal."
        ),
        "radar_axis_volatility_label":  "Volatility",
        "radar_axis_volatility_desc":   "how much the price fluctuates unusually",
        "radar_axis_volatility_phrase": "volatility",
        "radar_axis_volume_label":      "Volume",
        "radar_axis_volume_desc":       "whether unusually high/low volume is traded",
        "radar_axis_volume_phrase":     "trading volume",
        "radar_axis_price_label":       "Price",
        "radar_axis_price_desc":        "whether the price movement is a statistical outlier",
        "radar_axis_price_phrase":      "price movement",
        "radar_axis_context_label":     "Context",
        "radar_axis_context_desc":      "how much the stock deviates from its sector",
        "radar_axis_context_phrase":    "sector deviation",
        "radar_all_normal":  "All indicators normal — no anomalous behavior detected.",
        "radar_main_driver": "Currently, {phrase} stands out most.",
        "radar_broad":       "Broad, moderate anomaly — no single dramatic outlier.",
        "radar_elevated":    "Elevated {phrase}, other indicators are moderate.",
        "radar_slight":      "Slight anomaly in multiple areas, overall calm.",
        # ── Metric cards ──────────────────────────────────────────────────────
        "metric_conf":      "Confidence",
        "metric_var_help":  "Value at Risk — maximum expected loss in 95% of cases (1 day)",
        "metric_es_help":   "Expected Shortfall — average loss in the worst 5% of cases",
        "metric_dd_help":   "Drawdown probability — model score for significant price decline (next 10 days)",
        "metric_conf_help": "Confidence — model certainty; low value = mixed signals",
        "metric_var_sub":   "Bad day (worst 5%): about {value}",
        "metric_es_sub":    "When bad, avg. loss is {value}",
        "metric_dd_sub":    "Chance of further price drop",
        "metric_conf_sub":  "How certain the model is",
        "corr_sub_low":     "Isolated — barely affects your portfolio",
        "corr_sub_medium":  "Some overlap — monitor linked stocks",
        "corr_sub_high":    "Clusters with others — amplifies portfolio risk",
        # ── Outlook section ────────────────────────────────────────────────────
        "outlook_forecast":      "Forecast · next 10 days",
        "outlook_confidence":    "Confidence",
        "outlook_analysts":      "Analysts",
        "outlook_analysts_meta": "{total} Analysts · {period}",
        "outlook_bullish":       "Bullish",
        "outlook_no_data":       "No data available",
        "outlook_run_hint":      "run python src/data/analyst_ratings.py",
        "outlook_no_coverage":   "No analyst coverage — this ticker is not tracked by Finnhub's recommendation database.",
        "outlook_price_target":  "Price Target",
        # ── Detection models section ───────────────────────────────────────────
        "det_section_label":     "WHAT TRIGGERED THE ALERT",
        "det_model_pattern":     "Pattern Memory",
        "det_model_outlier":     "Outlier Detector",
        "det_model_spike":       "Spike Detector",
        "det_tip_pattern":       "Compares today's price and volume pattern against years of history. Fires when the recent sequence looks nothing like what it has seen before.",
        "det_tip_outlier":       "Checks many indicators at once (volatility, volume, momentum, …). Fires when several of them are simultaneously unusual — even if each one alone looks borderline.",
        "det_tip_spike":         "Measures how far today's price moved compared to the past 60 trading days. Fires when the move is statistically extreme — like a 2× or 3× larger day than usual.",
        "det_agreement_3":       "All three detection methods agree — strong signal.",
        "det_agreement_2":       "Two out of three detection methods agree.",
        "det_agreement_1":       "Only one method flagged this — treat as early warning.",
        "det_agreement_0":       "No detection method flagged this stock.",
        # ── Chat panel ────────────────────────────────────────────────────────
        "chat_hint_why":            "Why is this stock flagged?",
        "chat_hint_risk":           "What are the key risk metrics?",
        "chat_hint_analysts":       "What do analysts say about this?",
        "chat_hint_trend":          "Show me the trend indicators.",
        "chat_hint_portfolio_risk": "What are the biggest risks right now?",
        "chat_hint_sector":         "Which sector is most at risk?",
        "chat_hint_overview":       "Give me a portfolio overview.",
        "chat_hint_entry":          "Which stocks show FAVORABLE signals?",
        "chat_ask_stock":           "Ask about {ticker}…",
        "chat_ask_portfolio":       "Ask FinWatch AI Analyst…",
        "chat_about":               "Ask about {subject}:",
        "portfolio_label":          "the portfolio",
        # ── Page titles ───────────────────────────────────────────────────────
        "page_cc_title":  "Command Center",
        "page_cc_sub":    "PORTFOLIO OVERVIEW · ANOMALY ALERTS · SECTOR RISK",
        "page_dd_title":  "Stock Deep-Dive",
        "page_dd_sub":    "PRICE · ANOMALY · RISK · AI ANALYST",
        "page_pf_title":  "My Portfolio",
        "page_pf_sub":    "POSITIONS · P&L · RISK SIGNALS",
        # ── Badge labels ──────────────────────────────────────────────────────
        "badge_severity": "SEVERITY",
        "badge_signal":   "SIGNAL",
        # ── News boxes ────────────────────────────────────────────────────────
        "market_news_title": "Market News",
        "stock_news_title":  "Latest News",
        "news_no_recent":    "No recent news",
        # ── Sidebar ───────────────────────────────────────────────────────────
        "select_stock_label": "Select Stock",
    },
    "de": {
        # ── Price chart ────────────────────────────────────────────────────────
        "price":             "Kurs",
        "anomaly":           "Anomalie",
        "anomaly_confirmed": "Anomalie bestätigt",
        # ── Market regime ──────────────────────────────────────────────────────
        "regime_sideways":   "Seitwärts",
        # ── Drawdown tooltip ───────────────────────────────────────────────────
        "dd_tip": (
            "Wahrscheinlichkeit eines signifikanten Drawdowns in den nächsten 10 Tagen, "
            "gemäß XGBoost-Modell (0 = geringes Risiko, 1 = hohes Risiko)"
        ),
        # ── Radar explanation block ────────────────────────────────────────────
        "radar_key": (
            "Größerer Diamant = ungewöhnlicheres Verhalten = höheres Warnsignal. "
            "Klein & mittig = ruhig/normal."
        ),
        "radar_axis_volatility_label":  "Volatilität",
        "radar_axis_volatility_desc":   "wie stark der Kurs ungewöhnlich schwankt",
        "radar_axis_volatility_phrase": "Volatilität",
        "radar_axis_volume_label":      "Volumen",
        "radar_axis_volume_desc":       "ob ungewöhnlich viel/wenig gehandelt wird",
        "radar_axis_volume_phrase":     "Handelsvolumen",
        "radar_axis_price_label":       "Preis",
        "radar_axis_price_desc":        "ob die Preisbewegung ein statistischer Ausreißer ist",
        "radar_axis_price_phrase":      "Preisbewegung",
        "radar_axis_context_label":     "Kontext",
        "radar_axis_context_desc":      "wie stark die Aktie von ihrem Sektor abweicht",
        "radar_axis_context_phrase":    "Sektorabweichung",
        "radar_all_normal":  "Alle Indikatoren unauffällig — aktuell kein anomales Verhalten erkennbar.",
        "radar_main_driver": "Aktuell sticht vor allem {phrase} hervor.",
        "radar_broad":       "Breite, moderate Auffälligkeit — kein einzelner dramatischer Ausreißer.",
        "radar_elevated":    "Erhöhter Wert bei {phrase}, die anderen Indikatoren sind moderat.",
        "radar_slight":      "Leichte Auffälligkeit in mehreren Bereichen, insgesamt ruhig.",
        # ── Metric cards ──────────────────────────────────────────────────────
        "metric_conf":      "Konfidenz",
        "metric_var_help":  "Value at Risk — maximaler erwarteter Verlust in 95% der Fälle (1 Tag)",
        "metric_es_help":   "Expected Shortfall — durchschnittlicher Verlust in den schlimmsten 5% der Fälle",
        "metric_dd_help":   "Drawdown-Wahrscheinlichkeit — Modell-Score für signifikanten Preiseinbruch (nächste 10 Tage)",
        "metric_conf_help": "Konfidenz — Modellunsicherheit; niedriger Wert = gemischte Signale",
        "metric_var_sub":   "Schlechter Tag (5%-Worst-Case): ca. {value}",
        "metric_es_sub":    "Im Krisenfall: ø Verlust {value}",
        "metric_dd_sub":    "Wahrsch. weiterer Kursverlust",
        "metric_conf_sub":  "Sicherheit des Modells",
        "corr_sub_low":     "Einzelfall — kaum Portfolioauswirkung",
        "corr_sub_medium":  "Überschneidung — vernetzte Aktien beachten",
        "corr_sub_high":    "Stark verknüpft — Risikoverstärkung",
        # ── Outlook section ────────────────────────────────────────────────────
        "outlook_forecast":      "Prognose · nächste 10 Tage",
        "outlook_confidence":    "Konfidenz",
        "outlook_analysts":      "Analysten",
        "outlook_analysts_meta": "{total} Analysten · {period}",
        "outlook_bullish":       "Bullish",
        "outlook_no_data":       "Keine Daten verfügbar",
        "outlook_run_hint":      "python src/data/analyst_ratings.py ausführen",
        "outlook_no_coverage":   "Kein Analyst-Coverage — dieser Ticker ist nicht in der Finnhub-Datenbank erfasst.",
        "outlook_price_target":  "Kursziel",
        # ── Detection models section ───────────────────────────────────────────
        "det_section_label":     "WAS HAT DEN ALERT AUSGELÖST",
        "det_model_pattern":     "Mustererkennung",
        "det_model_outlier":     "Ausreißer-Detektor",
        "det_model_spike":       "Sprung-Detektor",
        "det_tip_pattern":       "Vergleicht das aktuelle Kurs- und Volumenmuster mit Jahren historischer Daten. Schlägt an wenn die letzten Tage so aussehen, wie das Modell es noch nie gesehen hat.",
        "det_tip_outlier":       "Prüft viele Kennzahlen gleichzeitig (Volatilität, Volumen, Momentum, …). Schlägt an wenn mehrere davon gleichzeitig ungewöhnlich sind — auch wenn jede einzelne nur grenzwertig wäre.",
        "det_tip_spike":         "Misst wie stark der heutige Kursbewegung im Vergleich zu den letzten 60 Handelstagen war. Schlägt an wenn die Bewegung statistisch extrem war — z.B. 2× oder 3× größer als ein normaler Tag.",
        "det_agreement_3":       "Alle drei Methoden sind sich einig — starkes Signal.",
        "det_agreement_2":       "Zwei von drei Methoden haben angeschlagen.",
        "det_agreement_1":       "Nur eine Methode hat angeschlagen — als Frühwarnung werten.",
        "det_agreement_0":       "Keine Methode hat diese Aktie markiert.",
        # ── Chat panel ────────────────────────────────────────────────────────
        "chat_hint_why":            "Warum ist diese Aktie auffällig?",
        "chat_hint_risk":           "Was sind die wichtigsten Risikokennzahlen?",
        "chat_hint_analysts":       "Was sagen Analysten dazu?",
        "chat_hint_trend":          "Zeig mir die Trendindikatoren.",
        "chat_hint_portfolio_risk": "Was sind die größten Risiken gerade?",
        "chat_hint_sector":         "Welcher Sektor ist am stärksten gefährdet?",
        "chat_hint_overview":       "Gib mir einen Portfolioüberblick.",
        "chat_hint_entry":          "Welche Aktien zeigen FAVORABLE-Signale?",
        "chat_ask_stock":           "Frage zu {ticker}…",
        "chat_ask_portfolio":       "FinWatch AI Analyst fragen…",
        "chat_about":               "Frage zu {subject}:",
        "portfolio_label":          "dem Portfolio",
        # ── Page titles ───────────────────────────────────────────────────────
        "page_cc_title":  "Command Center",
        "page_cc_sub":    "PORTFOLIO-ÜBERBLICK · ANOMALIE-ALERTS · SEKTORRISIKO",
        "page_dd_title":  "Aktien-Analyse",
        "page_dd_sub":    "KURS · ANOMALIE · RISIKO · KI-ANALYST",
        "page_pf_title":  "Mein Portfolio",
        "page_pf_sub":    "POSITIONEN · P&L · RISIKOSIGNALE",
        # ── Badge labels ──────────────────────────────────────────────────────
        "badge_severity": "SCHWERE",
        "badge_signal":   "SIGNAL",
        # ── News boxes ────────────────────────────────────────────────────────
        "market_news_title": "Marktnachrichten",
        "stock_news_title":  "Aktuelle Nachrichten",
        "news_no_recent":    "Keine aktuellen Nachrichten",
        # ── Sidebar ───────────────────────────────────────────────────────────
        "select_stock_label": "Aktie wählen",
    },
}


def t(key: str, **kwargs: object) -> str:
    """Return the UI string for the active language (reads st.session_state.lang)."""
    try:
        import streamlit as st
        lang = st.session_state.get("lang", "en") or "en"
    except Exception:
        lang = "en"
    s = _STRINGS.get(lang, _STRINGS["en"]).get(key) or _STRINGS["en"].get(key, key)
    return s.format(**kwargs) if kwargs else s
