"""
FinWatch AI — Agent (Groq SDK, plain tool loop)

Uses the Groq Python SDK directly instead of LlamaIndex FunctionAgent.
This avoids the parallel-tool-call parsing bug where llama-3.3-70b-versatile
merges the tool name and JSON arguments into a single string.
"""

from __future__ import annotations
import json
import logging
import os
from typing import Any, Callable
import groq

_log = logging.getLogger(__name__)


def _classify_error(exc: Exception) -> str:
    """Map any caught exception to a clean user-facing message; log the full detail."""
    _log.error("AI Analyst error (%s): %s", type(exc).__name__, exc, exc_info=True)
    msg = str(exc).lower()
    if isinstance(exc, groq.RateLimitError) or "rate_limit" in msg or "429" in msg:
        return (
            "⚠️ The AI Analyst has reached its daily request limit. "
            "Please try again later. (Data and dashboards remain fully available.)"
        )
    if (
        isinstance(exc, (groq.APIConnectionError, groq.APITimeoutError))
        or "connection" in msg
        or "timeout" in msg
        or "network" in msg
    ):
        return "⚠️ The AI Analyst is temporarily unavailable. Please try again in a moment."
    return "⚠️ Something went wrong while generating the answer. Please rephrase or try again."

from src.agent.tools import (
    get_stock_analysis,
    get_risk_metrics,
    explain_anomaly,
    get_market_context,
    get_news_sentiment,
    get_trend_analysis,
    get_portfolio_overview,
    get_macro_context,
    get_earnings_calendar,
    get_correlation_risk,
    get_sector_analysis,

)

_MODEL = "llama-3.3-70b-versatile"
_MAX_TURNS = 8       # tool-call iterations before giving up
_MAX_TOKENS = 1024   # enough for a thorough answer; keeps daily usage sane

_SYSTEM_PROMPT = """\
You are FinWatch AI, a financial risk analyst assistant.
Use the provided tools to answer questions about stocks, sectors, and portfolio risk.
Always call the relevant tool(s) first — never invent numbers.
Be concise and precise. Cite the concrete values the tools return.
If a tool returns an error, say so honestly.

Signal terminology: the tools return internal codes (ENTRY/HOLD/EXIT).
When reporting these to the user, translate them as follows:
  ENTRY → FAVORABLE (lower-risk profile, models see positive conditions)
  HOLD  → MONITOR   (mixed or neutral signals, no strong direction)
  EXIT  → ELEVATED  (elevated risk, models flag significant downside probability)
  NEUTRAL → NEUTRAL, WATCH → MONITOR, REDUCE/BUY_SIGNAL map to ELEVATED/FAVORABLE.

Regulatory constraint: describe risk posture only — never recommend buying,
selling, or holding any asset. You describe what the models see, not what
the user should do.
"""

_LANG_INSTRUCTIONS: dict[str, str] = {
    "en": (
        "Respond in English only. "
        "Keep signal labels FAVORABLE, MONITOR, ELEVATED in uppercase as-is."
    ),
    "de": (
        "Antworte ausschließlich auf Deutsch. "
        "Behalte die Signalbegriffe FAVORABLE, MONITOR, ELEVATED in Großbuchstaben unverändert. "
        "Alle anderen Texte, Erklärungen und Werte müssen auf Deutsch sein."
    ),
}

_TOOL_MAP: dict[str, Callable[..., Any]] = {
    "get_stock_analysis":    get_stock_analysis,
    "get_risk_metrics":      get_risk_metrics,
    "explain_anomaly":       explain_anomaly,
    "get_market_context":    get_market_context,
    "get_news_sentiment":    get_news_sentiment,
    "get_trend_analysis":    get_trend_analysis,
    "get_portfolio_overview": get_portfolio_overview,
    "get_macro_context":     get_macro_context,      
    "get_earnings_calendar": get_earnings_calendar,  
    "get_correlation_risk":  get_correlation_risk,
    "get_sector_analysis":   get_sector_analysis, 
}

# Concise schemas — short descriptions reduce prompt tokens on every call
_TOOLS_SCHEMA: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_analysis",
            "description": "Overall assessment for a stock: severity, trading signal, drawdown probability, anomaly type, confidence.",
            "parameters": {
                "type": "object",
                "properties": {"ticker": {"type": "string", "description": "e.g. 'AAPL'"}},
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_risk_metrics",
            "description": "Risk metrics: VaR 95%, Expected Shortfall 95%, ES ratio, 30-day max drawdown.",
            "parameters": {
                "type": "object",
                "properties": {"ticker": {"type": "string"}},
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "explain_anomaly",
            "description": "Why a stock is anomalous: SHAP drivers, anomaly group scores (volume/volatility/price/context), LLM narrative.",
            "parameters": {
                "type": "object",
                "properties": {"ticker": {"type": "string"}},
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_market_context",
            "description": "Market context for a stock: regime (bull/bear/neutral), volatility regime, market-wide anomaly flag.",
            "parameters": {
                "type": "object",
                "properties": {"ticker": {"type": "string"}},
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_news_sentiment",
            "description": "News sentiment: VADER score, FinBERT score, LLM news summary, top headlines.",
            "parameters": {
                "type": "object",
                "properties": {"ticker": {"type": "string"}},
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_trend_analysis",
            "description": "Trend indicators: MA50, MA200, 5d/10d momentum, RSI, trend strength, volume trend.",
            "parameters": {
                "type": "object",
                "properties": {"ticker": {"type": "string"}},
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_portfolio_overview",
            "description": "Portfolio-wide summary: total stocks monitored, severity distribution, signal distribution.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_sector_analysis",
            "description": "Sector breakdown: severity counts and average drawdown probability per sector.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


class FinWatchAgent:
    """Thin wrapper around the Groq SDK. Runs a synchronous tool-call loop."""

    def __init__(self) -> None:
        self._client = groq.Groq(api_key=os.environ["GROQ_API_KEY"])

    def run(self, prompt: str, lang: str = "en") -> tuple[str, list[str]]:
        """
        Run one user turn through the tool loop.

        Returns:
            answer      — the model's final text response
            tool_names  — deduplicated list of tools that were called
        """
        lang_note = _LANG_INSTRUCTIONS.get(lang, _LANG_INSTRUCTIONS["en"])
        system_content = f"{_SYSTEM_PROMPT}\nLanguage: {lang_note}"
        messages: list[dict] = [
            {"role": "system", "content": system_content},
            {"role": "user",   "content": prompt},
        ]
        tool_names: list[str] = []
        seen: set[str] = set()

        for _ in range(_MAX_TURNS):
            try:
                resp = self._client.chat.completions.create(
                    model=_MODEL,
                    messages=messages,
                    tools=_TOOLS_SCHEMA,
                    tool_choice="auto",
                    max_tokens=_MAX_TOKENS,
                    temperature=0.1,
                )
            except Exception as exc:
                return _classify_error(exc), tool_names
            msg = resp.choices[0].message

            # No tool calls → final answer
            if not msg.tool_calls:
                return (msg.content or "").strip(), tool_names

            # Collect tool names for the UI chips
            for tc in msg.tool_calls:
                name = tc.function.name
                if name not in seen:
                    seen.add(name)
                    tool_names.append(name)

            # Append assistant message (with tool_calls attached)
            messages.append(msg)

            # Execute each tool and append results
            for tc in msg.tool_calls:
                fn = _TOOL_MAP.get(tc.function.name)
                if fn is None:
                    result: object = {"error": f"Unknown tool: {tc.function.name}"}
                else:
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                        result = fn(**args)
                    except Exception as exc:
                        result = {"error": str(exc)}

                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      json.dumps(result, default=str),
                })

        return "Maximum tool iterations reached.", tool_names


def build_agent() -> FinWatchAgent:
    return FinWatchAgent()


if __name__ == "__main__":
    agent = build_agent()
    answer, tools = agent.run("Give me a full risk breakdown for DELL.")
    print("Tools called:", tools)
    print("\n--- ANSWER ---")
    print(answer)
