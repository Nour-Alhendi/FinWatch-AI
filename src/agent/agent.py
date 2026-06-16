# FinWatch AI — LlamaIndex Agent
# Registers the 8 tools and connects them to a Groq LLM.

import os
import asyncio
from llama_index.core.agent.workflow import ReActAgent
from llama_index.llms.groq import Groq

from src.agent.tools import (
    get_stock_analysis,
    get_risk_metrics,
    explain_anomaly,
    get_market_context,
    get_news_sentiment,
    get_trend_analysis,
    get_portfolio_overview,
    get_sector_analysis,
)

SYSTEM_PROMPT = """You are FinWatch AI, a financial risk analyst assistant.
You monitor a portfolio of equities for anomalies and drawdown risk.

You have tools to look up stock analysis, risk metrics, anomaly explanations,
market context, news sentiment, trend indicators, portfolio overview, and
sector analysis. Always use the tools to get real data — never make up numbers.

When answering:
- Be concise and precise, like a professional risk analyst.
- Cite the concrete numbers the tools return (e.g. VaR, drawdown probability).
- If a tool returns an error or no data, say so honestly.
- You may answer in the language the user writes in (English, German, or Arabic).

IMPORTANT — Regulatory constraint:
- You must NEVER give financial advice or recommendations to buy, sell, or hold.
- You do NOT tell the user what to do with their money or positions.
- You only describe and explain the data: risk metrics, anomalies, trends, context.
- If the user asks "should I buy/sell X?", explain the risk picture from the data
  and explicitly state that you cannot give investment recommendations.

This is decision-support and information only, not financial advice.
"""


def build_agent():
    llm = Groq(
        model="llama-3.3-70b-versatile",
        api_key=os.environ["GROQ_API_KEY"],
    )

    agent = ReActAgent(
        tools=[
            get_stock_analysis,
            get_risk_metrics,
            explain_anomaly,
            get_market_context,
            get_news_sentiment,
            get_trend_analysis,
            get_portfolio_overview,
            get_sector_analysis,
        ],
        llm=llm,
        system_prompt=SYSTEM_PROMPT,
    )
    return agent


async def main():
    agent = build_agent()
    response = await agent.run("Wie sieht das Risiko für AAPL aus?")
    print("\n--- ANTWORT ---")
    print(response)


if __name__ == "__main__":
    asyncio.run(main())