# FinWatch AI — REST API
# FastAPI wrapper around the 11 agent tools
# Run with: uvicorn src.api.main:app --reload

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import sys
from pathlib import Path

# make sure imports work
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.agent.tools import (
    get_stock_analysis,
    get_risk_metrics,
    explain_anomaly,
    get_market_context,
    get_news_sentiment,
    get_trend_analysis,
    get_portfolio_overview,
    get_sector_analysis,
    get_macro_context,
    get_earnings_calendar,
    get_correlation_risk,
)

app = FastAPI(
    title="FinWatch AI",
    description="AI-powered equity risk assessment & anomaly detection API",
    version="1.0.0",
)

# CORS — allow any frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health check ──────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "system": "FinWatch AI", "version": "1.0.0"}


# ── Stock endpoints ───────────────────────────────────────────
@app.get("/api/stock/{ticker}")
def stock_analysis(ticker: str):
    result = get_stock_analysis(ticker.upper())
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@app.get("/api/stock/{ticker}/risk")
def risk_metrics(ticker: str):
    result = get_risk_metrics(ticker.upper())
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@app.get("/api/stock/{ticker}/explain")
def explain(ticker: str):
    result = explain_anomaly(ticker.upper())
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@app.get("/api/stock/{ticker}/context")
def market_context(ticker: str):
    result = get_market_context(ticker.upper())
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@app.get("/api/stock/{ticker}/news")
def news_sentiment(ticker: str):
    result = get_news_sentiment(ticker.upper())
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@app.get("/api/stock/{ticker}/trend")
def trend_analysis(ticker: str):
    result = get_trend_analysis(ticker.upper())
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@app.get("/api/stock/{ticker}/correlation")
def correlation_risk(ticker: str):
    result = get_correlation_risk(ticker.upper())
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


# ── Portfolio endpoints ───────────────────────────────────────
@app.get("/api/portfolio")
def portfolio_overview():
    return get_portfolio_overview()


@app.get("/api/sectors")
def sector_analysis():
    return get_sector_analysis()


# ── Macro endpoints ───────────────────────────────────────────
@app.get("/api/macro")
def macro_context():
    return get_macro_context()


@app.get("/api/earnings")
def earnings_calendar():
    return get_earnings_calendar()


# ── Agent chat endpoint ───────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    ticker: Optional[str] = None


@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        from src.agent.agent import build_agent
        import asyncio
        agent = build_agent()
        msg = request.message
        if request.ticker:
            from finwatch.data.loader import COMPANY_NAMES
            company = COMPANY_NAMES.get(request.ticker, request.ticker)
            msg = f"(Context: viewing {company} ({request.ticker})) {msg}"
        response = await agent.run(msg)
        return {"answer": str(response)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))