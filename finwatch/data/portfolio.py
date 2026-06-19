"""
FinWatch AI — Portfolio Storage
=================================
Load / save user portfolios to data/portfolios/portfolios.json.

Schema:
{
  "My Portfolio": [
    {"ticker": "AAPL", "shares": 50, "entry_price": 210.00, "entry_date": "2024-11-01"}
  ]
}
"""

import json
from pathlib import Path
from datetime import date

# Anchor to project root (two levels up from finwatch/data/)
_ROOT = Path(__file__).resolve().parents[2]
PORTFOLIO_PATH = _ROOT / "data" / "portfolios" / "portfolios.json"


def load_portfolios() -> dict:
    PORTFOLIO_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not PORTFOLIO_PATH.exists():
        return {}
    try:
        with open(PORTFOLIO_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def save_portfolios(portfolios: dict) -> None:
    PORTFOLIO_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PORTFOLIO_PATH, "w") as f:
        json.dump(portfolios, f, indent=2)


def add_position(portfolios: dict, name: str, ticker: str,
                 shares: float, entry_price: float, entry_date: str = None) -> dict:
    if name not in portfolios:
        portfolios[name] = []
    # Remove existing entry for same ticker (update)
    portfolios[name] = [p for p in portfolios[name] if p["ticker"] != ticker]
    portfolios[name].append({
        "ticker":      ticker,
        "shares":      shares,
        "entry_price": entry_price,
        "entry_date":  entry_date or str(date.today()),
    })
    return portfolios


def remove_position(portfolios: dict, name: str, ticker: str) -> dict:
    if name in portfolios:
        portfolios[name] = [p for p in portfolios[name] if p["ticker"] != ticker]
    return portfolios


def create_portfolio(portfolios: dict, name: str) -> dict:
    if name not in portfolios:
        portfolios[name] = []
    return portfolios


def delete_portfolio(portfolios: dict, name: str) -> dict:
    portfolios.pop(name, None)
    return portfolios
