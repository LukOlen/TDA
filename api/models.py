"""Pydantic request / response models for the backtesting API."""
from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------

class BacktestRequest(BaseModel):
    ticker: str = Field(..., examples=["AAPL"], description="Ticker symbol")
    start_date: str = Field(..., examples=["2020-01-01"], description="YYYY-MM-DD")
    end_date: str = Field(..., examples=["2024-01-01"], description="YYYY-MM-DD")
    strategy: str = Field(
        ...,
        examples=["sma_crossover"],
        description="Strategy key: sma_crossover | mean_reversion | momentum",
    )
    initial_capital: float = Field(100_000.0, gt=0)
    commission: float = Field(0.001, ge=0, le=0.05)
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Strategy-specific hyper-parameters",
    )


# ---------------------------------------------------------------------------
# Response sub-models
# ---------------------------------------------------------------------------

class EquityPoint(BaseModel):
    date: str
    value: float


class TradeRecord(BaseModel):
    entry_date: str
    exit_date: str
    direction: str
    entry_price: float
    exit_price: float
    pnl: float
    pnl_pct: float


class BacktestMetrics(BaseModel):
    total_return: Optional[float]
    cagr: Optional[float]
    sharpe_ratio: Optional[float]
    sortino_ratio: Optional[float]
    max_drawdown: Optional[float]
    calmar_ratio: Optional[float]
    volatility: Optional[float]
    win_rate: Optional[float]
    profit_factor: Optional[float]
    avg_trade_return: Optional[float]
    total_trades: int
    benchmark_return: Optional[float]
    beta: Optional[float]
    alpha: Optional[float]


# ---------------------------------------------------------------------------
# Top-level response
# ---------------------------------------------------------------------------

class BacktestResponse(BaseModel):
    strategy_name: str
    ticker: str
    start_date: str
    end_date: str
    initial_capital: float
    metrics: BacktestMetrics
    equity_curve: list[EquityPoint]
    trades: list[TradeRecord]


# ---------------------------------------------------------------------------
# Strategy catalogue
# ---------------------------------------------------------------------------

class StrategyParam(BaseModel):
    name: str
    type: str
    default: Any
    description: str


class StrategyInfo(BaseModel):
    key: str
    name: str
    description: str
    params: list[StrategyParam]
