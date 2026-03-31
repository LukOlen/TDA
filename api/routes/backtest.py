"""
Backtest route handlers.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backtester.data import load_ohlcv
from backtester.engine import BacktestEngine
from backtester.strategies import REGISTRY
from api.models import (
    BacktestRequest,
    BacktestResponse,
    StrategyInfo,
    StrategyParam,
)

router = APIRouter(prefix="/api", tags=["backtest"])


# ---------------------------------------------------------------------------
# Strategy catalogue
# ---------------------------------------------------------------------------

_STRATEGY_CATALOGUE: list[StrategyInfo] = [
    StrategyInfo(
        key="sma_crossover",
        name="SMA Crossover",
        description=(
            "Trend-following strategy that goes long when a fast simple moving "
            "average crosses above a slow SMA, and exits when it crosses below."
        ),
        params=[
            StrategyParam(name="fast_period", type="int", default=20, description="Fast SMA period"),
            StrategyParam(name="slow_period", type="int", default=50, description="Slow SMA period"),
        ],
    ),
    StrategyInfo(
        key="mean_reversion",
        name="Bollinger Mean Reversion",
        description=(
            "Statistical mean-reversion strategy that buys when price pierces "
            "the lower Bollinger Band (oversold) and exits at the upper band or "
            "middle band (price has reverted to mean)."
        ),
        params=[
            StrategyParam(name="period", type="int", default=20, description="Rolling window length"),
            StrategyParam(name="num_std", type="float", default=2.0, description="Band width in standard deviations"),
        ],
    ),
    StrategyInfo(
        key="momentum",
        name="Breakout Momentum",
        description=(
            "Donchian-channel breakout strategy inspired by the Turtle Trading "
            "rules.  Enters long on an N-day high breakout; exits on an N-day "
            "low breakdown."
        ),
        params=[
            StrategyParam(name="lookback", type="int", default=20, description="Channel lookback period"),
        ],
    ),
]


@router.get("/strategies", response_model=list[StrategyInfo])
def list_strategies() -> list[StrategyInfo]:
    """Return all available strategies with their parameter schemas."""
    return _STRATEGY_CATALOGUE


# ---------------------------------------------------------------------------
# Run backtest
# ---------------------------------------------------------------------------

@router.post("/backtest", response_model=BacktestResponse)
def run_backtest(req: BacktestRequest) -> BacktestResponse:
    """
    Execute a backtest and return the full result.

    The endpoint fetches OHLCV data from Yahoo Finance, runs the requested
    strategy through the vectorised engine, and returns the equity curve,
    trade log, and performance metrics.
    """
    if req.strategy not in REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown strategy {req.strategy!r}. "
                f"Available: {list(REGISTRY.keys())}"
            ),
        )

    # Fetch market data
    try:
        data = load_ohlcv(req.ticker, start=req.start_date, end=req.end_date)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to fetch market data: {exc}",
        )

    # Instantiate strategy with user-supplied params
    strategy_cls = REGISTRY[req.strategy]
    try:
        strategy = strategy_cls(**req.params)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid strategy parameters: {exc}",
        )

    # Run backtest
    engine = BacktestEngine(
        data=data,
        strategy=strategy,
        initial_capital=req.initial_capital,
        commission=req.commission,
        ticker=req.ticker.upper(),
    )
    result = engine.run()

    return BacktestResponse(**result.to_dict())
