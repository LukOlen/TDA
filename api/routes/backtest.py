"""
Backtest route handlers.
"""
from __future__ import annotations

import asyncio
import dataclasses
import functools
import os
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, HTTPException

from backtester.data import load_ohlcv
from backtester.engine import BacktestEngine
from backtester.strategies import REGISTRY
from backtester.optimizer import (
    grid_search,
    random_search,
    walk_forward,
    VALID_METRICS,
)
from backtester.monte_carlo import run_monte_carlo
from api.models import (
    BacktestRequest,
    BacktestResponse,
    BatchBacktestRequest,
    BatchBacktestResponse,
    BatchBacktestError,
    CompareRequest,
    CompareResponse,
    OptimizeRequest,
    OptimizeResponse,
    OptimizationResultEntry,
    WalkForwardRequest,
    WalkForwardResponse,
    WalkForwardSplit,
    MonteCarloRequest,
    MonteCarloResponse,
    StrategyInfo,
    StrategyParam,
)

router = APIRouter(prefix="/api", tags=["backtest"])

# Shared thread-pool for CPU-bound backtest work
_EXECUTOR = ThreadPoolExecutor(max_workers=min(os.cpu_count() or 4, 8))


def _validate_strategy(key: str) -> type:
    """Return strategy class or raise 400."""
    if key not in REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown strategy {key!r}. Available: {list(REGISTRY.keys())}",
        )
    return REGISTRY[key]


def _load_data(ticker: str, start: str, end: str):
    """Load OHLCV or raise appropriate HTTP error."""
    try:
        return load_ohlcv(ticker, start=start, end=end)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to fetch market data: {exc}",
        )


def _run_backtest_sync(data, strategy_cls, params, initial_capital, commission, ticker):
    """Synchronous single-backtest helper."""
    try:
        strategy = strategy_cls(**params)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid strategy parameters: {exc}",
        )
    engine = BacktestEngine(
        data=data,
        strategy=strategy,
        initial_capital=initial_capital,
        commission=commission,
        ticker=ticker.upper(),
    )
    result = engine.run()
    return BacktestResponse(**result.to_dict())


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
    StrategyInfo(
        key="macd_momentum",
        name="MACD Momentum",
        description=(
            "Trend-following strategy that uses MACD histogram zero-crossings to "
            "identify momentum regime shifts.  Goes long when the histogram crosses "
            "above zero and short when it crosses below.  Always in the market "
            "(long or short) once the warmup period ends."
        ),
        params=[
            StrategyParam(name="fast_period",   type="int", default=12, description="Fast EMA period"),
            StrategyParam(name="slow_period",   type="int", default=26, description="Slow EMA period"),
            StrategyParam(name="signal_period", type="int", default=9,  description="Signal line EMA period"),
        ],
    ),
    StrategyInfo(
        key="rsi_trend_filter",
        name="RSI Trend Filter",
        description=(
            "Mean-reversion strategy using Wilder's RSI, gated by a long-term "
            "trend EMA.  Goes long only when RSI is oversold AND price is above the "
            "trend EMA; goes short only when RSI is overbought AND price is below it. "
            "Can be flat when RSI and trend disagree."
        ),
        params=[
            StrategyParam(name="rsi_period",   type="int",   default=14,  description="Wilder RSI smoothing period"),
            StrategyParam(name="oversold",     type="float", default=30.0, description="RSI oversold threshold (0–100)"),
            StrategyParam(name="overbought",   type="float", default=70.0, description="RSI overbought threshold (0–100)"),
            StrategyParam(name="trend_period", type="int",   default=200,  description="Long-term trend EMA period"),
        ],
    ),
    StrategyInfo(
        key="tsmom",
        name="Time-Series Momentum",
        description=(
            "Academic quant strategy from Moskowitz, Ooi & Pedersen (2012 JFE). "
            "Computes past returns over multiple horizons, scales each signal by "
            "target volatility / realised volatility, and averages across horizons. "
            "Supports both long and short positions."
        ),
        params=[
            StrategyParam(name="vol_target",   type="float", default=0.15, description="Target annualised volatility (e.g. 0.15 = 15%)"),
            StrategyParam(name="vol_lookback", type="int",   default=60,   description="Rolling window for realised volatility (days)"),
        ],
    ),
    StrategyInfo(
        key="keltner_breakout",
        name="Keltner Channel Breakout",
        description=(
            "ATR-based adaptive channel breakout strategy.  Enters long when price "
            "breaks above the upper Keltner Channel and short when it breaks below "
            "the lower channel.  Exits when price reverts to the midline EMA.  "
            "ATR computed via Wilder's smoothing — more stable than Bollinger Bands "
            "in trending markets."
        ),
        params=[
            StrategyParam(name="period",     type="int",   default=20,  description="EMA period for the midline"),
            StrategyParam(name="atr_period", type="int",   default=14,  description="Wilder ATR smoothing period"),
            StrategyParam(name="atr_mult",   type="float", default=1.5, description="Channel half-width in ATR units"),
        ],
    ),
]


@router.get("/strategies", response_model=list[StrategyInfo])
def list_strategies() -> list[StrategyInfo]:
    """Return all available strategies with their parameter schemas."""
    return _STRATEGY_CATALOGUE


# ---------------------------------------------------------------------------
# Single backtest
# ---------------------------------------------------------------------------

@router.post("/backtest", response_model=BacktestResponse)
async def run_backtest(req: BacktestRequest) -> BacktestResponse:
    """Execute a single backtest."""
    strategy_cls = _validate_strategy(req.strategy)
    data = _load_data(req.ticker, req.start_date, req.end_date)

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _EXECUTOR,
        functools.partial(
            _run_backtest_sync,
            data, strategy_cls, req.params,
            req.initial_capital, req.commission, req.ticker,
        ),
    )


# ---------------------------------------------------------------------------
# Batch backtest (multiple tickers)
# ---------------------------------------------------------------------------

@router.post("/backtest/batch", response_model=BatchBacktestResponse)
async def batch_backtest(req: BatchBacktestRequest) -> BatchBacktestResponse:
    """Run one strategy across multiple tickers concurrently."""
    strategy_cls = _validate_strategy(req.strategy)
    loop = asyncio.get_event_loop()

    def _single(ticker: str):
        try:
            data = load_ohlcv(ticker, start=req.start_date, end=req.end_date)
        except ValueError as exc:
            return BatchBacktestError(ticker=ticker, error=str(exc))
        except Exception as exc:
            return BatchBacktestError(
                ticker=ticker, error=f"Failed to fetch data: {exc}"
            )
        try:
            return _run_backtest_sync(
                data, strategy_cls, req.params,
                req.initial_capital, req.commission, ticker,
            )
        except HTTPException as exc:
            return BatchBacktestError(ticker=ticker, error=exc.detail)

    futures = [
        loop.run_in_executor(_EXECUTOR, _single, t)
        for t in req.tickers
    ]
    outcomes = await asyncio.gather(*futures)

    results = [o for o in outcomes if isinstance(o, BacktestResponse)]
    errors = [o for o in outcomes if isinstance(o, BatchBacktestError)]

    return BatchBacktestResponse(results=results, errors=errors)


# ---------------------------------------------------------------------------
# Strategy comparison
# ---------------------------------------------------------------------------

@router.post("/backtest/compare", response_model=CompareResponse)
async def compare_strategies(req: CompareRequest) -> CompareResponse:
    """Run multiple strategies on the same ticker for side-by-side comparison."""
    # Validate all strategy keys upfront
    strategy_classes = []
    for spec in req.strategies:
        strategy_classes.append(
            (_validate_strategy(spec.strategy), spec.params)
        )

    data = _load_data(req.ticker, req.start_date, req.end_date)
    loop = asyncio.get_event_loop()

    futures = [
        loop.run_in_executor(
            _EXECUTOR,
            functools.partial(
                _run_backtest_sync,
                data, cls, params,
                req.initial_capital, req.commission, req.ticker,
            ),
        )
        for cls, params in strategy_classes
    ]
    results = await asyncio.gather(*futures)

    return CompareResponse(
        ticker=req.ticker.upper(),
        start_date=req.start_date,
        end_date=req.end_date,
        results=list(results),
    )


# ---------------------------------------------------------------------------
# Parameter optimisation
# ---------------------------------------------------------------------------

_MAX_GRID_COMBOS = 5000


@router.post("/backtest/optimize", response_model=OptimizeResponse)
async def optimize_strategy(req: OptimizeRequest) -> OptimizeResponse:
    """Grid or random search over a strategy's parameter space."""
    strategy_cls = _validate_strategy(req.strategy)

    if req.target_metric not in VALID_METRICS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unknown target_metric {req.target_metric!r}. "
                f"Valid: {sorted(VALID_METRICS)}"
            ),
        )

    if req.method not in ("grid", "random"):
        raise HTTPException(
            status_code=422,
            detail="method must be 'grid' or 'random'",
        )

    param_grid = {pr.name: pr.values for pr in req.param_grid}

    # Guard against huge grids
    if req.method == "grid":
        total = 1
        for vals in param_grid.values():
            total *= len(vals)
        if total > _MAX_GRID_COMBOS:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Grid has {total} combinations (max {_MAX_GRID_COMBOS}). "
                    f"Use method='random' with n_samples instead."
                ),
            )

    data = _load_data(req.ticker, req.start_date, req.end_date)
    loop = asyncio.get_event_loop()

    if req.method == "grid":
        search_fn = functools.partial(
            grid_search, data, strategy_cls, param_grid, req.target_metric,
            req.initial_capital, req.commission, req.ticker.upper(), req.maximize,
        )
    else:
        search_fn = functools.partial(
            random_search, data, strategy_cls, param_grid, req.target_metric,
            req.n_samples, req.initial_capital, req.commission,
            req.ticker.upper(), req.maximize, req.seed,
        )

    raw_results = await loop.run_in_executor(_EXECUTOR, search_fn)

    entries = [
        OptimizationResultEntry(
            rank=r["rank"], params=r["params"], metrics=r["metrics"],
        )
        for r in raw_results
    ]

    best_params = raw_results[0]["params"] if raw_results else {}
    best_value = (
        raw_results[0]["metrics"].get(req.target_metric) if raw_results else None
    )

    return OptimizeResponse(
        strategy=req.strategy,
        ticker=req.ticker.upper(),
        target_metric=req.target_metric,
        method=req.method,
        total_combinations_evaluated=len(raw_results),
        best_params=best_params,
        best_metric_value=best_value,
        results=entries,
    )


# ---------------------------------------------------------------------------
# Walk-forward analysis
# ---------------------------------------------------------------------------

@router.post("/backtest/walk-forward", response_model=WalkForwardResponse)
async def walk_forward_analysis(req: WalkForwardRequest) -> WalkForwardResponse:
    """Walk-forward optimisation with in-sample / out-of-sample splits."""
    strategy_cls = _validate_strategy(req.strategy)

    if req.target_metric not in VALID_METRICS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unknown target_metric {req.target_metric!r}. "
                f"Valid: {sorted(VALID_METRICS)}"
            ),
        )

    param_grid = {pr.name: pr.values for pr in req.param_grid}
    data = _load_data(req.ticker, req.start_date, req.end_date)
    loop = asyncio.get_event_loop()

    wf_fn = functools.partial(
        walk_forward, data, strategy_cls, param_grid, req.target_metric,
        req.n_splits, req.in_sample_pct, req.anchored,
        req.initial_capital, req.commission, req.ticker.upper(), req.maximize,
    )

    try:
        raw = await loop.run_in_executor(_EXECUTOR, wf_fn)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    splits = [WalkForwardSplit(**s) for s in raw["splits"]]

    return WalkForwardResponse(
        strategy=req.strategy,
        ticker=req.ticker.upper(),
        n_splits=req.n_splits,
        target_metric=req.target_metric,
        splits=splits,
        aggregate_oos_metrics=raw["aggregate_oos_metrics"],
    )


# ---------------------------------------------------------------------------
# Monte Carlo simulation
# ---------------------------------------------------------------------------

def _run_monte_carlo_sync(req: MonteCarloRequest) -> dict:
    """Synchronous Monte Carlo helper."""
    strategy_cls = _validate_strategy(req.strategy)
    data = _load_data(req.ticker, req.start_date, req.end_date)

    try:
        strategy = strategy_cls(**req.params)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid strategy parameters: {exc}",
        )

    engine = BacktestEngine(
        data=data,
        strategy=strategy,
        initial_capital=req.initial_capital,
        commission=req.commission,
        ticker=req.ticker.upper(),
    )
    result = engine.run()

    try:
        mc = run_monte_carlo(
            returns=result.returns,
            n_simulations=req.n_simulations,
            horizon=req.horizon,
            confidence_levels=req.confidence_levels,
            method=req.method,
            block_size=req.block_size,
            seed=req.seed,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return {
        "ticker": req.ticker.upper(),
        "strategy": req.strategy,
        **dataclasses.asdict(mc),
    }


@router.post("/backtest/monte-carlo", response_model=MonteCarloResponse)
async def monte_carlo_analysis(req: MonteCarloRequest) -> MonteCarloResponse:
    """Run Monte Carlo risk simulation on a strategy's returns."""
    loop = asyncio.get_event_loop()
    try:
        raw = await loop.run_in_executor(
            _EXECUTOR,
            functools.partial(_run_monte_carlo_sync, req),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Monte Carlo simulation failed: {exc}",
        )
    return MonteCarloResponse(**raw)
