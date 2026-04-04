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


# ---------------------------------------------------------------------------
# Batch backtest
# ---------------------------------------------------------------------------

class BatchBacktestRequest(BaseModel):
    tickers: list[str] = Field(
        ..., min_length=1, max_length=50, description="List of ticker symbols"
    )
    start_date: str = Field(..., examples=["2020-01-01"])
    end_date: str = Field(..., examples=["2024-01-01"])
    strategy: str = Field(..., examples=["sma_crossover"])
    initial_capital: float = Field(100_000.0, gt=0)
    commission: float = Field(0.001, ge=0, le=0.05)
    params: dict[str, Any] = Field(default_factory=dict)


class BatchBacktestError(BaseModel):
    ticker: str
    error: str


class BatchBacktestResponse(BaseModel):
    results: list[BacktestResponse]
    errors: list[BatchBacktestError]


# ---------------------------------------------------------------------------
# Strategy comparison
# ---------------------------------------------------------------------------

class StrategySpec(BaseModel):
    strategy: str
    params: dict[str, Any] = Field(default_factory=dict)


class CompareRequest(BaseModel):
    ticker: str = Field(..., examples=["AAPL"])
    start_date: str = Field(..., examples=["2020-01-01"])
    end_date: str = Field(..., examples=["2024-01-01"])
    strategies: list[StrategySpec] = Field(
        ..., min_length=2, max_length=10,
        description="List of strategies to compare",
    )
    initial_capital: float = Field(100_000.0, gt=0)
    commission: float = Field(0.001, ge=0, le=0.05)


class CompareResponse(BaseModel):
    ticker: str
    start_date: str
    end_date: str
    results: list[BacktestResponse]


# ---------------------------------------------------------------------------
# Parameter optimisation
# ---------------------------------------------------------------------------

class ParamRange(BaseModel):
    """Discrete values for a single parameter to search over."""
    name: str
    values: list[Any] = Field(..., min_length=1)


class OptimizeRequest(BaseModel):
    ticker: str = Field(..., examples=["AAPL"])
    start_date: str = Field(..., examples=["2020-01-01"])
    end_date: str = Field(..., examples=["2024-01-01"])
    strategy: str = Field(..., examples=["sma_crossover"])
    param_grid: list[ParamRange] = Field(..., min_length=1)
    target_metric: str = Field("sharpe_ratio")
    maximize: bool = Field(True)
    method: str = Field("grid", description="'grid' or 'random'")
    n_samples: int = Field(50, gt=0, description="Samples for random search")
    initial_capital: float = Field(100_000.0, gt=0)
    commission: float = Field(0.001, ge=0, le=0.05)
    seed: Optional[int] = None


class OptimizationResultEntry(BaseModel):
    rank: int
    params: dict[str, Any]
    metrics: BacktestMetrics


class OptimizeResponse(BaseModel):
    strategy: str
    ticker: str
    target_metric: str
    method: str
    total_combinations_evaluated: int
    best_params: dict[str, Any]
    best_metric_value: Optional[float]
    results: list[OptimizationResultEntry]


# ---------------------------------------------------------------------------
# Walk-forward analysis
# ---------------------------------------------------------------------------

class WalkForwardRequest(BaseModel):
    ticker: str = Field(..., examples=["AAPL"])
    start_date: str = Field(..., examples=["2018-01-01"])
    end_date: str = Field(..., examples=["2024-01-01"])
    strategy: str = Field(..., examples=["sma_crossover"])
    param_grid: list[ParamRange] = Field(..., min_length=1)
    target_metric: str = Field("sharpe_ratio")
    maximize: bool = Field(True)
    n_splits: int = Field(5, ge=2, le=20)
    in_sample_pct: float = Field(0.7, gt=0.1, lt=1.0)
    anchored: bool = Field(False)
    initial_capital: float = Field(100_000.0, gt=0)
    commission: float = Field(0.001, ge=0, le=0.05)


class WalkForwardSplit(BaseModel):
    split_index: int
    in_sample_start: str
    in_sample_end: str
    oos_start: str
    oos_end: str
    best_params: dict[str, Any]
    in_sample_metrics: BacktestMetrics
    oos_metrics: BacktestMetrics
    oos_equity_curve: list[EquityPoint]


class WalkForwardResponse(BaseModel):
    strategy: str
    ticker: str
    n_splits: int
    target_metric: str
    splits: list[WalkForwardSplit]
    aggregate_oos_metrics: BacktestMetrics


# ---------------------------------------------------------------------------
# Stock screener
# ---------------------------------------------------------------------------

class ScreenRequest(BaseModel):
    tickers: Optional[list[str]] = Field(
        None,
        max_length=100,
        description="Ticker universe. None uses a default large-cap set (~30 stocks).",
    )
    benchmark: str = Field("SPY", description="Benchmark ticker")
    start_date: Optional[str] = Field(None, examples=["2023-04-01"])
    end_date: Optional[str] = Field(None, examples=["2024-04-01"])
    top_n: Optional[int] = Field(None, gt=0, description="Return only top N results")


class ScreenerScore(BaseModel):
    ticker: str
    rank: int
    composite_score: float
    relative_strength: float
    momentum_1m: float
    momentum_3m: float
    momentum_6m: float
    momentum_12m: Optional[float]
    sharpe: float
    volatility: float
    max_drawdown: float


class ScreenerError(BaseModel):
    ticker: str
    error: str


class ScreenResponse(BaseModel):
    benchmark: str
    start_date: str
    end_date: str
    total_screened: int
    results: list[ScreenerScore]
    errors: list[ScreenerError]


# ---------------------------------------------------------------------------
# Market regime detection
# ---------------------------------------------------------------------------

class RegimeRequest(BaseModel):
    benchmark: str = Field("SPY")
    start_date: Optional[str] = Field(None, examples=["2023-01-01"])
    end_date: Optional[str] = Field(None, examples=["2024-04-01"])
    bullish_threshold: float = Field(0.2, ge=0.0, le=1.0)
    bearish_threshold: float = Field(-0.2, ge=-1.0, le=0.0)


class RegimeIndicatorModel(BaseModel):
    name: str
    value: float
    score: float
    interpretation: str


class RegimeResponse(BaseModel):
    regime: str
    composite_score: float
    confidence: float
    as_of_date: str
    indicators: list[RegimeIndicatorModel]


# ---------------------------------------------------------------------------
# Monte Carlo simulation
# ---------------------------------------------------------------------------

class MonteCarloRequest(BaseModel):
    ticker: str = Field(..., examples=["AAPL"], description="Ticker symbol")
    start_date: str = Field(..., examples=["2020-01-01"], description="YYYY-MM-DD")
    end_date: str = Field(..., examples=["2024-01-01"], description="YYYY-MM-DD")
    strategy: str = Field(..., examples=["sma_crossover"])
    params: dict[str, Any] = Field(default_factory=dict)
    initial_capital: float = Field(100_000.0, gt=0)
    commission: float = Field(0.001, ge=0, le=0.05)
    n_simulations: int = Field(1000, gt=0, le=100_000)
    horizon: int = Field(252, gt=0, le=2520)
    confidence_levels: list[float] = Field(
        default_factory=lambda: [0.95, 0.99],
    )
    method: str = Field("bootstrap")
    block_size: int = Field(21, gt=0)
    seed: Optional[int] = None


class VaRResultModel(BaseModel):
    confidence_level: float
    var: float
    cvar: float


class ReturnDistributionModel(BaseModel):
    mean: float
    median: float
    std: float
    percentile_5: float
    percentile_25: float
    percentile_75: float
    percentile_95: float


class DrawdownDistributionModel(BaseModel):
    median: float
    percentile_95: float


class MonteCarloResponse(BaseModel):
    ticker: str
    strategy: str
    n_simulations: int
    horizon: int
    method: str
    var_results: list[VaRResultModel]
    probability_of_loss: float
    return_distribution: ReturnDistributionModel
    drawdown_distribution: DrawdownDistributionModel
    sharpe_distribution: ReturnDistributionModel
