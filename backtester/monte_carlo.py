"""Monte Carlo risk estimation for backtested strategy returns."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class SimulationMethod(str, Enum):
    bootstrap = "bootstrap"
    block_bootstrap = "block_bootstrap"
    parametric_normal = "parametric_normal"
    parametric_t = "parametric_t"


@dataclass
class VaRResult:
    confidence_level: float
    var: float
    cvar: float


@dataclass
class ReturnDistribution:
    mean: float
    median: float
    std: float
    percentile_5: float
    percentile_25: float
    percentile_75: float
    percentile_95: float


@dataclass
class DrawdownDistribution:
    median: float
    percentile_95: float


@dataclass
class MonteCarloResult:
    n_simulations: int
    horizon: int
    method: str
    var_results: list[VaRResult]
    probability_of_loss: float
    return_distribution: ReturnDistribution
    drawdown_distribution: DrawdownDistribution
    sharpe_distribution: ReturnDistribution


# ---------------------------------------------------------------------------
# Simulation methods
# ---------------------------------------------------------------------------

def _simulate_bootstrap(
    returns: np.ndarray,
    n_simulations: int,
    horizon: int,
    rng: np.random.Generator,
    **kwargs,
) -> np.ndarray:
    """IID resample from historical returns."""
    idx = rng.integers(0, len(returns), size=(n_simulations, horizon))
    return returns[idx]


def _simulate_block_bootstrap(
    returns: np.ndarray,
    n_simulations: int,
    horizon: int,
    rng: np.random.Generator,
    *,
    block_size: int = 21,
    **kwargs,
) -> np.ndarray:
    """Block bootstrap preserving autocorrelation structure."""
    n = len(returns)
    n_blocks = (horizon + block_size - 1) // block_size
    # Random start indices for each block (circular wrapping)
    starts = rng.integers(0, n, size=(n_simulations, n_blocks))
    paths = np.empty((n_simulations, n_blocks * block_size))
    for b in range(n_blocks):
        for step in range(block_size):
            paths[:, b * block_size + step] = returns[(starts[:, b] + step) % n]
    return paths[:, :horizon]


def _simulate_parametric_normal(
    returns: np.ndarray,
    n_simulations: int,
    horizon: int,
    rng: np.random.Generator,
    **kwargs,
) -> np.ndarray:
    """Sample from a fitted normal distribution."""
    mu = float(np.mean(returns))
    sigma = float(np.std(returns, ddof=1))
    return rng.normal(mu, sigma, size=(n_simulations, horizon))


def _simulate_parametric_t(
    returns: np.ndarray,
    n_simulations: int,
    horizon: int,
    rng: np.random.Generator,
    **kwargs,
) -> np.ndarray:
    """Sample from a fitted Student-t distribution (fat tails)."""
    df, loc, scale = stats.t.fit(returns)
    return stats.t.rvs(df, loc=loc, scale=scale, size=(n_simulations, horizon),
                        random_state=rng)


_SIMULATION_METHODS: dict[SimulationMethod, Callable] = {
    SimulationMethod.bootstrap: _simulate_bootstrap,
    SimulationMethod.block_bootstrap: _simulate_block_bootstrap,
    SimulationMethod.parametric_normal: _simulate_parametric_normal,
    SimulationMethod.parametric_t: _simulate_parametric_t,
}


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _compute_path_max_drawdown(equity_paths: np.ndarray) -> np.ndarray:
    """Max drawdown per path. Returns array of shape (n_simulations,), values <= 0."""
    running_max = np.maximum.accumulate(equity_paths, axis=1)
    drawdowns = (equity_paths - running_max) / running_max
    return np.min(drawdowns, axis=1)


def _compute_path_sharpe(return_paths: np.ndarray) -> np.ndarray:
    """Annualised Sharpe ratio per simulated path."""
    means = np.mean(return_paths, axis=1)
    stds = np.std(return_paths, axis=1, ddof=1)
    # Avoid division by zero
    stds = np.where(stds == 0, np.nan, stds)
    return (means / stds) * np.sqrt(252)


def _compute_var_cvar(
    terminal_returns: np.ndarray,
    confidence_levels: list[float],
) -> list[VaRResult]:
    """VaR and CVaR at each confidence level."""
    results = []
    for cl in confidence_levels:
        cutoff = np.percentile(terminal_returns, (1 - cl) * 100)
        tail = terminal_returns[terminal_returns <= cutoff]
        cvar = float(np.mean(tail)) if len(tail) > 0 else float(cutoff)
        results.append(VaRResult(
            confidence_level=cl,
            var=float(cutoff),
            cvar=cvar,
        ))
    return results


def _make_distribution(values: np.ndarray) -> ReturnDistribution:
    """Build a ReturnDistribution from a 1-D array."""
    clean = values[np.isfinite(values)]
    if len(clean) == 0:
        return ReturnDistribution(
            mean=0.0, median=0.0, std=0.0,
            percentile_5=0.0, percentile_25=0.0,
            percentile_75=0.0, percentile_95=0.0,
        )
    p5, p25, p50, p75, p95 = np.percentile(clean, [5, 25, 50, 75, 95])
    return ReturnDistribution(
        mean=float(np.mean(clean)),
        median=float(p50),
        std=float(np.std(clean, ddof=1)),
        percentile_5=float(p5),
        percentile_25=float(p25),
        percentile_75=float(p75),
        percentile_95=float(p95),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_monte_carlo(
    returns: pd.Series,
    n_simulations: int = 1000,
    horizon: int = 252,
    confidence_levels: list[float] | None = None,
    method: str | SimulationMethod = "bootstrap",
    block_size: int = 21,
    seed: int | None = None,
) -> MonteCarloResult:
    """Run Monte Carlo simulation on strategy returns.

    Parameters
    ----------
    returns : pd.Series
        Daily net returns from a backtest.
    n_simulations : int
        Number of simulated paths.
    horizon : int
        Number of trading days per simulated path.
    confidence_levels : list[float] | None
        VaR/CVaR confidence levels. Defaults to [0.95, 0.99].
    method : str | SimulationMethod
        Simulation method name.
    block_size : int
        Block length for block bootstrap method.
    seed : int | None
        RNG seed for reproducibility.

    Returns
    -------
    MonteCarloResult
    """
    if confidence_levels is None:
        confidence_levels = [0.95, 0.99]

    # Validate
    ret_array = np.asarray(returns, dtype=float)
    if len(ret_array) < 2:
        raise ValueError("Need at least 2 return observations for Monte Carlo simulation")

    if isinstance(method, str):
        try:
            method_enum = SimulationMethod(method)
        except ValueError:
            valid = [m.value for m in SimulationMethod]
            raise ValueError(f"Unknown method {method!r}. Valid: {valid}")
    else:
        method_enum = method

    sim_fn = _SIMULATION_METHODS[method_enum]
    rng = np.random.default_rng(seed)

    # Simulate return paths: (n_simulations, horizon)
    paths = sim_fn(ret_array, n_simulations, horizon, rng, block_size=block_size)

    # Terminal cumulative returns (compounded)
    terminal_returns = np.prod(1 + paths, axis=1) - 1

    # Equity curves for drawdown calculation (start at 1.0)
    equity_paths = np.cumprod(1 + paths, axis=1)

    # Max drawdown per path
    max_drawdowns = _compute_path_max_drawdown(equity_paths)

    # Sharpe per path
    sharpes = _compute_path_sharpe(paths)

    # Aggregate
    var_results = _compute_var_cvar(terminal_returns, confidence_levels)
    prob_loss = float(np.mean(terminal_returns < 0))
    return_dist = _make_distribution(terminal_returns)
    dd_median, dd_p95 = np.percentile(max_drawdowns, [50, 95])
    drawdown_dist = DrawdownDistribution(
        median=float(dd_median),
        percentile_95=float(dd_p95),
    )
    sharpe_dist = _make_distribution(sharpes)

    return MonteCarloResult(
        n_simulations=n_simulations,
        horizon=horizon,
        method=method_enum.value,
        var_results=var_results,
        probability_of_loss=prob_loss,
        return_distribution=return_dist,
        drawdown_distribution=drawdown_dist,
        sharpe_distribution=sharpe_dist,
    )
