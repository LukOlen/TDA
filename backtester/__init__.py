"""
TDA Backtester — Algorithmic Trading Backtesting Framework
"""
from .engine import BacktestEngine
from .strategy import BaseStrategy
from .results import BacktestResult
from .screener import screen_universe
from .regime import detect_regime
from .monte_carlo import run_monte_carlo
from .rotation import run_rotation_backtest

__all__ = [
    "BacktestEngine",
    "BaseStrategy",
    "BacktestResult",
    "screen_universe",
    "detect_regime",
    "run_monte_carlo",
    "run_rotation_backtest",
]
