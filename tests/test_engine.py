"""Unit tests for the BacktestEngine."""
import numpy as np
import pandas as pd
import pytest

from backtester.engine import BacktestEngine
from backtester.strategy import BaseStrategy
from backtester.results import BacktestResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_ohlcv(n: int = 300, start_price: float = 100.0) -> pd.DataFrame:
    """Generate synthetic trending OHLCV data."""
    rng = np.random.default_rng(42)
    close = start_price * np.exp(np.cumsum(rng.normal(0.0005, 0.01, n)))
    high = close * (1 + abs(rng.normal(0, 0.005, n)))
    low = close * (1 - abs(rng.normal(0, 0.005, n)))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    volume = rng.integers(1_000_000, 10_000_000, n).astype(float)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


class AlwaysLong(BaseStrategy):
    name = "AlwaysLong"

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series(1, index=data.index)


class AlwaysFlat(BaseStrategy):
    name = "AlwaysFlat"

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series(0, index=data.index)


class AlternatingStrategy(BaseStrategy):
    """Alternates long/flat every 20 bars."""
    name = "Alternating"

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        n = len(data)
        return pd.Series(
            [1 if (i // 20) % 2 == 0 else 0 for i in range(n)],
            index=data.index,
        )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestEngineConstruction:
    def test_valid_data_accepted(self):
        data = make_ohlcv()
        engine = BacktestEngine(data, AlwaysLong())
        assert engine.initial_capital == 100_000.0

    def test_empty_data_raises(self):
        with pytest.raises(ValueError, match="empty"):
            BacktestEngine(pd.DataFrame(), AlwaysLong())

    def test_missing_columns_raises(self):
        data = make_ohlcv().drop(columns=["volume"])
        with pytest.raises(ValueError, match="missing columns"):
            BacktestEngine(data, AlwaysLong())


# ---------------------------------------------------------------------------
# Result shape and types
# ---------------------------------------------------------------------------

class TestBacktestResult:
    def setup_method(self):
        self.data = make_ohlcv()
        self.result: BacktestResult = BacktestEngine(
            self.data, AlwaysLong()
        ).run()

    def test_returns_backtest_result(self):
        assert isinstance(self.result, BacktestResult)

    def test_equity_curve_length(self):
        assert len(self.result.equity_curve) == len(self.data)

    def test_equity_curve_starts_at_capital(self):
        assert self.result.equity_curve.iloc[0] == pytest.approx(100_000.0)

    def test_equity_curve_positive(self):
        assert (self.result.equity_curve > 0).all()

    def test_returns_length(self):
        assert len(self.result.returns) == len(self.data)

    def test_metrics_dict_populated(self):
        assert isinstance(self.result.metrics, dict)
        assert "sharpe_ratio" in self.result.metrics
        assert "max_drawdown" in self.result.metrics

    def test_ticker_set(self):
        engine = BacktestEngine(self.data, AlwaysLong(), ticker="TEST")
        result = engine.run()
        assert result.ticker == "TEST"


# ---------------------------------------------------------------------------
# Always-flat → no returns, no trades
# ---------------------------------------------------------------------------

class TestAlwaysFlatStrategy:
    def test_equity_stays_constant(self):
        data = make_ohlcv()
        result = BacktestEngine(data, AlwaysFlat()).run()
        # No positions held, equity should remain at initial_capital
        # Use max absolute deviation rather than pytest.approx (doesn't do element-wise pandas comparisons)
        assert (result.equity_curve - 100_000.0).abs().max() < 1.0

    def test_no_trades(self):
        data = make_ohlcv()
        result = BacktestEngine(data, AlwaysFlat()).run()
        assert result.trades.empty


# ---------------------------------------------------------------------------
# Look-ahead bias check
# ---------------------------------------------------------------------------

class TestNoLookAheadBias:
    def test_signal_shift(self):
        """The engine must shift signals forward by one bar."""
        data = make_ohlcv(50)
        engine = BacktestEngine(data, AlwaysLong())
        # Manually check that the first return is 0 (no position on day 0)
        result = engine.run()
        # First bar return should be 0 because position.shift(1) produces NaN → 0
        assert result.returns.iloc[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Trade extraction
# ---------------------------------------------------------------------------

class TestTradeExtraction:
    def test_alternating_generates_trades(self):
        data = make_ohlcv(200)
        result = BacktestEngine(data, AlternatingStrategy()).run()
        assert len(result.trades) > 0

    def test_trade_columns(self):
        data = make_ohlcv(200)
        result = BacktestEngine(data, AlternatingStrategy()).run()
        expected = {"entry_date", "exit_date", "direction", "entry_price",
                    "exit_price", "pnl", "pnl_pct"}
        assert expected.issubset(set(result.trades.columns))

    def test_exit_date_after_entry_date(self):
        data = make_ohlcv(200)
        result = BacktestEngine(data, AlternatingStrategy()).run()
        assert (result.trades["exit_date"] > result.trades["entry_date"]).all()


# ---------------------------------------------------------------------------
# Commission
# ---------------------------------------------------------------------------

class TestCommission:
    def test_higher_commission_lowers_returns(self):
        data = make_ohlcv()
        r_low = BacktestEngine(data, AlternatingStrategy(), commission=0.0).run()
        r_high = BacktestEngine(data, AlternatingStrategy(), commission=0.01).run()
        assert r_low.equity_curve.iloc[-1] >= r_high.equity_curve.iloc[-1]


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_to_dict_has_required_keys(self):
        data = make_ohlcv()
        result = BacktestEngine(data, AlwaysLong()).run()
        d = result.to_dict()
        assert "equity_curve" in d
        assert "metrics" in d
        assert "trades" in d

    def test_equity_curve_is_list_of_dicts(self):
        data = make_ohlcv()
        result = BacktestEngine(data, AlwaysLong()).run()
        ec = result.to_dict()["equity_curve"]
        assert isinstance(ec, list)
        assert "date" in ec[0]
        assert "value" in ec[0]
