from abc import ABC, abstractmethod
import pandas as pd


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.

    Subclasses must implement ``generate_signals``, which accepts an OHLCV
    DataFrame and returns a Series of integer signals aligned to the same index:

        1  — long
       -1  — short
        0  — flat / out of market
    """

    #: Human-readable strategy name used in results and API responses.
    name: str = "BaseStrategy"

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Derive trading signals from OHLCV data.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame with columns ``open``, ``high``, ``low``, ``close``,
            ``volume`` and a DatetimeIndex.

        Returns
        -------
        pd.Series
            Integer signal series (1 / -1 / 0) with the same index as *data*.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
