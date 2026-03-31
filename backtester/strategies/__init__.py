from .sma_crossover import SMACrossover
from .mean_reversion import BollingerMeanReversion
from .momentum import BreakoutMomentum

REGISTRY: dict[str, type] = {
    "sma_crossover": SMACrossover,
    "mean_reversion": BollingerMeanReversion,
    "momentum": BreakoutMomentum,
}

__all__ = ["SMACrossover", "BollingerMeanReversion", "BreakoutMomentum", "REGISTRY"]
