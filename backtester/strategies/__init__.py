from .sma_crossover import SMACrossover
from .mean_reversion import BollingerMeanReversion
from .momentum import BreakoutMomentum
from .macd_momentum import MACDMomentum
from .rsi_trend_filter import RSITrendFilter
from .tsmom import TSMOM
from .keltner_breakout import KeltnerBreakout
from .adaptive import AdaptiveStrategy
from .tda_adaptive import TDAAdaptiveStrategy
from .tda_ml_enhanced import TDAMLEnhancedStrategy

REGISTRY: dict[str, type] = {
    "sma_crossover":    SMACrossover,
    "mean_reversion":   BollingerMeanReversion,
    "momentum":         BreakoutMomentum,
    "macd_momentum":    MACDMomentum,
    "rsi_trend_filter": RSITrendFilter,
    "tsmom":            TSMOM,
    "keltner_breakout": KeltnerBreakout,
    "adaptive":         AdaptiveStrategy,
    "tda_adaptive":     TDAAdaptiveStrategy,
    "tda_ml_enhanced":  TDAMLEnhancedStrategy,
}

__all__ = [
    "SMACrossover",
    "BollingerMeanReversion",
    "BreakoutMomentum",
    "MACDMomentum",
    "RSITrendFilter",
    "TSMOM",
    "KeltnerBreakout",
    "AdaptiveStrategy",
    "TDAAdaptiveStrategy",
    "TDAMLEnhancedStrategy",
    "REGISTRY",
]
