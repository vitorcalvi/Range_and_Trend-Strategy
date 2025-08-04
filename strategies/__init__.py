"""
Dual Strategy Trading System

This package contains the complete dual strategy implementation:

1. RangeStrategy: RSI+MFI mean reversion for range-bound markets (ADX < 25)
2. TrendStrategy: RSI+MA trend following for trending markets (ADX > 25) 
3. StrategyManager: Market condition detection and strategy selection

The system automatically switches between strategies based on real-time
market analysis using ADX and supporting indicators.
"""

from .range_strategy import RangeStrategy
from .trend_strategy import TrendStrategy
from .strategy_manager import StrategyManager, MarketConditionDetector

__all__ = [
    'RangeStrategy',
    'TrendStrategy', 
    'StrategyManager',
    'MarketConditionDetector'
]

__version__ = "1.0.0"