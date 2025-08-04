import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Tuple, Any

class MarketConditionDetector:
    """FIXED: Simplified and more reliable market condition detection"""
    
    def __init__(self):
        self.adx_period = 14
        self.bb_period = 20
        
    def calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series) -> float:
        """FIXED: Simplified ADX calculation"""
        if len(close) < 20:
            return 20.0  # Default to ranging
        
        # Use last 30 periods for calculation (was 50, too much)
        h = high.iloc[-30:]
        l = low.iloc[-30:]
        c = close.iloc[-30:]
        
        tr_list, dm_plus_list, dm_minus_list = [], [], []
        
        for i in range(1, len(h)):
            # True Range
            tr_val = max(
                h.iloc[i] - l.iloc[i],
                abs(h.iloc[i] - c.iloc[i-1]),
                abs(l.iloc[i] - c.iloc[i-1])
            )
            tr_list.append(tr_val)
            
            # Directional Movement
            h_move = h.iloc[i] - h.iloc[i-1]
            l_move = l.iloc[i-1] - l.iloc[i]
            
            dm_plus_list.append(h_move if h_move > l_move and h_move > 0 else 0)
            dm_minus_list.append(l_move if l_move > h_move and l_move > 0 else 0)
        
        if len(tr_list) < 14:
            return 20.0
        
        # Calculate averages
        tr_avg = sum(tr_list[-14:]) / 14
        dm_plus_avg = sum(dm_plus_list[-14:]) / 14
        dm_minus_avg = sum(dm_minus_list[-14:]) / 14
        
        if tr_avg == 0:
            return 20.0
        
        # Calculate DI and DX
        di_plus = 100 * dm_plus_avg / tr_avg
        di_minus = 100 * dm_minus_avg / tr_avg
        di_sum = di_plus + di_minus
        
        if di_sum == 0:
            return 20.0
        
        dx = 100 * abs(di_plus - di_minus) / di_sum
        return np.clip(dx, 0, 100)
    
    def calculate_volatility_regime(self, close: pd.Series) -> str:
        """Calculate volatility regime using Bollinger Bands"""
        if len(close) < self.bb_period:
            return "NORMAL"
        
        close = close.fillna(close.iloc[-1])
        sma = close.rolling(self.bb_period, min_periods=self.bb_period).mean().iloc[-1]
        std = close.rolling(self.bb_period, min_periods=self.bb_period).std().iloc[-1]
        
        if pd.isna(sma) or pd.isna(std) or sma == 0:
            return "NORMAL"
        
        bb_width = (std * 2) / sma
        
        # FIXED: Simpler volatility classification
        if bb_width > 0.04:
            return "HIGH_VOL"
        elif bb_width < 0.015:
            return "LOW_VOL"
        else:
            return "NORMAL"
    
    def detect_market_condition(self, data_1m: pd.DataFrame, data_15m: pd.DataFrame) -> Dict[str, Any]:
        """FIXED: More permissive market condition detection"""
        if len(data_1m) < 30 or len(data_15m) < 20:  # FIXED: Lower requirements
            return {
                "condition": "WEAK_RANGE",  # FIXED: Default to tradeable condition
                "adx": 20.0, 
                "confidence": 0.7,  # FIXED: Higher default confidence
                "volatility": "NORMAL",
                "timestamp": datetime.now()
            }
        
        adx_15m = self.calculate_adx(data_15m['high'], data_15m['low'], data_15m['close'])
        vol_regime = self.calculate_volatility_regime(data_1m['close'])
        
        # FIXED: More lenient condition classification
        if adx_15m < 18:
            condition, confidence = "STRONG_RANGE", 0.85
        elif adx_15m < 30:  # FIXED: Raised from 25 to 30
            condition, confidence = "WEAK_RANGE", 0.75
        elif adx_15m < 45:  # FIXED: Raised from 40 to 45
            condition, confidence = "TRENDING", 0.8
        else:
            condition, confidence = "STRONG_TREND", 0.9
        
        return {
            "condition": condition,
            "adx": adx_15m,
            "volatility": vol_regime,
            "confidence": confidence,
            "timestamp": datetime.now()
        }

class StrategyManager:
    """FIXED: More active strategy management"""
    
    def __init__(self):
        self.detector = MarketConditionDetector()
        self.current_strategy = None
        self.last_switch_time = None
        self.switch_cooldown = 120  # FIXED: Reduced from 300 to 120 seconds
        self.market_condition = {"condition": "WEAK_RANGE", "adx": 20.0}  # FIXED: Default to tradeable
        
    def should_switch_strategy(self, new_condition: str) -> bool:
        """Determine if strategy should be switched"""
        if not self.current_strategy:
            return True
            
        # FIXED: Shorter cooldown check
        if (self.last_switch_time and 
            (datetime.now() - self.last_switch_time).total_seconds() < self.switch_cooldown):
            return False
            
        # Strategy mapping
        current_type = "RANGE" if self.current_strategy in ["STRONG_RANGE", "WEAK_RANGE"] else "TREND"
        new_type = "RANGE" if new_condition in ["STRONG_RANGE", "WEAK_RANGE"] else "TREND"
        
        return current_type != new_type
    
    def select_strategy(self, data_1m: pd.DataFrame, data_15m: pd.DataFrame) -> Tuple[str, Dict[str, Any]]:
        """Select appropriate strategy based on market conditions"""
        market_info = self.detector.detect_market_condition(data_1m, data_15m)
        condition = market_info["condition"]
        
        # FIXED: Always return a tradeable condition
        if condition == "INSUFFICIENT_DATA":
            market_info["condition"] = "WEAK_RANGE"
            condition = "WEAK_RANGE"
            
        if self.should_switch_strategy(condition):
            self.current_strategy = condition
            self.last_switch_time = datetime.now()
            
        strategy_type = "RANGE" if condition in ["STRONG_RANGE", "WEAK_RANGE"] else "TREND"
        self.market_condition = market_info
        return strategy_type, market_info
    
    def get_position_sizing_multiplier(self, strategy_type: str, market_info: Dict[str, Any]) -> float:
        """Get position sizing multiplier based on strategy and market conditions"""        
        # Base multiplier by strategy and condition
        if strategy_type == "TREND":
            base_multiplier = 1.3 if market_info["condition"] == "STRONG_TREND" else 1.1  # FIXED: Reduced multipliers
        else:
            base_multiplier = 0.9 if market_info["condition"] == "STRONG_RANGE" else 1.0  # FIXED: Less conservative
                
        # Volatility adjustment
        volatility = market_info.get("volatility", "NORMAL")
        vol_multipliers = {"HIGH_VOL": 0.8, "LOW_VOL": 1.1, "NORMAL": 1.0}  # FIXED: Less extreme adjustments
        
        return base_multiplier * vol_multipliers.get(volatility, 1.0)
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get current strategy information"""
        switch_cooldown_remaining = 0
        if self.last_switch_time:
            elapsed = (datetime.now() - self.last_switch_time).total_seconds()
            switch_cooldown_remaining = max(0, self.switch_cooldown - elapsed)
        
        return {
            "current_strategy": self.current_strategy,
            "market_condition": self.market_condition,
            "last_switch": self.last_switch_time,
            "switch_cooldown_remaining": switch_cooldown_remaining
        }