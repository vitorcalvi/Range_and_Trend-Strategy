import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Tuple, Any

class MarketConditionDetector:
    """Detects market conditions using ADX and supporting indicators"""
    
    def __init__(self):
        self.adx_period = 14
        self.bb_period = 20
        
    def calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series) -> float:
        """Calculate ADX using simple, robust method"""
        try:
            if len(close) < 20:
                return 25.0
            
            # Use last 50 periods for calculation
            h = high.iloc[-50:] if len(high) >= 50 else high
            l = low.iloc[-50:] if len(low) >= 50 else low
            c = close.iloc[-50:] if len(close) >= 50 else close
            
            tr, dm_plus, dm_minus = [], [], []
            
            for i in range(1, len(h)):
                # True Range
                tr_val = max(
                    h.iloc[i] - l.iloc[i],
                    abs(h.iloc[i] - c.iloc[i-1]),
                    abs(l.iloc[i] - c.iloc[i-1])
                )
                tr.append(tr_val)
                
                # Directional Movement
                h_move = h.iloc[i] - h.iloc[i-1]
                l_move = l.iloc[i-1] - l.iloc[i]
                
                dm_plus.append(h_move if h_move > l_move and h_move > 0 else 0)
                dm_minus.append(l_move if l_move > h_move and l_move > 0 else 0)
            
            if len(tr) < 14:
                return 25.0
            
            # Simple moving averages
            tr_avg = sum(tr[-14:]) / 14
            dm_plus_avg = sum(dm_plus[-14:]) / 14
            dm_minus_avg = sum(dm_minus[-14:]) / 14
            
            if tr_avg == 0:
                return 25.0
            
            # Calculate DI and DX
            di_plus = 100 * dm_plus_avg / tr_avg
            di_minus = 100 * dm_minus_avg / tr_avg
            di_sum = di_plus + di_minus
            
            if di_sum == 0:
                return 25.0
            
            dx = 100 * abs(di_plus - di_minus) / di_sum
            return max(0, min(100, dx))
            
        except Exception:
            return 25.0
    
    def calculate_volatility_regime(self, close: pd.Series) -> str:
        """Calculate volatility regime using Bollinger Bands"""
        try:
            if len(close) < self.bb_period:
                return "NORMAL"
            
            close = close.fillna(close.iloc[-1])
            sma = close.rolling(self.bb_period, min_periods=self.bb_period).mean()
            std = close.rolling(self.bb_period, min_periods=self.bb_period).std()
            
            current_sma = sma.iloc[-1]
            current_std = std.iloc[-1]
            
            if pd.isna(current_sma) or pd.isna(current_std) or current_sma == 0:
                return "NORMAL"
            
            bb_width = (current_std * 2) / current_sma
            
            if len(std) >= 50:
                avg_width = (std * 2 / sma).rolling(50, min_periods=25).mean().iloc[-1]
                if pd.isna(avg_width) or avg_width == 0:
                    return "NORMAL"
                
                width_ratio = bb_width / avg_width
                if width_ratio > 1.5:
                    return "HIGH_VOL"
                elif width_ratio < 0.5:
                    return "LOW_VOL"
                    
            return "NORMAL"
                
        except Exception:
            return "NORMAL"
    
    def detect_market_condition(self, data_1m: pd.DataFrame, data_15m: pd.DataFrame) -> Dict[str, Any]:
        """Detect current market condition"""
        if len(data_1m) < 50 or len(data_15m) < 30:
            return {"condition": "INSUFFICIENT_DATA", "adx": 25.0, "confidence": 0, "timestamp": datetime.now()}
        
        try:
            adx_15m = self.calculate_adx(data_15m['high'], data_15m['low'], data_15m['close'])
            vol_regime = self.calculate_volatility_regime(data_1m['close'])
            
            if not np.isfinite(adx_15m):
                adx_15m = 25.0
            
            # Determine condition and confidence
            if adx_15m < 20:
                condition, confidence = "STRONG_RANGE", 0.9
            elif adx_15m < 25:
                condition, confidence = "WEAK_RANGE", 0.7
            elif adx_15m < 40:
                condition, confidence = "TRENDING", 0.8
            else:
                condition, confidence = "STRONG_TREND", 0.95
            
            return {
                "condition": condition,
                "adx": adx_15m,
                "volatility": vol_regime,
                "confidence": confidence,
                "timestamp": datetime.now()
            }
            
        except Exception:
            return {
                "condition": "WEAK_RANGE",
                "adx": 25.0,
                "volatility": "NORMAL",
                "confidence": 0.5,
                "timestamp": datetime.now()
            }

class StrategyManager:
    """Manages dual strategy system with automatic switching"""
    
    def __init__(self):
        self.detector = MarketConditionDetector()
        self.current_strategy = None
        self.last_switch_time = None
        self.switch_cooldown = 300  # 5 minutes
        self.market_condition = {"condition": "UNKNOWN", "adx": 25.0}
        
    def should_switch_strategy(self, new_condition: str) -> bool:
        """Determine if strategy should be switched"""
        if not self.current_strategy:
            return True
            
        # Cooldown check
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
        
        if condition == "INSUFFICIENT_DATA":
            return "RANGE", market_info
            
        if self.should_switch_strategy(condition):
            self.current_strategy = condition
            self.last_switch_time = datetime.now()
            
        strategy_type = "RANGE" if condition in ["STRONG_RANGE", "WEAK_RANGE"] else "TREND"
        self.market_condition = market_info
        return strategy_type, market_info
    
    def get_active_timeframe(self, strategy_type: str) -> str:
        """Get the appropriate timeframe for the strategy"""
        return "1m" if strategy_type == "RANGE" else "15m"
    
    def get_position_sizing_multiplier(self, strategy_type: str, market_info: Dict[str, Any]) -> float:
        """Get position sizing multiplier based on strategy and market conditions"""        
        if strategy_type == "TREND":
            base_multiplier = 1.5 if market_info["condition"] == "STRONG_TREND" else 1.2
        else:
            base_multiplier = 0.8 if market_info["condition"] == "STRONG_RANGE" else 1.0
                
        # Adjust for volatility
        volatility = market_info.get("volatility", "NORMAL")
        if volatility == "HIGH_VOL":
            base_multiplier *= 0.7
        elif volatility == "LOW_VOL":
            base_multiplier *= 1.2
            
        return base_multiplier
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get current strategy information"""
        return {
            "current_strategy": self.current_strategy,
            "market_condition": self.market_condition,
            "last_switch": self.last_switch_time,
            "switch_cooldown_remaining": max(0, self.switch_cooldown - 
                (datetime.now() - self.last_switch_time).total_seconds()) 
                if self.last_switch_time else 0
        }