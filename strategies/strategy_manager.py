import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Tuple, Any

class MarketConditionDetector:
    """Simplified market condition detection"""
    
    def __init__(self):
        self.adx_period = 14
        self.adx_threshold = 20  # Standard threshold for 3m testing
    
    def calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series) -> float:
        """Calculate ADX indicator"""
        if len(close) < self.adx_period + 1:
            return 20.0
        
        # Calculate True Range
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        
        # Calculate directional movements
        up_move = high - high.shift()
        down_move = low.shift() - low
        
        pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smooth the values
        tr_smooth = pd.Series(tr).rolling(self.adx_period).mean()
        pos_dm_smooth = pd.Series(pos_dm).rolling(self.adx_period).mean()
        neg_dm_smooth = pd.Series(neg_dm).rolling(self.adx_period).mean()
        
        # Calculate DI+ and DI-
        pos_di = 100 * pos_dm_smooth / tr_smooth
        neg_di = 100 * neg_dm_smooth / tr_smooth
        
        # Calculate DX and ADX
        dx = 100 * (pos_di - neg_di).abs() / (pos_di + neg_di)
        adx = dx.rolling(self.adx_period).mean().iloc[-1]
        
        return adx if not pd.isna(adx) else 20.0
    
    def detect_market_condition(self, data_3m: pd.DataFrame) -> Dict[str, Any]:
        """Detect market condition from 3m data"""
        if len(data_3m) < 30:
            return {
                "condition": "INSUFFICIENT_DATA",
                "adx": 0,
                "confidence": 0.5,
                "volatility": "NORMAL"
            }
        
        adx = self.calculate_adx(data_3m['high'], data_3m['low'], data_3m['close'])
        
        # Simplified volatility calculation
        returns = data_3m['close'].pct_change().dropna()
        volatility = returns.std()
        
        if volatility > 0.02:
            vol_regime = "HIGH_VOL"
        elif volatility < 0.005:
            vol_regime = "LOW_VOL"
        else:
            vol_regime = "NORMAL"
        
        # Determine market condition
        if adx < self.adx_threshold:
            condition = "STRONG_RANGE"
            confidence = 0.8
        elif adx < 30:
            condition = "WEAK_RANGE"
            confidence = 0.7
        elif adx < 40:
            condition = "TRENDING"
            confidence = 0.8
        else:
            condition = "STRONG_TREND"
            confidence = 0.9
        
        return {
            "condition": condition,
            "adx": round(adx, 1),
            "confidence": confidence,
            "volatility": vol_regime
        }

class StrategyManager:
    """Simplified strategy manager"""
    
    def __init__(self):
        self.detector = MarketConditionDetector()
        self.current_strategy = None
        self.last_switch_time = None
        self.last_trade_time = None
        self.switch_cooldown = 300  # 5 minutes
        self.trade_cooldown = 120   # 2 minutes
    
    def select_strategy(self, data_3m: pd.DataFrame) -> Tuple[str, Dict[str, Any]]:
        """Select strategy based on market condition"""
        market_info = self.detector.detect_market_condition(data_3m)
        condition = market_info["condition"]
        
        # Determine strategy type
        if condition in ["STRONG_RANGE", "WEAK_RANGE"]:
            strategy_type = "RANGE"
        else:
            strategy_type = "TREND"
        
        # Check if we should switch
        if self.current_strategy != strategy_type:
            if self.last_switch_time:
                elapsed = (datetime.now() - self.last_switch_time).total_seconds()
                if elapsed < self.switch_cooldown:
                    strategy_type = self.current_strategy  # Keep current
                else:
                    self.last_switch_time = datetime.now()
            else:
                self.last_switch_time = datetime.now()
            
            self.current_strategy = strategy_type
        
        # Add trade cooldown info
        market_info['trade_allowed'] = True
        if self.last_trade_time:
            elapsed = (datetime.now() - self.last_trade_time).total_seconds()
            if elapsed < self.trade_cooldown:
                market_info['trade_allowed'] = False
                market_info['cooldown_remaining'] = self.trade_cooldown - elapsed
        
        return strategy_type, market_info
    
    def record_trade(self):
        """Record trade time for cooldown"""
        self.last_trade_time = datetime.now()
    
    def get_position_sizing_multiplier(self, market_info: Dict[str, Any]) -> float:
        """Get position sizing multiplier based on market conditions"""
        base_multiplier = 1.0
        
        # Adjust for volatility
        if market_info['volatility'] == "HIGH_VOL":
            base_multiplier *= 0.7
        elif market_info['volatility'] == "LOW_VOL":
            base_multiplier *= 1.1
        
        # Adjust for confidence
        base_multiplier *= market_info['confidence']
        
        return np.clip(base_multiplier, 0.5, 1.5)