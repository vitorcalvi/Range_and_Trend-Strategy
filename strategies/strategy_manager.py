import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Tuple, Any

class MarketConditionDetector:
    """ULTRA-AGGRESSIVE 3-Minute Market condition detection for stress testing"""
    
    def __init__(self):
        # ULTRA-FAST PARAMETERS FOR STRESS TESTING
        self.adx_period = 5     # Super fast ADX
        self.bb_period = 8      # Very fast BB for volatility
        self.detection_count = 0  # Track detection frequency
        
    def calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series) -> float:
        """Ultra-fast ADX calculation for stress testing"""
        if len(close) < 8:
            return 25.0
        
        # Use only last 12 candles for speed
        h, l, c = high.iloc[-12:], low.iloc[-12:], close.iloc[-12:]
        tr_list, dm_plus_list, dm_minus_list = [], [], []
        
        for i in range(1, len(h)):
            tr_val = max(h.iloc[i] - l.iloc[i], 
                        abs(h.iloc[i] - c.iloc[i-1]), 
                        abs(l.iloc[i] - c.iloc[i-1]))
            tr_list.append(tr_val)
            
            h_move = h.iloc[i] - h.iloc[i-1]
            l_move = l.iloc[i-1] - l.iloc[i]
            
            dm_plus_list.append(h_move if h_move > l_move and h_move > 0 else 0)
            dm_minus_list.append(l_move if l_move > h_move and l_move > 0 else 0)
        
        if len(tr_list) < 5:
            return 25.0
        
        # Ultra-fast 5-period average for stress testing
        tr_avg = sum(tr_list[-5:]) / 5
        dm_plus_avg = sum(dm_plus_list[-5:]) / 5
        dm_minus_avg = sum(dm_minus_list[-5:]) / 5
        
        if tr_avg == 0:
            return 25.0
        
        di_plus = 100 * dm_plus_avg / tr_avg
        di_minus = 100 * dm_minus_avg / tr_avg
        di_sum = di_plus + di_minus
        
        if di_sum == 0:
            return 25.0
        
        dx = 100 * abs(di_plus - di_minus) / di_sum
        return np.clip(dx, 0, 100)
    
    def calculate_volatility_regime(self, close: pd.Series) -> str:
        """Ultra-fast volatility calculation for stress testing"""
        if len(close) < self.bb_period:
            return "NORMAL"
        
        close = close.fillna(close.iloc[-1])
        # Use EMA instead of SMA for faster response
        ema = close.ewm(span=self.bb_period, min_periods=self.bb_period).mean().iloc[-1]
        std = close.rolling(self.bb_period, min_periods=self.bb_period).std().iloc[-1]
        
        if pd.isna(ema) or pd.isna(std) or ema == 0:
            return "NORMAL"
        
        bb_width = (std * 2) / ema
        
        # ULTRA-AGGRESSIVE thresholds for more volatility signals
        if bb_width > 0.015:  # Lower threshold
            return "HIGH_VOL"
        elif bb_width < 0.005:  # Lower threshold
            return "LOW_VOL"
        else:
            return "NORMAL"
    
    def detect_market_condition(self, data_3m_primary: pd.DataFrame, data_3m_secondary: pd.DataFrame) -> Dict[str, Any]:
        """ULTRA-AGGRESSIVE market condition detection for stress testing"""
        self.detection_count += 1
        
        # STRESS TEST: Minimal data requirements
        if len(data_3m_primary) < 10 or len(data_3m_secondary) < 8:
            return {
                "condition": "WEAK_RANGE",
                "adx": 25.0, 
                "confidence": 0.7,
                "volatility": "NORMAL",
                "timestamp": datetime.now(),
                "detection_count": self.detection_count
            }
        
        adx_3m = self.calculate_adx(data_3m_primary['high'], data_3m_primary['low'], data_3m_primary['close'])
        vol_regime = self.calculate_volatility_regime(data_3m_primary['close'])
        
        # ULTRA-AGGRESSIVE thresholds - more likely to detect ranging for more signals
        if adx_3m < 15:  # Lower threshold
            condition, confidence = "STRONG_RANGE", 0.85
        elif adx_3m < 25:  # Lower threshold
            condition, confidence = "WEAK_RANGE", 0.75
        elif adx_3m < 40:  # Lower threshold
            condition, confidence = "TRENDING", 0.8
        else:
            condition, confidence = "STRONG_TREND", 0.9
        
        return {
            "condition": condition,
            "adx": adx_3m,
            "volatility": vol_regime,
            "confidence": confidence,
            "timestamp": datetime.now(),
            "detection_count": self.detection_count,
            "stress_test_mode": True
        }

class StrategyManager:
    """ULTRA-AGGRESSIVE Strategy manager for stress testing"""
    
    def __init__(self):
        self.detector = MarketConditionDetector()
        self.current_strategy = None
        self.last_switch_time = None
        self.last_trade_time = None
        
        # ULTRA-AGGRESSIVE COOLDOWNS FOR STRESS TESTING
        self.switch_cooldown = 60    # Only 1 minute between strategy switches!
        self.trade_cooldown = 10     # Only 10 seconds between trades!
        
        self.market_condition = {"condition": "WEAK_RANGE", "adx": 25.0}
        self.strategy_switches = 0   # Track switch frequency
        self.total_trades_allowed = 0
        
    def should_switch_strategy(self, new_condition: str) -> bool:
        """Ultra-fast strategy switching for stress testing"""
        if not self.current_strategy:
            return True
            
        # STRESS TEST: Much shorter cooldown
        if (self.last_switch_time and 
            (datetime.now() - self.last_switch_time).total_seconds() < self.switch_cooldown):
            return False
            
        current_type = "RANGE" if self.current_strategy in ["STRONG_RANGE", "WEAK_RANGE"] else "TREND"
        new_type = "RANGE" if new_condition in ["STRONG_RANGE", "WEAK_RANGE"] else "TREND"
        
        # ULTRA-AGGRESSIVE: Switch more frequently
        return current_type != new_type or abs(hash(new_condition) % 100) < 5  # 5% random switch chance
    
    def should_allow_new_trade(self) -> bool:
        """Ultra-short trade cooldown for stress testing"""
        if not self.last_trade_time:
            return True
        elapsed = (datetime.now() - self.last_trade_time).total_seconds()
        return elapsed >= self.trade_cooldown  # Only 10 seconds!
    
    def record_trade(self):
        """Record trade with stress test tracking"""
        self.last_trade_time = datetime.now()
        self.total_trades_allowed += 1
    
    def select_strategy(self, data_3m_primary: pd.DataFrame, data_3m_secondary: pd.DataFrame) -> Tuple[str, Dict[str, Any]]:
        """ULTRA-AGGRESSIVE strategy selection for stress testing"""
        market_info = self.detector.detect_market_condition(data_3m_primary, data_3m_secondary)
        condition = market_info["condition"]
        
        if condition == "INSUFFICIENT_DATA":
            market_info["condition"] = "WEAK_RANGE"
            condition = "WEAK_RANGE"
            
        if self.should_switch_strategy(condition):
            self.current_strategy = condition
            self.last_switch_time = datetime.now()
            self.strategy_switches += 1
            
        strategy_type = "RANGE" if condition in ["STRONG_RANGE", "WEAK_RANGE"] else "TREND"
        self.market_condition = market_info
        
        # Add stress test info
        market_info['trade_allowed'] = self.should_allow_new_trade()
        market_info['trade_cooldown_remaining'] = max(0, 
            self.trade_cooldown - (datetime.now() - self.last_trade_time).total_seconds()
            if self.last_trade_time else 0
        )
        market_info['strategy_switches'] = self.strategy_switches
        market_info['trades_allowed'] = self.total_trades_allowed
        market_info['stress_test_mode'] = True
        
        return strategy_type, market_info
    
    def get_position_sizing_multiplier(self, strategy_type: str, market_info: Dict[str, Any]) -> float:
        """ULTRA-AGGRESSIVE position sizing for stress testing"""        
        # STRESS TEST: More aggressive base multipliers
        if strategy_type == "TREND":
            base_multiplier = 1.2 if market_info["condition"] == "STRONG_TREND" else 1.0
        else:
            base_multiplier = 1.0 if market_info["condition"] == "STRONG_RANGE" else 0.9
                
        # Volatility adjustment - less conservative for stress testing
        volatility = market_info.get("volatility", "NORMAL")
        vol_multipliers = {"HIGH_VOL": 0.8, "LOW_VOL": 1.2, "NORMAL": 1.0}  # More aggressive
        
        # STRESS TEST: Boost multiplier for more active trading
        stress_boost = 1.1
        
        return base_multiplier * vol_multipliers.get(volatility, 1.0) * stress_boost
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get ultra-aggressive strategy info with stress test metrics"""
        switch_cooldown_remaining = 0
        if self.last_switch_time:
            elapsed = (datetime.now() - self.last_switch_time).total_seconds()
            switch_cooldown_remaining = max(0, self.switch_cooldown - elapsed)
        
        trade_cooldown_remaining = 0
        if self.last_trade_time:
            elapsed = (datetime.now() - self.last_trade_time).total_seconds()
            trade_cooldown_remaining = max(0, self.trade_cooldown - elapsed)
        
        return {
            "current_strategy": self.current_strategy,
            "market_condition": self.market_condition,
            "last_switch": self.last_switch_time,
            "switch_cooldown_remaining": switch_cooldown_remaining,
            "last_trade": self.last_trade_time,
            "trade_cooldown_remaining": trade_cooldown_remaining,
            "next_trade_allowed": trade_cooldown_remaining == 0,
            "timeframe": "3m",
            "mode": "ULTRA_AGGRESSIVE_STRESS_TEST",
            
            # STRESS TEST METRICS
            "strategy_switches": self.strategy_switches,
            "trades_allowed": self.total_trades_allowed,
            "detection_count": self.detector.detection_count,
            "switch_frequency": f"{self.strategy_switches} switches",
            "trade_frequency": f"{self.total_trades_allowed} trades allowed",
            
            # WARNING
            "warning": "STRESS TEST MODE - EXTREMELY HIGH FREQUENCY",
            "switch_cooldown": f"{self.switch_cooldown}s (ultra-fast)",
            "trade_cooldown": f"{self.trade_cooldown}s (ultra-fast)"
        }