import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Tuple, Any

class MarketConditionDetector:
    """ULTRA-AGGRESSIVE: Market condition detection with ADX >15 sensitivity"""
    
    def __init__(self):
        # ULTRA-SENSITIVE thresholds
        self.adx_period = 14
        self.bb_period = 15  # Faster BB for 3m charts
        
        # RESEARCH: Ultra-sensitive ADX thresholds
        self.adx_thresholds = {
            'ultra_sensitive': 15,    # RESEARCH: ADX >15 for earliest detection
            'strong_trend': 20,       # RESEARCH: ADX >20 standard confirmation
            'very_strong': 25,        # Original threshold maintained
            'extreme_trend': 40       # Reduce position size threshold
        }
        
        # Volatility triggers for position sizing
        self.volatility_triggers = {
            'enable_trading': 0.6,    # Execute when volatility >0.6x average
            'increase_size': 1.2,     # Increase when volatility >1.2x average  
            'reduce_size': 2.5,       # Reduce when volatility >2.5x average
            'halt_trading': 4.0       # Halt when volatility >4x average
        }
        
    def calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series) -> float:
        """Ultra-sensitive ADX calculation optimized for 3m timeframe"""
        if len(close) < 18:  # Reduced minimum data requirement
            return self.adx_thresholds['ultra_sensitive']  # Default to 15
        
        h = high.iloc[-25:] if len(high) >= 25 else high  # Reduced window
        l = low.iloc[-25:] if len(low) >= 25 else low
        c = close.iloc[-25:] if len(close) >= 25 else close
        
        tr_list, dm_plus_list, dm_minus_list = [], [], []
        
        for i in range(1, len(h)):
            tr_val = max(
                h.iloc[i] - l.iloc[i],
                abs(h.iloc[i] - c.iloc[i-1]),
                abs(l.iloc[i] - c.iloc[i-1])
            )
            tr_list.append(tr_val)
            
            h_move = h.iloc[i] - h.iloc[i-1]
            l_move = l.iloc[i-1] - l.iloc[i]
            
            dm_plus_list.append(h_move if h_move > l_move and h_move > 0 else 0)
            dm_minus_list.append(l_move if l_move > h_move and l_move > 0 else 0)
        
        if len(tr_list) < 12:  # Reduced minimum
            return self.adx_thresholds['ultra_sensitive']
        
        # Use shorter smoothing period for faster response
        smoothing_period = min(10, len(tr_list))  # Reduced from 14
        tr_avg = sum(tr_list[-smoothing_period:]) / smoothing_period
        dm_plus_avg = sum(dm_plus_list[-smoothing_period:]) / smoothing_period
        dm_minus_avg = sum(dm_minus_list[-smoothing_period:]) / smoothing_period
        
        if tr_avg == 0:
            return self.adx_thresholds['ultra_sensitive']
        
        di_plus = 100 * dm_plus_avg / tr_avg
        di_minus = 100 * dm_minus_avg / tr_avg
        di_sum = di_plus + di_minus
        
        if di_sum == 0:
            return self.adx_thresholds['ultra_sensitive']
        
        dx = 100 * abs(di_plus - di_minus) / di_sum
        
        # Apply sensitivity boost for ultra-aggressive detection
        sensitivity_multiplier = 1.15  # 15% boost for earlier detection
        adjusted_dx = dx * sensitivity_multiplier
        
        return np.clip(adjusted_dx, 0, 100)
    
    def calculate_volatility_regime(self, close: pd.Series) -> str:
        """Calculate volatility using faster BB(15) for 3m timeframe"""
        if len(close) < self.bb_period:
            return "NORMAL"
        
        close = close.fillna(close.iloc[-1])
        sma = close.rolling(self.bb_period, min_periods=self.bb_period).mean().iloc[-1]
        std = close.rolling(self.bb_period, min_periods=self.bb_period).std().iloc[-1]
        
        if pd.isna(sma) or pd.isna(std) or sma == 0:
            return "NORMAL"
        
        bb_width = (std * 2) / sma
        
        # Ultra-aggressive volatility thresholds (more sensitive)
        if bb_width > 0.035:      # Reduced from 0.04
            return "HIGH_VOL"
        elif bb_width < 0.012:    # Reduced from 0.015  
            return "LOW_VOL"
        else:
            return "NORMAL"
    
    def calculate_volatility_multiplier(self, close: pd.Series) -> float:
        """Calculate volatility multiplier for position sizing"""
        if len(close) < 10:
            return 1.0
        
        current_vol = close.rolling(10).std().iloc[-1] / close.rolling(10).mean().iloc[-1]
        avg_vol = close.rolling(30).std().mean() / close.rolling(30).mean().mean() if len(close) >= 30 else current_vol
        
        if avg_vol == 0:
            return 1.0
        
        vol_ratio = current_vol / avg_vol
        
        # Apply volatility triggers
        if vol_ratio >= self.volatility_triggers['halt_trading']:
            return 0.0  # Halt trading
        elif vol_ratio >= self.volatility_triggers['reduce_size']:
            return 0.5  # Reduce size
        elif vol_ratio >= self.volatility_triggers['increase_size']:
            return 1.3  # Increase size
        elif vol_ratio >= self.volatility_triggers['enable_trading']:
            return 1.0  # Normal size
        else:
            return 0.7  # Reduce size in low volatility
    
    def detect_market_condition(self, data_1m: pd.DataFrame, data_15m: pd.DataFrame) -> Dict[str, Any]:
        """ULTRA-AGGRESSIVE: Market condition detection with ADX >15"""
        if len(data_1m) < 20 or len(data_15m) < 15:  # Reduced requirements
            return {
                "condition": "WEAK_RANGE",
                "adx": self.adx_thresholds['ultra_sensitive'],
                "confidence": 0.7,
                "volatility": "NORMAL",
                "volatility_multiplier": 1.0,
                "timestamp": datetime.now()
            }
        
        # Use 3m equivalent data (simulate by using more recent 1m data)
        recent_1m = data_1m.tail(40)  # Approximate 3m worth of 1m data
        
        adx_15m = self.calculate_adx(data_15m['high'], data_15m['low'], data_15m['close'])
        adx_1m = self.calculate_adx(recent_1m['high'], recent_1m['low'], recent_1m['close'])
        
        # Use average of both timeframes for better accuracy
        combined_adx = (adx_15m * 0.7 + adx_1m * 0.3)  # Weight 15m more heavily
        
        vol_regime = self.calculate_volatility_regime(recent_1m['close'])
        vol_multiplier = self.calculate_volatility_multiplier(recent_1m['close'])
        
        # ULTRA-AGGRESSIVE thresholds (much lower)
        if combined_adx < self.adx_thresholds['ultra_sensitive']:     # < 15
            condition, confidence = "STRONG_RANGE", 0.85
        elif combined_adx < self.adx_thresholds['strong_trend']:      # < 20
            condition, confidence = "WEAK_RANGE", 0.75
        elif combined_adx < self.adx_thresholds['very_strong']:       # < 25
            condition, confidence = "TRENDING", 0.8
        else:  # >= 25
            condition, confidence = "STRONG_TREND", 0.9
        
        # Adjust confidence based on volatility
        if vol_regime == "HIGH_VOL":
            confidence *= 1.1  # Higher confidence in high volatility
        elif vol_regime == "LOW_VOL":
            confidence *= 0.9  # Lower confidence in low volatility
        
        confidence = np.clip(confidence, 0.6, 1.0)
        
        return {
            "condition": condition,
            "adx": combined_adx,
            "adx_15m": adx_15m,
            "adx_1m": adx_1m,
            "volatility": vol_regime,
            "volatility_multiplier": vol_multiplier,
            "confidence": confidence,
            "timestamp": datetime.now()
        }

class StrategyManager:
    """ULTRA-AGGRESSIVE: Strategy manager with reduced cooldowns and high frequency"""
    
    def __init__(self):
        self.detector = MarketConditionDetector()
        self.current_strategy = None
        self.last_switch_time = None
        self.last_trade_time = None
        
        # ULTRA-AGGRESSIVE cooldowns (much shorter)
        self.switch_cooldown = 180      # 3 minutes between strategy switches (was 300)
        self.trade_cooldown = 120       # 2 minutes between trades (was 600)
        self.high_frequency_mode = True  # Enable ultra-aggressive mode
        
        # Performance tracking for optimization
        self.signal_frequency_target = 10  # 8-12 signals per hour target
        self.last_hour_signals = []
        
        self.market_condition = {"condition": "WEAK_RANGE", "adx": 15.0}
        
    def should_switch_strategy(self, new_condition: str) -> bool:
        """ULTRA-AGGRESSIVE: Check strategy switch with reduced cooldown"""
        if not self.current_strategy:
            return True
            
        # Ultra-aggressive cooldown (3 minutes)
        if (self.last_switch_time and 
            (datetime.now() - self.last_switch_time).total_seconds() < self.switch_cooldown):
            return False
            
        current_type = "RANGE" if self.current_strategy in ["STRONG_RANGE", "WEAK_RANGE"] else "TREND"
        new_type = "RANGE" if new_condition in ["STRONG_RANGE", "WEAK_RANGE"] else "TREND"
        
        return current_type != new_type
    
    def should_allow_new_trade(self) -> bool:
        """ULTRA-AGGRESSIVE: Check trade cooldown (2 minutes)"""
        if not self.last_trade_time:
            return True
            
        elapsed = (datetime.now() - self.last_trade_time).total_seconds()
        return elapsed >= self.trade_cooldown
    
    def record_trade(self):
        """Record trade and update signal frequency tracking"""
        now = datetime.now()
        self.last_trade_time = now
        
        # Track signals for frequency analysis
        self.last_hour_signals.append(now)
        
        # Clean old signals (keep only last hour)
        cutoff = now - timedelta(hours=1)
        self.last_hour_signals = [sig_time for sig_time in self.last_hour_signals if sig_time >= cutoff]
    
    def get_signal_frequency(self) -> Dict[str, Any]:
        """Get current signal frequency statistics"""
        now = datetime.now()
        cutoff = now - timedelta(hours=1)
        
        # Count signals in last hour
        recent_signals = [sig_time for sig_time in self.last_hour_signals if sig_time >= cutoff]
        signals_per_hour = len(recent_signals)
        
        # Calculate frequency metrics
        target_min, target_max = 8, 12
        frequency_score = 100 if target_min <= signals_per_hour <= target_max else 0
        
        if signals_per_hour < target_min:
            frequency_score = (signals_per_hour / target_min) * 100
        elif signals_per_hour > target_max:
            frequency_score = max(0, 100 - ((signals_per_hour - target_max) * 10))
        
        return {
            'signals_last_hour': signals_per_hour,
            'target_range': f'{target_min}-{target_max}',
            'frequency_score': round(frequency_score, 1),
            'status': 'OPTIMAL' if target_min <= signals_per_hour <= target_max else 
                     'TOO_LOW' if signals_per_hour < target_min else 'TOO_HIGH'
        }
    
    def select_strategy(self, data_1m: pd.DataFrame, data_15m: pd.DataFrame) -> Tuple[str, Dict[str, Any]]:
        """ULTRA-AGGRESSIVE: Strategy selection with enhanced market detection"""
        market_info = self.detector.detect_market_condition(data_1m, data_15m)
        condition = market_info["condition"]
        
        if condition == "INSUFFICIENT_DATA":
            market_info["condition"] = "WEAK_RANGE"
            condition = "WEAK_RANGE"
            
        if self.should_switch_strategy(condition):
            self.current_strategy = condition
            self.last_switch_time = datetime.now()
            
        strategy_type = "RANGE" if condition in ["STRONG_RANGE", "WEAK_RANGE"] else "TREND"
        self.market_condition = market_info
        
        # Add ultra-aggressive specific info
        market_info['trade_allowed'] = self.should_allow_new_trade()
        market_info['trade_cooldown_remaining'] = max(0, 
            self.trade_cooldown - (datetime.now() - self.last_trade_time).total_seconds()
            if self.last_trade_time else 0
        )
        market_info['signal_frequency'] = self.get_signal_frequency()
        market_info['high_frequency_mode'] = self.high_frequency_mode
        
        return strategy_type, market_info
    
    def get_position_sizing_multiplier(self, strategy_type: str, market_info: Dict[str, Any]) -> float:
        """ULTRA-AGGRESSIVE: Dynamic position sizing with volatility adaptation"""
        
        # Base multipliers (reduced for higher frequency)
        if strategy_type == "TREND":
            base_multiplier = 0.8 if market_info["condition"] == "STRONG_TREND" else 0.65
        else:
            base_multiplier = 0.6 if market_info["condition"] == "STRONG_RANGE" else 0.5
                
        # Apply volatility multiplier from detector
        vol_multiplier = market_info.get("volatility_multiplier", 1.0)
        
        # Additional volatility adjustment
        volatility = market_info.get("volatility", "NORMAL")
        vol_adjustments = {"HIGH_VOL": 0.7, "LOW_VOL": 1.1, "NORMAL": 1.0}
        vol_adjustment = vol_adjustments.get(volatility, 1.0)
        
        # Signal frequency adjustment (reduce size if too many signals)
        freq_info = market_info.get('signal_frequency', {})
        freq_status = freq_info.get('status', 'OPTIMAL')
        
        if freq_status == 'TOO_HIGH':
            freq_adjustment = 0.8  # Reduce size if over-trading
        elif freq_status == 'TOO_LOW':
            freq_adjustment = 1.1  # Slightly increase if under-trading
        else:
            freq_adjustment = 1.0
        
        final_multiplier = base_multiplier * vol_multiplier * vol_adjustment * freq_adjustment
        
        # Ensure reasonable bounds for ultra-aggressive system
        return np.clip(final_multiplier, 0.3, 1.5)
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get ultra-aggressive strategy info with enhanced metrics"""
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
            "signal_frequency": self.get_signal_frequency(),
            "high_frequency_mode": self.high_frequency_mode,
            "adx_threshold": self.detector.adx_thresholds['ultra_sensitive'],
            "ultra_aggressive_features": {
                "adx_sensitivity": "15+ (ultra-sensitive)",
                "switch_cooldown": f"{self.switch_cooldown//60}min",
                "trade_cooldown": f"{self.trade_cooldown//60}min",
                "target_signals_hour": "8-12",
                "volatility_adaptation": "enabled",
                "frequency_optimization": "enabled"
            }
        }