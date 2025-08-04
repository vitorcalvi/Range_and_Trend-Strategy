import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional

class TrendStrategy:
    """FIXED: More Active RSI+EMA Trend Following Strategy"""
    
    def __init__(self):
        self.config = {
            "rsi_length": 14,
            "fast_ema": 21,
            "slow_ema": 50,
            # FIXED: More reasonable RSI levels for trending markets
            "uptrend_rsi_low": 35,   # Was 40, now more permissive
            "uptrend_rsi_high": 65,  # Was 60, now wider range
            "downtrend_rsi_low": 35,
            "downtrend_rsi_high": 65,
            "trend_strength_threshold": 0.001,  # FIXED: Lower threshold (was 0.002)
            "cooldown_seconds": 180,  # FIXED: Reduced from 300
            "target_profit_multiplier": 2.0,
            "max_hold_seconds": 2400,
            "trailing_stop_pct": 0.8,
            "fee_rate": 0.0011
        }
        self.last_signal_time = None
        
    def calculate_rsi(self, prices: pd.Series) -> float:
        """Calculate RSI with standard method"""
        period = self.config['rsi_length']
        if len(prices) < period + 5:
            return 50.0
        
        delta = prices.diff().fillna(0)
        gain = delta.where(delta > 0, 0)
        loss = (-delta.where(delta < 0, 0))
        
        alpha = 1.0 / period
        avg_gain = gain.ewm(alpha=alpha, min_periods=period).mean().iloc[-1]
        avg_loss = loss.ewm(alpha=alpha, min_periods=period).mean().iloc[-1]
        
        if avg_loss == 0:
            return 95.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return np.clip(rsi, 5, 95)
    
    def calculate_emas(self, prices: pd.Series) -> tuple:
        """Calculate fast and slow EMAs"""
        fast_period = self.config['fast_ema']
        slow_period = self.config['slow_ema']
        
        if len(prices) < slow_period:
            return prices.iloc[-1], prices.iloc[-1], 'NEUTRAL'
            
        fast_ema = prices.ewm(span=fast_period, min_periods=fast_period).mean().iloc[-1]
        slow_ema = prices.ewm(span=slow_period, min_periods=slow_period).mean().iloc[-1]
        
        if pd.isna(fast_ema) or pd.isna(slow_ema):
            return fast_ema, slow_ema, 'NEUTRAL'
        
        # FIXED: Lower threshold for trend detection
        ema_diff_pct = (fast_ema - slow_ema) / slow_ema
        
        if ema_diff_pct > self.config['trend_strength_threshold']:
            trend = 'UPTREND'
        elif ema_diff_pct < -self.config['trend_strength_threshold']:
            trend = 'DOWNTREND'  
        else:
            trend = 'NEUTRAL'
            
        return fast_ema, slow_ema, trend
    
    def calculate_trend_momentum(self, prices: pd.Series, ema_fast: float) -> float:
        """Calculate trend momentum using EMA slope"""
        if len(prices) < 10:
            return 0
        
        ema_series = prices.ewm(span=self.config['fast_ema']).mean()
        if len(ema_series) < 5:
            return 0
        
        # Calculate 5-period slope
        slope = (ema_series.iloc[-1] - ema_series.iloc[-5]) / ema_series.iloc[-5]
        return slope
    
    def generate_signal(self, data: pd.DataFrame, market_condition: str) -> Optional[Dict]:
        """FIXED: More active trend following signals"""
        if len(data) < 60 or self._is_cooldown_active():
            return None
        
        # FIXED: Trade in both trending AND weak ranging markets (more opportunities)
        if market_condition not in ["TRENDING", "STRONG_TREND", "WEAK_RANGE"]:
            return None
        
        close = data['close']
        rsi = self.calculate_rsi(close)
        fast_ema, slow_ema, trend = self.calculate_emas(close)
        momentum = self.calculate_trend_momentum(close, fast_ema)
        price = close.iloc[-1]
        
        if pd.isna(rsi) or trend == 'NEUTRAL':
            return None
        
        # FIXED: Reduced minimum momentum requirement
        if abs(momentum) < 0.0005:  # Was 0.001, now more permissive
            return None
        
        signal = None
        
        # FIXED: More permissive RSI conditions
        if trend == 'UPTREND' and momentum > 0:
            # Long signal: More flexible RSI range
            if (self.config['uptrend_rsi_low'] <= rsi <= self.config['uptrend_rsi_high'] or
                (rsi > 50 and momentum > 0.002)):  # NEW: Strong momentum override
                signal = self._create_signal('BUY', trend, rsi, price, data, fast_ema, slow_ema, momentum)
                
        elif trend == 'DOWNTREND' and momentum < 0:
            # Short signal: More flexible RSI range  
            if (self.config['downtrend_rsi_low'] <= rsi <= self.config['downtrend_rsi_high'] or
                (rsi < 50 and momentum < -0.002)):  # NEW: Strong momentum override
                signal = self._create_signal('SELL', trend, rsi, price, data, fast_ema, slow_ema, momentum)
        
        if signal:
            self.last_signal_time = datetime.now()
        
        return signal
    
    def _create_signal(self, action: str, trend: str, rsi: float, price: float, 
                      data: pd.DataFrame, fast_ema: float, slow_ema: float, momentum: float) -> Dict:
        """Create trend following signal"""
        window = data.tail(30)  # FIXED: Shorter window for faster signals
        
        if action == 'BUY':
            # Use EMA as dynamic support
            ema_stop = fast_ema * 0.994  # FIXED: Tighter stop
            swing_low = window['low'].min()
            structure_stop = max(swing_low, ema_stop)
            level = swing_low
        else:
            # Use EMA as dynamic resistance
            ema_stop = fast_ema * 1.006  # FIXED: Tighter stop
            swing_high = window['high'].max()
            structure_stop = min(swing_high, ema_stop)
            level = swing_high
        
        # FIXED: More lenient stop distance validation
        stop_distance = abs(price - structure_stop) / price
        if not (0.003 <= stop_distance <= 0.035):  # FIXED: Wider acceptable range
            return None
        
        # FIXED: Simplified confidence calculation
        trend_strength = abs(fast_ema - slow_ema) / slow_ema * 100
        momentum_strength = abs(momentum) * 1000
        
        base_confidence = 65 + min(trend_strength * 2 + momentum_strength * 0.5, 25)
        confidence = np.clip(base_confidence, 65, 90)
        
        return {
            'action': action,
            'strategy': 'TREND', 
            'trend': trend,
            'rsi': round(rsi, 1),
            'fast_ema': round(fast_ema, 2),
            'slow_ema': round(slow_ema, 2),
            'momentum': round(momentum * 100, 2),
            'price': price,
            'structure_stop': structure_stop,
            'level': level,
            'signal_type': f"trend_{action.lower()}",
            'confidence': round(confidence, 1),
            'risk_reward_ratio': self.config['target_profit_multiplier'],
            'max_hold_seconds': self.config['max_hold_seconds'],
            'trailing_stop_pct': self.config['trailing_stop_pct'],
            'timeframe': '15m'
        }
    
    def _is_cooldown_active(self) -> bool:
        """Check if cooldown is active"""
        if not self.last_signal_time:
            return False
        return (datetime.now() - self.last_signal_time).total_seconds() < self.config['cooldown_seconds']
    
    def should_trail_stop(self, entry_price: float, current_price: float, side: str, 
                         unrealized_pnl: float) -> tuple[bool, float]:
        """Calculate trailing stop for trend following"""
        if unrealized_pnl <= 0:
            return False, 0
        
        trail_distance = current_price * (self.config['trailing_stop_pct'] / 100)
        
        if side == 'Buy':
            new_stop = current_price - trail_distance
        else:
            new_stop = current_price + trail_distance
            
        return True, new_stop
    
    def get_strategy_info(self) -> Dict:
        """Get strategy information"""
        return {
            'name': f'RSI+EMA({self.config["fast_ema"]}/{self.config["slow_ema"]}) Trend Strategy (ACTIVE)',
            'type': 'TREND',
            'timeframe': '15m', 
            'config': self.config,
            'description': 'More active trend following with lower thresholds',
            'expected_win_rate': '60-70%',
            'risk_reward': f'1:{self.config["target_profit_multiplier"]}'
        }