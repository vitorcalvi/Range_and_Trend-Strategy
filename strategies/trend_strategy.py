import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional

class TrendStrategy:
    """Simplified Trend Strategy with correct calculations"""
    
    def __init__(self):
        self.config = {
            # Core indicators
            "rsi_length": 6,
            "fast_ema": 9,       # More reasonable for 3m
            "slow_ema": 21,      # Standard EMA periods
            
            # Entry conditions
            "uptrend_rsi_low": 40,
            "uptrend_rsi_high": 80,
            "downtrend_rsi_low": 20,
            "downtrend_rsi_high": 60,
            
            # Trading parameters
            "min_confidence": 65,
            "cooldown_seconds": 180,
            "risk_reward_ratio": 2.5,  # Higher R/R for trends
            "min_trend_strength": 0.002,  # 0.2% minimum EMA difference
        }
        self.last_signal_time = None
    
    def calculate_rsi(self, prices: pd.Series) -> float:
        """Calculate RSI"""
        period = self.config['rsi_length']
        if len(prices) < period + 1:
            return 50.0
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        if loss.iloc[-1] == 0:
            return 100.0
        
        rs = gain.iloc[-1] / loss.iloc[-1]
        rsi = 100 - (100 / (1 + rs))
        return np.clip(rsi, 0, 100)
    
    def calculate_emas(self, prices: pd.Series) -> tuple:
        """Calculate EMAs and trend direction"""
        if len(prices) < self.config['slow_ema']:
            return prices.iloc[-1], prices.iloc[-1], 'NEUTRAL'
        
        fast_ema = prices.ewm(span=self.config['fast_ema'], adjust=False).mean().iloc[-1]
        slow_ema = prices.ewm(span=self.config['slow_ema'], adjust=False).mean().iloc[-1]
        
        # Calculate trend strength
        ema_diff_pct = (fast_ema - slow_ema) / slow_ema if slow_ema > 0 else 0
        
        if ema_diff_pct > self.config['min_trend_strength']:
            trend = 'UPTREND'
        elif ema_diff_pct < -self.config['min_trend_strength']:
            trend = 'DOWNTREND'
        else:
            trend = 'NEUTRAL'
        
        return fast_ema, slow_ema, trend
    
    def generate_signal(self, data: pd.DataFrame, market_condition: str) -> Optional[Dict]:
        """Generate trading signal"""
        # Check cooldown
        if self.last_signal_time:
            if (datetime.now() - self.last_signal_time).total_seconds() < self.config['cooldown_seconds']:
                return None
        
        # Only trade in trending markets
        if market_condition not in ["TRENDING", "STRONG_TREND"]:
            return None
        
        if len(data) < self.config['slow_ema'] + 1:
            return None
        
        close = data['close']
        rsi = self.calculate_rsi(close)
        fast_ema, slow_ema, trend = self.calculate_emas(close)
        
        if trend == 'NEUTRAL':
            return None
        
        signal = None
        current_price = close.iloc[-1]
        
        # Buy signal: Uptrend + RSI pullback
        if trend == 'UPTREND':
            if self.config['uptrend_rsi_low'] <= rsi <= self.config['uptrend_rsi_high']:
                stop_price = min(slow_ema * 0.97, current_price * 0.97)  # 3% stop
                confidence = self._calculate_confidence(rsi, trend, fast_ema, slow_ema)
                if confidence >= self.config['min_confidence']:
                    signal = self._create_signal('BUY', current_price, stop_price, confidence, 
                                                rsi, fast_ema, slow_ema, trend)
        
        # Sell signal: Downtrend + RSI bounce
        elif trend == 'DOWNTREND':
            if self.config['downtrend_rsi_low'] <= rsi <= self.config['downtrend_rsi_high']:
                stop_price = max(slow_ema * 1.03, current_price * 1.03)  # 3% stop
                confidence = self._calculate_confidence(rsi, trend, fast_ema, slow_ema)
                if confidence >= self.config['min_confidence']:
                    signal = self._create_signal('SELL', current_price, stop_price, confidence,
                                                rsi, fast_ema, slow_ema, trend)
        
        if signal:
            self.last_signal_time = datetime.now()
        
        return signal
    
    def _calculate_confidence(self, rsi: float, trend: str, fast_ema: float, slow_ema: float) -> float:
        """Calculate signal confidence"""
        # Trend strength component
        ema_diff = abs(fast_ema - slow_ema) / slow_ema if slow_ema > 0 else 0
        trend_strength = min(50, ema_diff * 1000)  # Scale to 0-50
        
        # RSI component
        if trend == 'UPTREND':
            rsi_strength = max(0, (70 - rsi) / 30) * 50 if rsi < 70 else 25
        else:
            rsi_strength = max(0, (rsi - 30) / 30) * 50 if rsi > 30 else 25
        
        return min(95, 45 + trend_strength + rsi_strength)
    
    def _create_signal(self, action: str, price: float, stop_price: float, confidence: float,
                      rsi: float, fast_ema: float, slow_ema: float, trend: str) -> Dict:
        """Create signal dictionary"""
        return {
            'action': action,
            'strategy': 'TREND',
            'price': price,
            'structure_stop': stop_price,
            'confidence': round(confidence, 1),
            'rsi': round(rsi, 1),
            'fast_ema': round(fast_ema, 2),
            'slow_ema': round(slow_ema, 2),
            'trend': trend,
            'risk_reward_ratio': self.config['risk_reward_ratio'],
            'timeframe': '3m'
        }