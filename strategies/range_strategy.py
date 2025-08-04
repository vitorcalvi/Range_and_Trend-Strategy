import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional

class RangeStrategy:
    """RSI+Bollinger Bands Range Strategy - Optimized"""
    
    def __init__(self):
        self.config = {
            "rsi_length": 14,  # FIXED: Standard RSI period
            "bb_length": 20,   # FIXED: Bollinger Bands period
            "bb_std": 2.0,     # FIXED: Standard deviation multiplier
            "oversold": 30,    # FIXED: True oversold level
            "overbought": 70,  # FIXED: True overbought level
            "cooldown_seconds": 60,  # FIXED: Reasonable cooldown
            "base_profit_usdt": 35,
            "max_hold_seconds": 300,  # FIXED: 5 minutes max
            "fee_rate": 0.0011
        }
        self.last_signal_time = None
        
    def calculate_rsi(self, prices: pd.Series) -> float:
        """Calculate RSI with standard 14-period"""
        period = self.config['rsi_length']
        if len(prices) < period + 5:
            return 50.0
        
        delta = prices.diff().fillna(0)
        gain = delta.where(delta > 0, 0)
        loss = (-delta.where(delta < 0, 0))
        
        # Use Wilder's smoothing (standard RSI calculation)
        alpha = 1.0 / period
        avg_gain = gain.ewm(alpha=alpha, min_periods=period).mean().iloc[-1]
        avg_loss = loss.ewm(alpha=alpha, min_periods=period).mean().iloc[-1]
        
        if avg_loss == 0:
            return 95.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return np.clip(rsi, 5, 95)
    
    def calculate_bollinger_position(self, prices: pd.Series) -> tuple:
        """Calculate Bollinger Bands and current price position"""
        period = self.config['bb_length']
        std_mult = self.config['bb_std']
        
        if len(prices) < period:
            return 0.5, 0  # Neutral position, no signal
        
        sma = prices.rolling(period).mean().iloc[-1]
        std = prices.rolling(period).std().iloc[-1]
        current_price = prices.iloc[-1]
        
        if pd.isna(sma) or pd.isna(std) or std == 0:
            return 0.5, 0
        
        upper_band = sma + (std * std_mult)
        lower_band = sma - (std * std_mult)
        
        # Calculate position within bands (0 = lower band, 1 = upper band)
        bb_position = (current_price - lower_band) / (upper_band - lower_band)
        bb_position = np.clip(bb_position, 0, 1)
        
        # Calculate band width for volatility filter
        band_width = (upper_band - lower_band) / sma
        
        return bb_position, band_width
    
    def generate_signal(self, data: pd.DataFrame, market_condition: str) -> Optional[Dict]:
        """Generate range trading signals with Bollinger Bands"""
        if len(data) < 25 or self._is_cooldown_active():
            return None
        
        # Only trade in ranging markets
        if market_condition not in ["STRONG_RANGE", "WEAK_RANGE"]:
            return None
        
        rsi = self.calculate_rsi(data['close'])
        bb_position, band_width = self.calculate_bollinger_position(data['close'])
        price = data['close'].iloc[-1]
        
        if pd.isna(rsi) or band_width == 0:
            return None
        
        # FIXED: Removed restrictive volatility filter - let signals through
        # if band_width < 0.02:  # Too restrictive
        #     return None
        
        signal = None
        
        # FIXED: Simplified conditions - price below lower band OR RSI oversold
        # Long signal: Price below lower Bollinger Band OR RSI oversold
        if rsi <= self.config['oversold'] or bb_position <= 0.1:
            signal = self._create_signal('BUY', rsi, bb_position, price, data, market_condition)
        # Short signal: Price above upper Bollinger Band OR RSI overbought  
        elif rsi >= self.config['overbought'] or bb_position >= 0.9:
            signal = self._create_signal('SELL', rsi, bb_position, price, data, market_condition)
        
        if signal:
            self.last_signal_time = datetime.now()
        
        return signal
    
    def _create_signal(self, action: str, rsi: float, bb_position: float, price: float, 
                      data: pd.DataFrame, market_condition: str) -> Dict:
        """Create range trading signal"""
        window = data.tail(20)
        
        if action == 'BUY':
            structure_stop = window['low'].min() * 0.998
            level = window['low'].min()
        else:
            structure_stop = window['high'].max() * 1.002
            level = window['high'].max()
        
        # Validate stop distance
        stop_distance = abs(price - structure_stop) / price
        if not (0.001 <= stop_distance <= 0.008):  # FIXED: Better range
            return None
        
        # FIXED: Enhanced confidence calculation
        rsi_strength = abs(50 - rsi) / 20  # 0-1 scale
        bb_strength = abs(0.5 - bb_position) * 2  # 0-1 scale
        base_confidence = (rsi_strength + bb_strength) * 40 + 60  # 60-100 range
        
        if market_condition == "STRONG_RANGE":
            base_confidence *= 1.1
        
        confidence = np.clip(base_confidence, 65, 95)
        
        return {
            'action': action,
            'strategy': 'RANGE',
            'market_condition': market_condition,
            'rsi': round(rsi, 1),
            'bb_position': round(bb_position, 2),
            'price': price,
            'structure_stop': structure_stop,
            'level': level,
            'signal_type': f"range_{action.lower()}",
            'confidence': round(confidence, 1),
            'base_profit_usdt': self.config['base_profit_usdt'],
            'max_hold_seconds': self.config['max_hold_seconds'],
            'timeframe': '1m'
        }
    
    def _is_cooldown_active(self) -> bool:
        """Check if cooldown is active"""
        if not self.last_signal_time:
            return False
        return (datetime.now() - self.last_signal_time).total_seconds() < self.config['cooldown_seconds']
    
    def get_strategy_info(self) -> Dict:
        """Get strategy information"""
        return {
            'name': 'RSI+Bollinger Range Strategy (Optimized)',
            'type': 'RANGE',
            'timeframe': '1m',
            'config': self.config,
            'description': f'RSI({self.config["rsi_length"]}) + BB({self.config["bb_length"]}) mean reversion - ${self.config["base_profit_usdt"]} target'
        }