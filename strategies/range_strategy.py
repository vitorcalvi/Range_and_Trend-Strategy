import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional

class RangeStrategy:
    """Simplified Range Strategy with correct calculations"""
    
    def __init__(self):
        self.config = {
            # Core indicators
            "rsi_length": 6,
            "bb_length": 20,     # Standard BB period
            "bb_std": 2.0,       # Standard deviation
            "oversold": 30,      # Standard oversold
            "overbought": 70,    # Standard overbought
            
            # Trading parameters
            "min_confidence": 65,
            "cooldown_seconds": 120,
            "risk_reward_ratio": 2.0,  # Need 2R for profitability
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
    
    def calculate_bollinger_bands(self, prices: pd.Series) -> tuple:
        """Calculate Bollinger Bands position"""
        period = self.config['bb_length']
        if len(prices) < period:
            return 0.5, 0, 0, 0
        
        sma = prices.rolling(period).mean().iloc[-1]
        std = prices.rolling(period).std().iloc[-1]
        current_price = prices.iloc[-1]
        
        upper_band = sma + (std * self.config['bb_std'])
        lower_band = sma - (std * self.config['bb_std'])
        
        if upper_band == lower_band:
            return 0.5, 0, upper_band, lower_band
        
        bb_position = (current_price - lower_band) / (upper_band - lower_band)
        band_width = (upper_band - lower_band) / sma if sma > 0 else 0
        
        return np.clip(bb_position, 0, 1), band_width, upper_band, lower_band
    
    def generate_signal(self, data: pd.DataFrame, market_condition: str) -> Optional[Dict]:
        """Generate trading signal"""
        # Check cooldown
        if self.last_signal_time:
            if (datetime.now() - self.last_signal_time).total_seconds() < self.config['cooldown_seconds']:
                return None
        
        # Only trade in ranging markets
        if market_condition not in ["STRONG_RANGE", "WEAK_RANGE"]:
            return None
        
        if len(data) < max(self.config['rsi_length'], self.config['bb_length']) + 1:
            return None
        
        close = data['close']
        rsi = self.calculate_rsi(close)
        bb_position, band_width, upper_band, lower_band = self.calculate_bollinger_bands(close)
        
        # Skip if bands too narrow
        if band_width < 0.01:  # Less than 1% width
            return None
        
        signal = None
        current_price = close.iloc[-1]
        
        # Buy signal: Oversold + near lower band
        if rsi <= self.config['oversold'] and bb_position <= 0.2:
            stop_price = lower_band * 0.97  # 3% below lower band
            confidence = self._calculate_confidence(rsi, bb_position, True)
            if confidence >= self.config['min_confidence']:
                signal = self._create_signal('BUY', current_price, stop_price, confidence, rsi, bb_position)
        
        # Sell signal: Overbought + near upper band
        elif rsi >= self.config['overbought'] and bb_position >= 0.8:
            stop_price = upper_band * 1.03  # 3% above upper band
            confidence = self._calculate_confidence(rsi, bb_position, False)
            if confidence >= self.config['min_confidence']:
                signal = self._create_signal('SELL', current_price, stop_price, confidence, rsi, bb_position)
        
        if signal:
            self.last_signal_time = datetime.now()
        
        return signal
    
    def _calculate_confidence(self, rsi: float, bb_position: float, is_buy: bool) -> float:
        """Calculate signal confidence"""
        if is_buy:
            rsi_strength = max(0, (self.config['oversold'] - rsi) / self.config['oversold']) * 50
            bb_strength = max(0, (0.2 - bb_position) / 0.2) * 50
        else:
            rsi_strength = max(0, (rsi - self.config['overbought']) / (100 - self.config['overbought'])) * 50
            bb_strength = max(0, (bb_position - 0.8) / 0.2) * 50
        
        return min(95, 50 + rsi_strength + bb_strength)
    
    def _create_signal(self, action: str, price: float, stop_price: float, 
                      confidence: float, rsi: float, bb_position: float) -> Dict:
        """Create signal dictionary"""
        return {
            'action': action,
            'strategy': 'RANGE',
            'price': price,
            'structure_stop': stop_price,
            'confidence': round(confidence, 1),
            'rsi': round(rsi, 1),
            'bb_position': round(bb_position, 3),
            'risk_reward_ratio': self.config['risk_reward_ratio'],
            'timeframe': '3m'
        }