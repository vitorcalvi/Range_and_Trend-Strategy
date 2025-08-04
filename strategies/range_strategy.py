import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional

class RangeStrategy:
    """RSI+Bollinger Bands Range Strategy - FIXED for More Trading"""
    
    def __init__(self):
        self.config = {
            "rsi_length": 14,
            "bb_length": 20,
            "bb_std": 2.0,
            # FIXED: More reasonable RSI levels
            "oversold": 35,      # Was 30 - too extreme
            "overbought": 65,    # Was 70 - too extreme  
            "rsi_neutral_low": 45,   # NEW: Neutral zone trading
            "rsi_neutral_high": 55,  # NEW: Neutral zone trading
            "cooldown_seconds": 30,  # FIXED: Reduced from 60
            "base_profit_usdt": 35,
            "max_hold_seconds": 300,
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
            return 0.5, 0
        
        sma = prices.rolling(period).mean().iloc[-1]
        std = prices.rolling(period).std().iloc[-1]
        current_price = prices.iloc[-1]
        
        if pd.isna(sma) or pd.isna(std) or std == 0:
            return 0.5, 0
        
        upper_band = sma + (std * std_mult)
        lower_band = sma - (std * std_mult)
        
        # Calculate position within bands
        bb_position = (current_price - lower_band) / (upper_band - lower_band)
        bb_position = np.clip(bb_position, 0, 1)
        
        band_width = (upper_band - lower_band) / sma
        
        return bb_position, band_width
    
    def calculate_price_momentum(self, prices: pd.Series) -> float:
        """Calculate short-term price momentum"""
        if len(prices) < 5:
            return 0
        
        # 5-period momentum
        momentum = (prices.iloc[-1] - prices.iloc[-5]) / prices.iloc[-5]
        return momentum
    
    def generate_signal(self, data: pd.DataFrame, market_condition: str) -> Optional[Dict]:
        """FIXED: More active signal generation"""
        if len(data) < 25 or self._is_cooldown_active():
            return None
        
        # Only trade in ranging markets
        if market_condition not in ["STRONG_RANGE", "WEAK_RANGE"]:
            return None
        
        rsi = self.calculate_rsi(data['close'])
        bb_position, band_width = self.calculate_bollinger_position(data['close'])
        momentum = self.calculate_price_momentum(data['close'])
        price = data['close'].iloc[-1]
        
        if pd.isna(rsi) or band_width == 0:
            return None
        
        signal = None
        
        # FIXED: Multiple signal conditions (more opportunities)
        
        # Strong oversold conditions
        if (rsi <= self.config['oversold'] or bb_position <= 0.15):
            signal = self._create_signal('BUY', rsi, bb_position, price, data, market_condition, 'strong_oversold')
            
        # Strong overbought conditions  
        elif (rsi >= self.config['overbought'] or bb_position >= 0.85):
            signal = self._create_signal('SELL', rsi, bb_position, price, data, market_condition, 'strong_overbought')
            
        # NEW: Neutral zone mean reversion (more active trading)
        elif (self.config['rsi_neutral_low'] <= rsi <= self.config['rsi_neutral_high'] and
              band_width > 0.015):  # Ensure some volatility
            
            # Mean reversion in neutral zone based on BB position and momentum
            if bb_position <= 0.3 and momentum < -0.003:  # Lower BB area + negative momentum
                signal = self._create_signal('BUY', rsi, bb_position, price, data, market_condition, 'neutral_reversion')
            elif bb_position >= 0.7 and momentum > 0.003:  # Upper BB area + positive momentum  
                signal = self._create_signal('SELL', rsi, bb_position, price, data, market_condition, 'neutral_reversion')
        
        if signal:
            self.last_signal_time = datetime.now()
        
        return signal
    
    def _create_signal(self, action: str, rsi: float, bb_position: float, price: float, 
                      data: pd.DataFrame, market_condition: str, signal_reason: str) -> Dict:
        """Create range trading signal"""
        window = data.tail(30)  # Longer window for better structure
        
        if action == 'BUY':
            structure_stop = window['low'].min() * 0.997  # Slightly tighter
            level = window['low'].min()
        else:
            structure_stop = window['high'].max() * 1.003  # Slightly tighter
            level = window['high'].max()
        
        # Validate stop distance
        stop_distance = abs(price - structure_stop) / price
        if not (0.0008 <= stop_distance <= 0.012):  # FIXED: Wider acceptable range
            return None
        
        # FIXED: Confidence based on signal type and conditions
        base_confidence = 70
        
        if signal_reason == 'strong_oversold' or signal_reason == 'strong_overbought':
            rsi_strength = abs(50 - rsi) / 20
            bb_strength = abs(0.5 - bb_position) * 2
            base_confidence = 75 + (rsi_strength + bb_strength) * 15
        elif signal_reason == 'neutral_reversion':
            bb_strength = abs(0.5 - bb_position) * 2
            base_confidence = 65 + bb_strength * 20
        
        if market_condition == "STRONG_RANGE":
            base_confidence *= 1.1
        
        confidence = np.clip(base_confidence, 60, 95)
        
        return {
            'action': action,
            'strategy': 'RANGE',
            'market_condition': market_condition,
            'signal_reason': signal_reason,  # NEW: Track signal type
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
            'name': 'RSI+BB Range Strategy (ACTIVE)',
            'type': 'RANGE',
            'timeframe': '1m',
            'config': self.config,
            'description': f'Multi-signal range strategy: RSI({self.config["rsi_length"]}) + BB({self.config["bb_length"]}) - ${self.config["base_profit_usdt"]} target'
        }