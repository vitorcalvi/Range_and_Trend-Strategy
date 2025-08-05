import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional

class RangeStrategy:
    """3-Minute RSI+Bollinger Bands Range Strategy"""
    
    def __init__(self):
        self.config = {
            "rsi_length": 10,        # Faster for 3m
            "bb_length": 15,         # Faster for 3m
            "bb_std": 2.0,
            # More aggressive RSI levels for 3m
            "oversold": 40,          # Higher for 3m (more signals)
            "overbought": 60,        # Lower for 3m (more signals)
            "rsi_neutral_low": 48,   # Tighter neutral zone
            "rsi_neutral_high": 52,  # Tighter neutral zone
            "cooldown_seconds": 15,  # Faster cooldown for 3m
            "base_profit_usdt": 92,  # $92 target (2.4% of $3,817)
            "max_hold_seconds": 2700, # 45 minutes max
            "fee_rate": 0.0011
        }
        self.last_signal_time = None
        
    def calculate_rsi(self, prices: pd.Series) -> float:
        """Calculate RSI with faster 10-period for 3m"""
        period = self.config['rsi_length']
        if len(prices) < period + 3:
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
        """Calculate Bollinger Bands with faster 15-period for 3m"""
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
        """Calculate short-term momentum for 3m"""
        if len(prices) < 3:
            return 0
        
        # 3-period momentum for fast 3m signals
        momentum = (prices.iloc[-1] - prices.iloc[-3]) / prices.iloc[-3]
        return momentum
    
    def generate_signal(self, data: pd.DataFrame, market_condition: str) -> Optional[Dict]:
        """3-Minute range signal generation"""
        if len(data) < 20 or self._is_cooldown_active():
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
        
        # 3m Range signals: More aggressive entry conditions
        
        # Strong oversold conditions (more frequent for 3m)
        if (rsi <= self.config['oversold'] or bb_position <= 0.2):
            signal = self._create_signal('BUY', rsi, bb_position, price, data, market_condition, 'oversold_3m')
            
        # Strong overbought conditions
        elif (rsi >= self.config['overbought'] or bb_position >= 0.8):
            signal = self._create_signal('SELL', rsi, bb_position, price, data, market_condition, 'overbought_3m')
            
        # Fast neutral zone mean reversion for 3m
        elif (self.config['rsi_neutral_low'] <= rsi <= self.config['rsi_neutral_high'] and
              band_width > 0.01):  # Lower volatility threshold for 3m
            
            # Quick mean reversion based on BB position and momentum
            if bb_position <= 0.35 and momentum < -0.002:  # Lower BB + negative momentum
                signal = self._create_signal('BUY', rsi, bb_position, price, data, market_condition, 'neutral_reversion_3m')
            elif bb_position >= 0.65 and momentum > 0.002:  # Upper BB + positive momentum  
                signal = self._create_signal('SELL', rsi, bb_position, price, data, market_condition, 'neutral_reversion_3m')
        
        # Additional 3m scalping opportunity: BB squeeze breakout
        elif band_width < 0.008:  # Very tight range
            if bb_position <= 0.25 and momentum < -0.001:
                signal = self._create_signal('BUY', rsi, bb_position, price, data, market_condition, 'squeeze_breakout_3m')
            elif bb_position >= 0.75 and momentum > 0.001:
                signal = self._create_signal('SELL', rsi, bb_position, price, data, market_condition, 'squeeze_breakout_3m')
        
        if signal:
            self.last_signal_time = datetime.now()
        
        return signal
    
    def _create_signal(self, action: str, rsi: float, bb_position: float, price: float, 
                      data: pd.DataFrame, market_condition: str, signal_reason: str) -> Dict:
        """Create 3m range trading signal"""
        window = data.tail(20)  # Shorter window for 3m
        
        if action == 'BUY':
            structure_stop = window['low'].min() * 0.995  # Tighter stop for 3m
            level = window['low'].min()
        else:
            structure_stop = window['high'].max() * 1.005  # Tighter stop for 3m
            level = window['high'].max()
        
        # Validate stop distance for 3m trading
        stop_distance = abs(price - structure_stop) / price
        if not (0.005 <= stop_distance <= 0.02):  # Tighter range for 3m
            return None
        
        # 3m confidence calculation
        base_confidence = 70
        
        if '3m' in signal_reason:
            if 'oversold' in signal_reason or 'overbought' in signal_reason:
                rsi_strength = abs(50 - rsi) / 15  # Adjusted for 3m RSI levels
                bb_strength = abs(0.5 - bb_position) * 2
                base_confidence = 75 + (rsi_strength + bb_strength) * 12
            elif 'neutral_reversion' in signal_reason:
                bb_strength = abs(0.5 - bb_position) * 2
                base_confidence = 65 + bb_strength * 15
            elif 'squeeze_breakout' in signal_reason:
                momentum_strength = abs(bb_position - 0.5) * 3
                base_confidence = 68 + momentum_strength * 10
        
        if market_condition == "STRONG_RANGE":
            base_confidence *= 1.1
        
        confidence = np.clip(base_confidence, 65, 92)  # Higher min for 3m
        
        return {
            'action': action,
            'strategy': 'RANGE',
            'market_condition': market_condition,
            'signal_reason': signal_reason,
            'rsi': round(rsi, 1),
            'bb_position': round(bb_position, 2),
            'price': price,
            'structure_stop': structure_stop,
            'level': level,
            'signal_type': f"range_3m_{action.lower()}",
            'confidence': round(confidence, 1),
            'base_profit_usdt': self.config['base_profit_usdt'],
            'max_hold_seconds': self.config['max_hold_seconds'],
            'timeframe': '3m'
        }
    
    def _is_cooldown_active(self) -> bool:
        """Check if cooldown is active (15 seconds for 3m)"""
        if not self.last_signal_time:
            return False
        return (datetime.now() - self.last_signal_time).total_seconds() < self.config['cooldown_seconds']
    
    def get_strategy_info(self) -> Dict:
        """Get 3m strategy information"""
        return {
            'name': '3m RSI+BB Range Strategy (SCALPING)',
            'type': 'RANGE',
            'timeframe': '3m',
            'config': self.config,
            'description': f'Fast 3m scalping: RSI({self.config["rsi_length"]}) + BB({self.config["bb_length"]}) - ${self.config["base_profit_usdt"]} target',
            'optimization': 'Aggressive 3m parameters + $92 profit target',
            'expected_trades': '8-15 per day',
            'hold_time': '15-45 minutes'
        }