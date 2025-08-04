import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional

class RangeStrategy:
    """RSI+MFI Range-Bound Strategy for sideways markets - FIXED"""
    
    def __init__(self):
        self.config = {
            "rsi_length": 5,
            "mfi_length": 5,
            "oversold_weak": 40,
            "oversold_strong": 50,
            "mfi_threshold_weak": 35,
            "mfi_threshold_strong": 50,
            "overbought": 60,
            "cooldown_seconds": 0.5,
            "base_profit_usdt": 35,  # FIXED: Increased from 15 to 35
            "max_hold_seconds": 180,
            "fee_rate": 0.0011  # 0.11% round-trip fee (0.055% taker x 2)
        }
        self.last_signal_time = None
        
    def calculate_rsi(self, prices: pd.Series) -> float:
        """Calculate current RSI value"""
        period = self.config['rsi_length']
        if len(prices) < period + 5:
            return 50.0
        
        delta = prices.diff().fillna(0)
        gain = delta.where(delta > 0, 0)
        loss = (-delta.where(delta < 0, 0))
        
        alpha = 2.0 / (period + 1)
        avg_gain = gain.ewm(alpha=alpha, min_periods=period).mean().iloc[-1]
        avg_loss = loss.ewm(alpha=alpha, min_periods=period).mean().iloc[-1]
        
        if avg_loss == 0:
            return 95.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return np.clip(rsi, 5, 95)
    
    def calculate_mfi(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> float:
        """Calculate current MFI value"""
        period = self.config['mfi_length']
        if len(close) < period + 5 or volume.sum() == 0:
            return 50.0
        
        tp = (high + low + close) / 3
        money_flow = tp * volume
        
        mf_change = tp.diff().fillna(0)
        pos_mf = money_flow.where(mf_change > 0, 0)
        neg_mf = money_flow.where(mf_change <= 0, 0)
        
        alpha = 2.0 / (period + 1)
        pos_mf_avg = pos_mf.ewm(alpha=alpha, min_periods=period).mean().iloc[-1]
        neg_mf_avg = neg_mf.ewm(alpha=alpha, min_periods=period).mean().iloc[-1]
        
        if neg_mf_avg == 0:
            return 85.0
        
        mfi_ratio = pos_mf_avg / neg_mf_avg
        mfi = 100 - (100 / (1 + mfi_ratio))
        return np.clip(mfi, 15, 85)
    
    def calculate_profit_target(self, position_size_usdt: float) -> float:
        """Calculate profit target including fees"""
        fee_cost = position_size_usdt * self.config['fee_rate']
        return self.config['base_profit_usdt'] + fee_cost
    
    def validate_trade_profitability(self, position_size_usdt: float, stop_distance_pct: float) -> bool:
        """FIXED: Validate trade can be profitable given fees"""
        fee_cost = position_size_usdt * self.config['fee_rate']
        gross_profit_needed = self.config['base_profit_usdt'] + fee_cost
        
        # Minimum price movement needed for profit
        min_movement_pct = gross_profit_needed / position_size_usdt
        
        # Stop distance should allow for at least 2x the minimum movement
        max_acceptable_stop_pct = min_movement_pct * 0.5  # 50% of required movement as max stop
        
        return stop_distance_pct <= max_acceptable_stop_pct
    
    def generate_signal(self, data: pd.DataFrame, market_condition: str) -> Optional[Dict]:
        """Generate range trading signals"""
        if len(data) < 20 or self._is_cooldown_active():
            return None
        
        rsi = self.calculate_rsi(data['close'])
        mfi = self.calculate_mfi(data['high'], data['low'], data['close'], data['volume'])
        price = data['close'].iloc[-1]
        
        if pd.isna(rsi) or pd.isna(mfi):
            return None
        
        # Get thresholds based on market condition
        oversold_level, mfi_level = (
            (self.config['oversold_strong'], self.config['mfi_threshold_strong'])
            if market_condition == "STRONG_RANGE" else
            (self.config['oversold_weak'], self.config['mfi_threshold_weak'])
        )
        
        signal = None
        
        # Long signal: Mean reversion from oversold
        if rsi <= oversold_level and mfi <= mfi_level:
            signal = self._create_signal('BUY', rsi, mfi, price, data, market_condition)
        # Short signal: Mean reversion from overbought
        elif rsi >= self.config['overbought'] and mfi >= self.config['mfi_threshold_strong']:
            signal = self._create_signal('SELL', rsi, mfi, price, data, market_condition)
        
        if signal:
            self.last_signal_time = datetime.now()
        
        return signal
    
    def _create_signal(self, action: str, rsi: float, mfi: float, price: float, 
                      data: pd.DataFrame, market_condition: str) -> Dict:
        """Create range trading signal"""
        window = data.tail(20)
        
        if action == 'BUY':
            structure_stop = window['low'].min() * 0.9985
            level = window['low'].min()
        else:
            structure_stop = window['high'].max() * 1.0015
            level = window['high'].max()
        
        # Validate stop distance
        stop_distance = abs(price - structure_stop) / price
        if not (0.0005 <= stop_distance <= 0.005):
            return None
        
        # FIXED: Validate trade profitability before creating signal
        # Use typical position size for validation
        typical_position_size = 9000  # Typical range strategy position size
        if not self.validate_trade_profitability(typical_position_size, stop_distance):
            return None
        
        # Calculate confidence
        rsi_strength = abs(50 - rsi)
        mfi_strength = abs(50 - mfi)
        base_confidence = (rsi_strength + mfi_strength) * 1.5
        
        if market_condition == "STRONG_RANGE":
            base_confidence *= 1.1
        
        confidence = np.clip(base_confidence, 60, 95)
        
        return {
            'action': action,
            'strategy': 'RANGE',
            'market_condition': market_condition,
            'rsi': round(rsi, 1),
            'mfi': round(mfi, 1),
            'price': price,
            'structure_stop': structure_stop,
            'level': level,
            'signal_type': f"range_{action.lower()}",
            'confidence': round(confidence, 1),
            'base_profit_usdt': self.config['base_profit_usdt'],
            'max_hold_seconds': self.config['max_hold_seconds'],
            'timeframe': '1m',
            'required_movement_pct': round((self.config['base_profit_usdt'] + typical_position_size * self.config['fee_rate']) / typical_position_size * 100, 3)
        }
    
    def _is_cooldown_active(self) -> bool:
        """Check if cooldown is active"""
        if not self.last_signal_time:
            return False
        return (datetime.now() - self.last_signal_time).total_seconds() < self.config['cooldown_seconds']
    
    def get_strategy_info(self) -> Dict:
        """Get strategy information"""
        return {
            'name': 'RSI+MFI Range Strategy (Fixed)',
            'type': 'RANGE',
            'timeframe': '1m',
            'config': self.config,
            'description': f'Mean reversion scalping - ${self.config["base_profit_usdt"]} net profit target'
        }