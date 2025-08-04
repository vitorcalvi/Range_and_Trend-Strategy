import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional

class RangeStrategy:
    """RSI+MFI Range-Bound Strategy for sideways markets"""
    
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
            "target_profit_usdt": 15,
            "max_hold_seconds": 180
        }
        self.last_signal_time = None
        
    def calculate_rsi(self, prices: pd.Series) -> pd.Series:
        """Calculate RSI with fast EMA smoothing"""
        period = self.config['rsi_length']
        if len(prices) < period + 5:
            return pd.Series(50.0, index=prices.index)
        
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta.where(delta < 0, 0))
        
        alpha = 2.0 / (period + 1)
        avg_gain = gain.ewm(alpha=alpha, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=alpha, min_periods=period).mean()
        
        rs = avg_gain / (avg_loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.fillna(50.0).clip(5, 95)
    
    def calculate_mfi(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate MFI with fast EMA smoothing"""
        period = self.config['mfi_length']
        if len(close) < period + 5 or volume.sum() == 0:
            return pd.Series(50.0, index=close.index)
        
        tp = (high + low + close) / 3
        money_flow = tp * volume
        
        mf_change = tp.diff()
        pos_mf = money_flow.where(mf_change > 0, 0)
        neg_mf = money_flow.where(mf_change <= 0, 0)
        
        alpha = 2.0 / (period + 1)
        pos_mf_avg = pos_mf.ewm(alpha=alpha, min_periods=period).mean()
        neg_mf_avg = neg_mf.ewm(alpha=alpha, min_periods=period).mean()
        
        mfi_ratio = pos_mf_avg / (neg_mf_avg + 1e-8)
        mfi = 100 - (100 / (1 + mfi_ratio))
        
        return mfi.fillna(50.0).clip(15, 85)
    
    def generate_signal(self, data: pd.DataFrame, market_condition: str) -> Optional[Dict]:
        """Generate range trading signals"""
        if len(data) < 20 or self._is_cooldown_active():
            return None
        
        rsi = self.calculate_rsi(data['close']).iloc[-1]
        mfi = self.calculate_mfi(data['high'], data['low'], data['close'], data['volume']).iloc[-1]
        price = data['close'].iloc[-1]
        
        if pd.isna(rsi) or pd.isna(mfi):
            return None
        
        # Get thresholds based on market condition
        if market_condition == "STRONG_RANGE":
            oversold_level = self.config['oversold_strong']
            mfi_level = self.config['mfi_threshold_strong']
        else:
            oversold_level = self.config['oversold_weak']
            mfi_level = self.config['mfi_threshold_weak']
        
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
        if stop_distance < 0.0005 or stop_distance > 0.005:
            return None
        
        # Calculate confidence
        rsi_strength = abs(50 - rsi)
        mfi_strength = abs(50 - mfi)
        base_confidence = (rsi_strength + mfi_strength) * 1.5
        
        if market_condition == "STRONG_RANGE":
            base_confidence *= 1.1
        
        confidence = min(95, max(60, base_confidence))
        
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
            'target_profit_usdt': self.config['target_profit_usdt'],
            'max_hold_seconds': self.config['max_hold_seconds'],
            'timeframe': '1m'
        }
    
    def _is_cooldown_active(self) -> bool:
        """Check if cooldown is active"""
        if not self.last_signal_time:
            return False
        return (datetime.now() - self.last_signal_time).total_seconds() < self.config['cooldown_seconds']
    
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate strategy indicators"""
        if len(data) < max(self.config['rsi_length'], self.config['mfi_length']) + 5:
            return {}
        
        try:
            rsi = self.calculate_rsi(data['close'])
            mfi = self.calculate_mfi(data['high'], data['low'], data['close'], data['volume'])
            
            if rsi.isna().all() or mfi.isna().all():
                return {}
                
            return {'rsi': rsi, 'mfi': mfi}
        except:
            return {}
    
    def get_strategy_info(self) -> Dict:
        """Get strategy information"""
        return {
            'name': 'RSI+MFI Range Strategy',
            'type': 'RANGE',
            'timeframe': '1m',
            'config': self.config,
            'description': 'Mean reversion scalping for range-bound markets'
        }