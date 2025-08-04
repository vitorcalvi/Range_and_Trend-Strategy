# strategies/range_strategy.py
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional

class RangeStrategy:
    """Streamlined RSI+MFI Range Strategy"""
    
    def __init__(self):
        self.config = {
            "rsi_length": 5, "mfi_length": 5,
            "oversold_weak": 40, "oversold_strong": 50,
            "mfi_threshold_weak": 35, "mfi_threshold_strong": 50,
            "overbought": 60, "cooldown_seconds": 0.5,
            "target_profit_usdt": 15, "max_hold_seconds": 180
        }
        self.last_signal_time = None
        
    def calculate_rsi(self, prices: pd.Series) -> pd.Series:
        """Fast RSI calculation"""
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
        """Fast MFI calculation"""
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
        """Generate range signals with cooldown check"""
        if len(data) < 20 or self._is_cooldown_active():
            return None
        
        rsi = self.calculate_rsi(data['close']).iloc[-1]
        mfi = self.calculate_mfi(data['high'], data['low'], data['close'], data['volume']).iloc[-1]
        price = data['close'].iloc[-1]
        
        if pd.isna(rsi) or pd.isna(mfi):
            return None
        
        # Dynamic thresholds based on market condition
        oversold_level = self.config['oversold_strong'] if market_condition == "STRONG_RANGE" else self.config['oversold_weak']
        mfi_level = self.config['mfi_threshold_strong'] if market_condition == "STRONG_RANGE" else self.config['mfi_threshold_weak']
        
        signal = None
        
        # Signal generation
        if rsi <= oversold_level and mfi <= mfi_level:
            signal = self._create_signal('BUY', rsi, mfi, price, data, market_condition)
        elif rsi >= self.config['overbought'] and mfi >= self.config['mfi_threshold_strong']:
            signal = self._create_signal('SELL', rsi, mfi, price, data, market_condition)
        
        if signal:
            self.last_signal_time = datetime.now()
        
        return signal
    
    def _create_signal(self, action: str, rsi: float, mfi: float, price: float, 
                      data: pd.DataFrame, market_condition: str) -> Dict:
        """Create signal with validation"""
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
        
        # Calculate confidence
        base_confidence = (abs(50 - rsi) + abs(50 - mfi)) * 1.5
        if market_condition == "STRONG_RANGE":
            base_confidence *= 1.1
        confidence = min(95, max(60, base_confidence))
        
        return {
            'action': action, 'strategy': 'RANGE', 'market_condition': market_condition,
            'rsi': round(rsi, 1), 'mfi': round(mfi, 1), 'price': price,
            'structure_stop': structure_stop, 'level': level,
            'signal_type': f"range_{action.lower()}", 'confidence': round(confidence, 1),
            'target_profit_usdt': self.config['target_profit_usdt'],
            'max_hold_seconds': self.config['max_hold_seconds'], 'timeframe': '1m'
        }
    
    def _is_cooldown_active(self) -> bool:
        """Check cooldown"""
        if not self.last_signal_time:
            return False
        return (datetime.now() - self.last_signal_time).total_seconds() < self.config['cooldown_seconds']
    
    def get_strategy_info(self) -> Dict:
        """Get strategy info"""
        return {
            'name': 'RSI+MFI Range Strategy', 'type': 'RANGE', 'timeframe': '1m',
            'config': self.config, 'description': 'Mean reversion scalping for range-bound markets'
        }


# strategies/trend_strategy.py
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional

class TrendStrategy:
    """Streamlined RSI+MA Trend Strategy"""
    
    def __init__(self):
        self.config = {
            "rsi_length": 14, "ma_length": 15, "ma_type": "EMA",
            "uptrend_rsi_low": 30, "uptrend_rsi_high": 45,
            "downtrend_rsi_low": 55, "downtrend_rsi_high": 70,
            "momentum_threshold": 0.001, "cooldown_seconds": 300,
            "target_profit_multiplier": 2.5, "max_hold_seconds": 3600,
            "trailing_stop_pct": 0.5
        }
        self.last_signal_time = None
        
    def calculate_rsi(self, prices: pd.Series) -> pd.Series:
        """Standard RSI calculation"""
        period = self.config['rsi_length']
        if len(prices) < period + 5:
            return pd.Series(50.0, index=prices.index)
        
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta.where(delta < 0, 0))
        
        alpha = 1.0 / period
        avg_gain = gain.ewm(alpha=alpha, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=alpha, min_periods=period).mean()
        
        rs = avg_gain / (avg_loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.fillna(50.0)
    
    def calculate_ma(self, prices: pd.Series) -> pd.Series:
        """Moving average calculation"""
        period = self.config['ma_length']
        if len(prices) < period:
            return pd.Series(prices.iloc[-1], index=prices.index)
            
        try:
            if self.config['ma_type'] == "EMA":
                ma = prices.ewm(span=period, min_periods=period).mean()
            else:
                ma = prices.rolling(period, min_periods=period).mean()
            return ma.fillna(prices.iloc[-1])
        except:
            return pd.Series(prices.iloc[-1], index=prices.index)
    
    def detect_trend(self, close: pd.Series, ma: pd.Series) -> str:
        """Detect trend direction"""
        if len(close) < 5 or len(ma) < 5:
            return 'NEUTRAL'
            
        try:
            current_price = close.iloc[-1]
            current_ma = ma.iloc[-1]
            
            if pd.isna(current_price) or pd.isna(current_ma):
                return 'NEUTRAL'
            
            # Check MA slope
            ma_slope = (ma.iloc[-1] - ma.iloc[-5]) / ma.iloc[-5] if len(ma) >= 5 else 0
            
            if current_price > current_ma and ma_slope > self.config['momentum_threshold']:
                return 'UPTREND'
            elif current_price < current_ma and ma_slope < -self.config['momentum_threshold']:
                return 'DOWNTREND'
            else:
                return 'NEUTRAL'
        except:
            return 'NEUTRAL'
    
    def generate_signal(self, data: pd.DataFrame, market_condition: str) -> Optional[Dict]:
        """Generate trend signals"""
        if len(data) < 30 or self._is_cooldown_active():
            return None
        
        # Only trade in trending markets
        if market_condition not in ["TRENDING", "STRONG_TREND"]:
            return None
        
        close = data['close']
        rsi = self.calculate_rsi(close).iloc[-1]
        ma = self.calculate_ma(close)
        trend = self.detect_trend(close, ma)
        price = close.iloc[-1]
        
        if pd.isna(rsi) or trend == 'NEUTRAL':
            return None
        
        signal = None
        
        # Signal generation
        if (trend == 'UPTREND' and self.config['uptrend_rsi_low'] <= rsi <= self.config['uptrend_rsi_high']):
            signal = self._create_signal('BUY', trend, rsi, price, data, ma.iloc[-1])
        elif (trend == 'DOWNTREND' and self.config['downtrend_rsi_low'] <= rsi <= self.config['downtrend_rsi_high']):
            signal = self._create_signal('SELL', trend, rsi, price, data, ma.iloc[-1])
        
        if signal:
            self.last_signal_time = datetime.now()
        
        return signal
    
    def _create_signal(self, action: str, trend: str, rsi: float, price: float, 
                      data: pd.DataFrame, ma_value: float) -> Dict:
        """Create trend signal with validation"""
        window = data.tail(30)
        
        if action == 'BUY':
            swing_low = window['low'].min()
            structure_stop = max(swing_low, ma_value * 0.995)
            level = swing_low
        else:
            swing_high = window['high'].max()
            structure_stop = min(swing_high, ma_value * 1.005)
            level = swing_high
        
        # Validate stop distance
        stop_distance = abs(price - structure_stop) / price
        if not (0.003 <= stop_distance <= 0.02):
            return None
        
        # Calculate target and confidence
        risk_amount = abs(price - structure_stop)
        target_price = price + (risk_amount * self.config['target_profit_multiplier']) if action == 'BUY' else price - (risk_amount * self.config['target_profit_multiplier'])
        
        trend_strength = abs(price - ma_value) / ma_value * 100
        base_confidence = min(95, 60 + trend_strength * 10 + abs(rsi - 50) * 0.5)
        
        return {
            'action': action, 'strategy': 'TREND', 'trend': trend,
            'rsi': round(rsi, 1), 'ma_value': round(ma_value, 2), 'price': price,
            'structure_stop': structure_stop, 'target_price': target_price, 'level': level,
            'signal_type': f"trend_{action.lower()}", 'confidence': round(base_confidence, 1),
            'risk_reward_ratio': self.config['target_profit_multiplier'],
            'max_hold_seconds': self.config['max_hold_seconds'],
            'trailing_stop_pct': self.config['trailing_stop_pct'], 'timeframe': '15m'
        }
    
    def _is_cooldown_active(self) -> bool:
        """Check cooldown"""
        if not self.last_signal_time:
            return False
        return (datetime.now() - self.last_signal_time).total_seconds() < self.config['cooldown_seconds']
    
    def get_strategy_info(self) -> Dict:
        """Get strategy info"""
        return {
            'name': 'RSI+MA Trend Strategy', 'type': 'TREND', 'timeframe': '15m', 
            'config': self.config, 'description': 'Trend following with RSI pullbacks and 1:2.5 RR',
            'win_rate': '70-83%', 'risk_reward': '1:2.5'
        }


# strategies/strategy_manager.py
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Tuple, Any

class MarketConditionDetector:
    """Streamlined market condition detection"""
    
    def __init__(self):
        self.adx_period = 14
        
    def calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series) -> float:
        """Simplified ADX calculation"""
        try:
            if len(close) < 20:
                return 25.0
            
            # Use last 50 periods
            h = high.iloc[-50:] if len(high) >= 50 else high
            l = low.iloc[-50:] if len(low) >= 50 else low
            c = close.iloc[-50:] if len(close) >= 50 else close
            
            tr, dm_plus, dm_minus = [], [], []
            
            for i in range(1, len(h)):
                # True Range
                tr_val = max(h.iloc[i] - l.iloc[i], abs(h.iloc[i] - c.iloc[i-1]), abs(l.iloc[i] - c.iloc[i-1]))
                tr.append(tr_val)
                
                # Directional Movement
                h_move = h.iloc[i] - h.iloc[i-1]
                l_move = l.iloc[i-1] - l.iloc[i]
                
                dm_plus.append(h_move if h_move > l_move and h_move > 0 else 0)
                dm_minus.append(l_move if l_move > h_move and l_move > 0 else 0)
            
            if len(tr) < 14:
                return 25.0
            
            # Simple averages
            tr_avg = sum(tr[-14:]) / 14
            dm_plus_avg = sum(dm_plus[-14:]) / 14
            dm_minus_avg = sum(dm_minus[-14:]) / 14
            
            if tr_avg == 0:
                return 25.0
            
            # Calculate DI and DX
            di_plus = 100 * dm_plus_avg / tr_avg
            di_minus = 100 * dm_minus_avg / tr_avg
            di_sum = di_plus + di_minus
            
            if di_sum == 0:
                return 25.0
            
            dx = 100 * abs(di_plus - di_minus) / di_sum
            return max(0, min(100, dx))
            
        except:
            return 25.0
    
    def calculate_volatility_regime(self, close: pd.Series) -> str:
        """Simple volatility regime detection"""
        try:
            if len(close) < 20:
                return "NORMAL"
            
            close = close.fillna(close.iloc[-1])
            sma = close.rolling(20, min_periods=20).mean()
            std = close.rolling(20, min_periods=20).std()
            
            current_sma = sma.iloc[-1]
            current_std = std.iloc[-1]
            
            if pd.isna(current_sma) or pd.isna(current_std) or current_sma == 0:
                return "NORMAL"
            
            bb_width = (current_std * 2) / current_sma
            
            if len(std) >= 50:
                avg_width = (std * 2 / sma).rolling(50, min_periods=25).mean().iloc[-1]
                if pd.isna(avg_width) or avg_width == 0:
                    return "NORMAL"
                
                width_ratio = bb_width / avg_width
                if width_ratio > 1.5:
                    return "HIGH_VOL"
                elif width_ratio < 0.5:
                    return "LOW_VOL"
                    
            return "NORMAL"
                
        except:
            return "NORMAL"
    
    def detect_market_condition(self, data_1m: pd.DataFrame, data_15m: pd.DataFrame) -> Dict[str, Any]:
        """Detect market condition"""
        if len(data_1m) < 50 or len(data_15m) < 30:
            return {"condition": "INSUFFICIENT_DATA", "adx": 25.0, "confidence": 0, "timestamp": datetime.now()}
        
        try:
            adx_15m = self.calculate_adx(data_15m['high'], data_15m['low'], data_15m['close'])
            vol_regime = self.calculate_volatility_regime(data_1m['close'])
            
            if not np.isfinite(adx_15m):
                adx_15m = 25.0
            
            # Determine condition
            if adx_15m < 20:
                condition, confidence = "STRONG_RANGE", 0.9
            elif adx_15m < 25:
                condition, confidence = "WEAK_RANGE", 0.7
            elif adx_15m < 40:
                condition, confidence = "TRENDING", 0.8
            else:
                condition, confidence = "STRONG_TREND", 0.95
            
            return {
                "condition": condition, "adx": adx_15m, "volatility": vol_regime,
                "confidence": confidence, "timestamp": datetime.now()
            }
            
        except:
            return {
                "condition": "WEAK_RANGE", "adx": 25.0, "volatility": "NORMAL",
                "confidence": 0.5, "timestamp": datetime.now()
            }

class StrategyManager:
    """Streamlined strategy management"""
    
    def __init__(self):
        self.detector = MarketConditionDetector()
        self.current_strategy = None
        self.last_switch_time = None
        self.switch_cooldown = 300  # 5 minutes
        self.market_condition = {"condition": "UNKNOWN", "adx": 25.0}
        
    def should_switch_strategy(self, new_condition: str) -> bool:
        """Determine strategy switch"""
        if not self.current_strategy:
            return True
            
        # Cooldown check
        if (self.last_switch_time and 
            (datetime.now() - self.last_switch_time).total_seconds() < self.switch_cooldown):
            return False
            
        # Strategy mapping
        current_type = "RANGE" if self.current_strategy in ["STRONG_RANGE", "WEAK_RANGE"] else "TREND"
        new_type = "RANGE" if new_condition in ["STRONG_RANGE", "WEAK_RANGE"] else "TREND"
        
        return current_type != new_type
    
    def select_strategy(self, data_1m: pd.DataFrame, data_15m: pd.DataFrame) -> Tuple[str, Dict[str, Any]]:
        """Select strategy based on market conditions"""
        market_info = self.detector.detect_market_condition(data_1m, data_15m)
        condition = market_info["condition"]
        
        if condition == "INSUFFICIENT_DATA":
            return "RANGE", market_info
            
        if self.should_switch_strategy(condition):
            self.current_strategy = condition
            self.last_switch_time = datetime.now()
            
        strategy_type = "RANGE" if condition in ["STRONG_RANGE", "WEAK_RANGE"] else "TREND"
        self.market_condition = market_info
        return strategy_type, market_info
    
    def get_position_sizing_multiplier(self, strategy_type: str, market_info: Dict[str, Any]) -> float:
        """Get position sizing multiplier"""        
        if strategy_type == "TREND":
            base_multiplier = 1.5 if market_info["condition"] == "STRONG_TREND" else 1.2
        else:
            base_multiplier = 0.8 if market_info["condition"] == "STRONG_RANGE" else 1.0
                
        # Volatility adjustment
        volatility = market_info.get("volatility", "NORMAL")
        if volatility == "HIGH_VOL":
            base_multiplier *= 0.7
        elif volatility == "LOW_VOL":
            base_multiplier *= 1.2
            
        return base_multiplier