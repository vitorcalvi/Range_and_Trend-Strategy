import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Optional

class TrendStrategy:
    """Optimized Trend Strategy - 1.2R Breakeven + 0.5 ATR Trailing"""
    
    def __init__(self):
        # Corrected fee model with maker/taker blend
        self.taker_fee_rate = Decimal('0.00055')    # 0.055% taker
        self.maker_fee_rate = Decimal('0.0001')     # 0.01% maker
        self.maker_fill_ratio = Decimal('0.3')      # 30% maker assumption
        self.blended_fee_rate = (self.maker_fill_ratio * self.maker_fee_rate + 
                               (1 - self.maker_fill_ratio) * self.taker_fee_rate)
        self.slippage_rate = Decimal('0.0002')      # 0.02% slippage
        self.total_cost_rate = self.blended_fee_rate + self.slippage_rate  # ~0.062%
        
        self.config = {
            "rsi_length": 14,
            "fast_ema": 21,
            "slow_ema": 50,
            "atr_length": 14,
            
            # Optimized RSI levels for trend pullbacks
            "uptrend_rsi_low": 38,     # Pullback in uptrend
            "uptrend_rsi_high": 62,    # Rejection level in uptrend
            "downtrend_rsi_low": 38,   # Rejection level in downtrend
            "downtrend_rsi_high": 62,  # Pullback in downtrend
            
            # CORRECTED: 2R setup parameters
            "risk_reward_ratio": 2.0,      # 2R target
            "risk_percentage": 0.018,      # 1.8% position risk (1R)
            "min_confidence": 70,          # Minimum confidence threshold
            
            # Trailing stop parameters (NEW)
            "breakeven_threshold": 1.2,    # 1.2R breakeven move to activate trailing
            "trailing_atr_multiplier": 0.5, # 0.5 ATR trailing distance
            "trailing_fallback_pct": 0.005,  # 0.5% fallback if no ATR
            "max_drawdown_from_peak": 0.2,   # 20% drawdown from peak triggers exit
            
            # Dynamic position sizing
            "fee_target_percentage": 0.025,  # Fees should be 2.5% of expected gross
            "min_position_usdt": 3000,       # Minimum for fee efficiency
            "max_position_usdt": 10000,      # Maximum for safety
            
            # Timing and momentum
            "trend_strength_threshold": 0.0008,  # Minimum EMA separation
            "cooldown_seconds": 180,         # 3 minutes between signals
            "max_hold_seconds": 2400,        # 40 minutes max hold
            "momentum_confirmation": True,    # Require momentum confirmation
            
            # Backward compatibility
            "base_profit_usdt": 45,          # Base profit target (legacy)
        }
        self.last_signal_time = None
        
        # Calculate corrected break-even rate
        self._calculate_breakeven_rate()
        
    def _calculate_breakeven_rate(self):
        """Calculate corrected break-even win rate for 2R setup"""
        # CORRECTED FORMULA: Win_Rate = (Risk + Fee) / (Risk + Reward)
        risk_pct = float(self.config['risk_percentage'])      # 1.8%
        reward_pct = risk_pct * self.config['risk_reward_ratio']  # 3.6%
        fee_pct = float(self.total_cost_rate)  # ~0.062%
        
        numerator = risk_pct + fee_pct      # 0.018 + 0.00062 = 0.01862
        denominator = risk_pct + reward_pct  # 0.018 + 0.036 = 0.054
        self.breakeven_rate = numerator / denominator  # 0.34481 = 34.48%
        
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
        """Calculate fast and slow EMAs with trend determination"""
        fast_period = self.config['fast_ema']
        slow_period = self.config['slow_ema']
        
        if len(prices) < slow_period:
            return prices.iloc[-1], prices.iloc[-1], 'NEUTRAL'
            
        fast_ema = prices.ewm(span=fast_period, min_periods=fast_period).mean().iloc[-1]
        slow_ema = prices.ewm(span=slow_period, min_periods=slow_period).mean().iloc[-1]
        
        if pd.isna(fast_ema) or pd.isna(slow_ema):
            return fast_ema, slow_ema, 'NEUTRAL'
        
        # Enhanced trend detection
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
        if len(ema_series) < 8:
            return 0
        
        # Calculate 8-period slope for momentum
        slope = (ema_series.iloc[-1] - ema_series.iloc[-8]) / ema_series.iloc[-8]
        return slope
    
    def calculate_atr(self, data: pd.DataFrame) -> float:
        """Calculate Average True Range for trailing stops"""
        if len(data) < self.config['atr_length'] + 1:
            return 0
        
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Calculate True Range
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(self.config['atr_length']).mean().iloc[-1]
        
        return float(atr) if not pd.isna(atr) else 0
    
    def should_activate_trailing_stop(self, unrealized_pnl: float, position_size_usdt: float) -> bool:
        """Check if position has reached 1.2R breakeven threshold"""
        if position_size_usdt <= 0:
            return False
            
        # Calculate 1.2R threshold
        risk_amount = position_size_usdt * float(self.config['risk_percentage'])
        breakeven_threshold = risk_amount * self.config['breakeven_threshold']  # 1.2R
        
        # Subtract fees from threshold
        fee_cost = position_size_usdt * float(self.total_cost_rate)
        net_threshold = breakeven_threshold - fee_cost
        
        return unrealized_pnl >= net_threshold
    
    def calculate_trailing_stop_price(self, current_price: float, side: str, atr_value: float) -> float:
        """Calculate trailing stop price using 0.5 ATR"""
        if atr_value > 0:
            trail_distance = atr_value * self.config['trailing_atr_multiplier']  # 0.5 ATR
        else:
            # Fallback to percentage-based trailing
            trail_distance = current_price * self.config['trailing_fallback_pct']  # 0.5%
        
        if side.lower() == 'buy':
            return current_price - trail_distance
        else:
            return current_price + trail_distance
    
    def generate_signal(self, data: pd.DataFrame, market_condition: str) -> Optional[Dict]:
        """Generate optimized trend signals with trailing stop preparation"""
        if len(data) < 60 or self._is_cooldown_active():
            return None
        
        # Only trade in trending markets
        if market_condition not in ["TRENDING", "STRONG_TREND"]:
            return None
        
        close = data['close']
        rsi = self.calculate_rsi(close)
        fast_ema, slow_ema, trend = self.calculate_emas(close)
        momentum = self.calculate_trend_momentum(close, fast_ema)
        atr_value = self.calculate_atr(data)
        price = close.iloc[-1]
        
        if pd.isna(rsi) or trend == 'NEUTRAL':
            return None
        
        # Momentum confirmation if required
        if self.config['momentum_confirmation'] and abs(momentum) < 0.0005:
            return None
        
        signal = None
        
        # Enhanced trend following signals
        if trend == 'UPTREND' and momentum > 0:
            # Long signal on RSI pullback in uptrend
            if (self.config['uptrend_rsi_low'] <= rsi <= self.config['uptrend_rsi_high'] or
                (rsi > 50 and momentum > 0.002)):  # Strong momentum override
                signal = self._create_trend_signal('BUY', trend, rsi, price, data, 
                                                 fast_ema, slow_ema, momentum, atr_value, market_condition)
                
        elif trend == 'DOWNTREND' and momentum < 0:
            # Short signal on RSI pullback in downtrend
            if (self.config['downtrend_rsi_low'] <= rsi <= self.config['downtrend_rsi_high'] or
                (rsi < 50 and momentum < -0.002)):  # Strong momentum override
                signal = self._create_trend_signal('SELL', trend, rsi, price, data, 
                                                 fast_ema, slow_ema, momentum, atr_value, market_condition)
        
        if signal:
            self.last_signal_time = datetime.now()
        
        return signal
    
    def _create_trend_signal(self, action: str, trend: str, rsi: float, price: float, 
                           data: pd.DataFrame, fast_ema: float, slow_ema: float, 
                           momentum: float, atr_value: float, market_condition: str) -> Dict:
        """Create optimized trend signal with trailing stop setup"""
        
        # Use medium window for structure identification
        window = data.tail(40)
        
        if action == 'BUY':
            # Use EMA and recent structure for stop
            ema_stop = fast_ema * 0.994  # 0.6% below fast EMA
            swing_low = window['low'].min()
            structure_stop = max(swing_low * 0.997, ema_stop)  # Choose closer stop
            level = swing_low
        else:
            # Use EMA and recent structure for stop
            ema_stop = fast_ema * 1.006  # 0.6% above fast EMA
            swing_high = window['high'].max()
            structure_stop = min(swing_high * 1.003, ema_stop)  # Choose closer stop
            level = swing_high
        
        # Validate stop distance for 2R setup
        risk_distance = abs(price - structure_stop) / price
        if not (0.010 <= risk_distance <= 0.040):  # 1% to 4% risk range
            return None
        
        # Calculate 2R target
        reward_distance = risk_distance * self.config['risk_reward_ratio']
        
        if action == 'BUY':
            target_price = price + (price * reward_distance)
        else:
            target_price = price - (price * reward_distance)
        
        # Fee efficiency validation
        estimated_position = 6000  # Larger position for trends
        fee_cost = estimated_position * float(self.total_cost_rate)
        gross_profit_target = estimated_position * reward_distance
        fee_efficiency_ratio = gross_profit_target / fee_cost if fee_cost > 0 else 0
        
        if fee_efficiency_ratio < 20:  # Require 20x fee efficiency for trends
            return None
        
        # Calculate 1.2R breakeven threshold for trailing activation
        breakeven_1_2r = estimated_position * float(self.config['risk_percentage']) * self.config['breakeven_threshold']
        breakeven_net = breakeven_1_2r - fee_cost
        
        # Calculate trailing stop price (for reference)
        trailing_stop_price = self.calculate_trailing_stop_price(price, action, atr_value)
        
        # Optimized confidence calculation
        base_confidence = 70
        
        # Trend strength bonus
        trend_strength = abs(fast_ema - slow_ema) / slow_ema * 100
        trend_bonus = min(trend_strength * 1.5, 12)
        
        # Momentum bonus
        momentum_strength = abs(momentum) * 1000
        momentum_bonus = min(momentum_strength * 0.8, 10)
        
        # RSI positioning bonus (pullback quality)
        if action == 'BUY':
            rsi_bonus = max(0, (50 - rsi) * 0.2) if rsi < 50 else 0
        else:
            rsi_bonus = max(0, (rsi - 50) * 0.2) if rsi > 50 else 0
        
        # Market condition bonus
        condition_bonus = 8 if market_condition == "STRONG_TREND" else 5
        
        confidence = base_confidence + trend_bonus + momentum_bonus + rsi_bonus + condition_bonus
        confidence = np.clip(confidence, 70, 95)
        
        return {
            'action': action,
            'strategy': 'TREND',
            'setup_type': 'Optimized 2R with 1.2R Trailing',
            'trend': trend,
            'market_condition': market_condition,
            'rsi': round(rsi, 1),
            'fast_ema': round(fast_ema, 4),
            'slow_ema': round(slow_ema, 4),
            'momentum': round(momentum * 100, 2),
            'atr': round(atr_value, 4),
            'price': price,
            'structure_stop': structure_stop,
            'target_price': target_price,
            'level': level,
            'risk_reward_ratio': self.config['risk_reward_ratio'],
            'risk_percentage': round(risk_distance * 100, 2),
            'reward_percentage': round(reward_distance * 100, 2),
            'signal_type': f"trend_{action.lower()}",
            'confidence': round(confidence, 1),
            'max_hold_seconds': self.config['max_hold_seconds'],
            'timeframe': '15m',
            'trailing_stop_config': {
                'breakeven_threshold': f"{self.config['breakeven_threshold']}R",
                'breakeven_dollar_threshold': f"${breakeven_net:.2f}",
                'atr_multiplier': self.config['trailing_atr_multiplier'],
                'current_atr': round(atr_value, 4),
                'trailing_distance': f"${atr_value * self.config['trailing_atr_multiplier']:.2f}" if atr_value > 0 else f"{self.config['trailing_fallback_pct']*100:.1f}%",
                'example_trailing_stop': round(trailing_stop_price, 2),
                'max_drawdown_trigger': f"{self.config['max_drawdown_from_peak']*100:.0f}% from peak"
            },
            'fee_efficiency': {
                'estimated_fees': f"${fee_cost:.2f}",
                'gross_target': f"${gross_profit_target:.2f}",
                'efficiency_ratio': f"{fee_efficiency_ratio:.1f}x",
                'fee_percentage': f"{(fee_cost/gross_profit_target)*100:.1f}%"
            },
            'corrected_breakeven': {
                'theoretical_win_rate': f"{self.breakeven_rate*100:.2f}%",
                'formula': 'Win_Rate = (Risk + Fee) / (Risk + Reward)',
                'calculation': f"({risk_distance*100:.2f}% + {float(self.total_cost_rate)*100:.3f}%) / ({risk_distance*100:.2f}% + {reward_distance*100:.2f}%)",
                'improvement': 'Much better than range strategy 41.24% requirement'
            }
        }
    
    def _is_cooldown_active(self) -> bool:
        """Check if cooldown is active"""
        if not self.last_signal_time:
            return False
        return (datetime.now() - self.last_signal_time).total_seconds() < self.config['cooldown_seconds']
    
    def get_strategy_info(self) -> Dict:
        """Get comprehensive strategy information"""
        return {
            'name': 'Optimized Trend Strategy',
            'type': 'TREND',
            'setup': '2R with 1.2R Breakeven + 0.5 ATR Trailing',
            'timeframe': '15m', 
            'config': self.config,
            'corrected_breakeven': {
                'win_rate_required': f"{self.breakeven_rate*100:.2f}%",
                'formula': 'Win_Rate = (Risk + Fee) / (Risk + Reward)',
                'advantage': f"{(41.24 - self.breakeven_rate*100):.2f}% easier than range strategy"
            },
            'trailing_stop_system': {
                'activation': '1.2R breakeven move',
                'method': '0.5 ATR or 0.5% fallback',
                'drawdown_trigger': '20% from peak profit',
                'purpose': 'Capture outsized trend moves beyond 2R'
            },
            'fee_model': {
                'maker_fee': f"{float(self.maker_fee_rate)*100:.3f}%",
                'taker_fee': f"{float(self.taker_fee_rate)*100:.3f}%",
                'blended_fee': f"{float(self.blended_fee_rate)*100:.3f}%",
                'total_cost': f"{float(self.total_cost_rate)*100:.3f}%"
            },
            'dynamic_sizing': {
                'fee_target': f"{self.config['fee_target_percentage']*100:.1f}% of gross profit",
                'position_range': f"${self.config['min_position_usdt']} - ${self.config['max_position_usdt']}",
                'efficiency_requirement': '20x profit-to-fee ratio minimum'
            },
            'optimizations': {
                'ema_dynamic_stops': 'EMA-based stop placement',
                'momentum_confirmation': 'Requires directional momentum',
                'rsi_pullback_entry': 'Enter on pullbacks in trend direction',
                'atr_trailing_stops': 'ATR-based trailing for outsized moves',
                'limit_first_execution': 'Attempts maker fills before taker'
            },
            'expected_performance': {
                'theoretical_win_rate': f"{self.breakeven_rate*100:.2f}%",
                'target_win_rate': '38-42%',
                'risk_reward': f'1:{self.config["risk_reward_ratio"]} base + trailing upside',
                'hold_time': f'Up to {self.config["max_hold_seconds"]/60:.0f} minutes'
            },
            'description': f'Fee-optimized trend following: RSI({self.config["rsi_length"]}) + EMA({self.config["fast_ema"]}/{self.config["slow_ema"]}) with 2R targets, 1.2R trailing activation, and {self.breakeven_rate*100:.2f}% break-even requirement'
        }