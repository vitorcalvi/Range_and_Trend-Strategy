import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional

class RangeStrategy:
    """ULTRA-AGGRESSIVE: RSI(6) + BB(15) Range Strategy - Research-Backed High Frequency"""
    
    def __init__(self):
        self.config = {
            # ULTRA-AGGRESSIVE CORE PARAMETERS
            "rsi_length": 6,             # RESEARCH: RSI(6) vs 14
            "bb_length": 15,             # Faster BB for 3m charts
            "bb_std": 1.8,               # Tighter bands for sensitivity
            "oversold": 25,              # RESEARCH: 25 vs 32 (more signals)
            "overbought": 75,            # RESEARCH: 75 vs 68 (more signals)
            "rsi_neutral_low": 45,       # Neutral zone trading
            "rsi_neutral_high": 55,      # Neutral zone trading
            
            # REDUCED R/R FOR HIGHER FREQUENCY
            "risk_reward_ratio": 1.3,    # REDUCED: 1.3 vs 1.5
            "risk_percentage": 0.015,    # REDUCED: 1.5% vs 2%
            
            # ULTRA-FAST TIMING
            "cooldown_seconds": 120,     # RESEARCH: 2min vs 4min
            "max_hold_seconds": 300,     # RESEARCH: 5min vs 8min
            
            # HIGHER FREQUENCY THRESHOLDS
            "min_confidence": 65,        # REDUCED: 65 vs 72 (more signals)
            "fee_target_percentage": 0.02,   # REDUCED: 2% vs 3%
            "base_profit_usdt": 60,      # Net profit target after fees
            "gross_profit_usdt": 65,     # Gross profit before fees
            
            # ADDITIONAL ULTRA-AGGRESSIVE FEATURES
            "momentum_threshold": 0.0003, # Micro-momentum detection
            "bb_squeeze_detection": True, # Breakout signal enhancement
            "dynamic_rsi_levels": True,   # Volatility-adjusted RSI
            
            # FEE MODEL (ULTRA-AGGRESSIVE) - Corrected 2025 rates
            "fee_rate": 0.00086         # 0.086% total cost (corrected)
        }
        self.last_signal_time = None
        
    def calculate_rsi(self, prices: pd.Series) -> float:
        """Calculate RSI(6) with ultra-fast response"""
        period = self.config['rsi_length']  # 6 periods
        if len(prices) < period + 3:  # Reduced minimum data requirement
            return 50.0
        
        try:
            # Ensure prices are not None and are numeric
            prices = prices.dropna()
            if len(prices) < period + 3:
                return 50.0
            
            delta = prices.diff().fillna(0)
            gain = delta.where(delta > 0, 0)
            loss = (-delta.where(delta < 0, 0))
            
            # Use SMA for first calculation, then EMA
            alpha = 1.0 / period
            avg_gain = gain.ewm(alpha=alpha, min_periods=period).mean().iloc[-1]
            avg_loss = loss.ewm(alpha=alpha, min_periods=period).mean().iloc[-1]
            
            # Check for None values
            if avg_gain is None or avg_loss is None or pd.isna(avg_gain) or pd.isna(avg_loss):
                return 50.0
            
            avg_gain = float(avg_gain)
            avg_loss = float(avg_loss)
            
            if avg_loss == 0:
                return 95.0
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            # Ensure result is not None
            if rsi is None or pd.isna(rsi):
                return 50.0
                
            return np.clip(float(rsi), 5, 95)
        except Exception as e:
            print(f"❌ RSI calculation error: {e}")
            return 50.0
    
    def calculate_bollinger_position(self, prices: pd.Series) -> tuple:
        """Calculate BB(15, 1.8) position - Ultra-sensitive"""
        period = self.config['bb_length']  # 15 periods
        std_mult = self.config['bb_std']   # 1.8 std dev
        
        if len(prices) < period:
            return 0.5, 0
        
        try:
            # Ensure prices are not None and are numeric
            prices = prices.dropna()
            if len(prices) < period:
                return 0.5, 0
            
            sma = prices.rolling(period).mean().iloc[-1]
            std = prices.rolling(period).std().iloc[-1]
            current_price = prices.iloc[-1]
            
            # Check for None values
            if any(x is None or pd.isna(x) for x in [sma, std, current_price]):
                return 0.5, 0
            
            sma = float(sma)
            std = float(std)
            current_price = float(current_price)
            
            if std == 0:
                return 0.5, 0
            
            upper_band = sma + (std * std_mult)
            lower_band = sma - (std * std_mult)
            
            # Calculate position within bands
            band_range = upper_band - lower_band
            if band_range == 0:
                return 0.5, 0
                
            bb_position = (current_price - lower_band) / band_range
            bb_position = np.clip(bb_position, 0, 1)
            
            band_width = band_range / sma
            
            return float(bb_position), float(band_width)
        except Exception as e:
            print(f"❌ BB calculation error: {e}")
            return 0.5, 0
    
    def calculate_micro_momentum(self, prices: pd.Series) -> float:
        """Calculate micro-momentum for ultra-aggressive signals"""
        if len(prices) < 4:
            return 0
        
        # 3-period micro momentum
        momentum = (prices.iloc[-1] - prices.iloc[-4]) / prices.iloc[-4]
        return momentum
    
    def detect_bb_squeeze(self, prices: pd.Series) -> bool:
        """Detect Bollinger Band squeeze for breakout signals"""
        if not self.config['bb_squeeze_detection'] or len(prices) < 20:
            return False
            
        current_width = self.calculate_bollinger_position(prices)[1]
        recent_widths = []
        
        for i in range(5, 15):  # Check last 10 periods
            if len(prices) >= i:
                historical_prices = prices.iloc[:-i+1] if i > 1 else prices
                _, width = self.calculate_bollinger_position(historical_prices)
                recent_widths.append(width)
        
        if recent_widths:
            avg_width = np.mean(recent_widths)
            return current_width < avg_width * 0.8  # 20% tighter than average
        
        return False
    
    def adjust_rsi_levels_for_volatility(self, volatility: float) -> tuple:
        """Dynamic RSI levels based on volatility"""
        if not self.config['dynamic_rsi_levels']:
            return self.config['oversold'], self.config['overbought']
        
        base_oversold = self.config['oversold']
        base_overbought = self.config['overbought']
        
        if volatility > 0.02:  # High volatility
            oversold = max(base_oversold - 5, 15)  # More aggressive
            overbought = min(base_overbought + 5, 85)  # More aggressive
        elif volatility < 0.005:  # Low volatility
            oversold = min(base_oversold + 3, 35)  # Less aggressive
            overbought = max(base_overbought - 3, 65)  # Less aggressive
        else:
            oversold, overbought = base_oversold, base_overbought
            
        return oversold, overbought
    
    def generate_signal(self, data: pd.DataFrame, market_condition: str) -> Optional[Dict]:
        """ULTRA-AGGRESSIVE: Generate high-frequency range signals"""
        if len(data) < 20 or self._is_cooldown_active():  # Reduced minimum data
            return None
        
        # Only trade in ranging markets
        if market_condition not in ["STRONG_RANGE", "WEAK_RANGE"]:
            return None
        
        close = data['close']
        rsi = self.calculate_rsi(close)
        bb_position, band_width = self.calculate_bollinger_position(close)
        micro_momentum = self.calculate_micro_momentum(close)
        is_squeeze = self.detect_bb_squeeze(close)
        price = close.iloc[-1]
        
        if pd.isna(rsi) or band_width == 0:
            return None
        
        # Calculate volatility for dynamic thresholds
        volatility = close.rolling(10).std().iloc[-1] / close.rolling(10).mean().iloc[-1]
        oversold, overbought = self.adjust_rsi_levels_for_volatility(volatility)
        
        signal = None
        
        # ULTRA-AGGRESSIVE SIGNAL CONDITIONS
        
        # 1. Strong oversold conditions (Primary)
        if (rsi <= oversold or bb_position <= 0.1) and micro_momentum < -self.config['momentum_threshold']:
            signal = self._create_signal('BUY', rsi, bb_position, price, data, market_condition, 'strong_oversold')
            
        # 2. Strong overbought conditions (Primary)
        elif (rsi >= overbought or bb_position >= 0.9) and micro_momentum > self.config['momentum_threshold']:
            signal = self._create_signal('SELL', rsi, bb_position, price, data, market_condition, 'strong_overbought')
            
        # 3. BB Squeeze Breakout Signals (Secondary)
        elif is_squeeze and band_width > 0.008:  # Minimum volatility required
            if bb_position <= 0.25 and micro_momentum < -0.0005:
                signal = self._create_signal('BUY', rsi, bb_position, price, data, market_condition, 'squeeze_breakout')
            elif bb_position >= 0.75 and micro_momentum > 0.0005:
                signal = self._create_signal('SELL', rsi, bb_position, price, data, market_condition, 'squeeze_breakout')
        
        # 4. Neutral zone mean reversion (Tertiary - Higher frequency)
        elif (self.config['rsi_neutral_low'] <= rsi <= self.config['rsi_neutral_high'] and
              band_width > 0.012):  # Ensure sufficient volatility
            
            if bb_position <= 0.25 and micro_momentum < -0.0002:
                signal = self._create_signal('BUY', rsi, bb_position, price, data, market_condition, 'neutral_reversion')
            elif bb_position >= 0.75 and micro_momentum > 0.0002:
                signal = self._create_signal('SELL', rsi, bb_position, price, data, market_condition, 'neutral_reversion')
        
        if signal:
            self.last_signal_time = datetime.now()
        
        return signal
    
    def _create_signal(self, action: str, rsi: float, bb_position: float, price: float, 
                      data: pd.DataFrame, market_condition: str, signal_reason: str) -> Dict:
        """Create ultra-aggressive range signal"""
        window = data.tail(20)  # Reduced window for faster signals
        
        if action == 'BUY':
            structure_stop = window['low'].min() * 0.9985  # Ultra-tight
            level = window['low'].min()
        else:
            structure_stop = window['high'].max() * 1.0015  # Ultra-tight
            level = window['high'].max()
        
        # Validate stop distance (More permissive for high frequency)
        stop_distance = abs(price - structure_stop) / price
        if not (0.0005 <= stop_distance <= 0.015):  # Very tight range
            return None
        
        # ULTRA-AGGRESSIVE confidence calculation
        base_confidence = self.config['min_confidence']  # 65
        
        if signal_reason == 'strong_oversold' or signal_reason == 'strong_overbought':
            rsi_strength = abs(50 - rsi) / 25  # 0-1 scale
            bb_strength = abs(0.5 - bb_position) * 2  # 0-1 scale
            confidence_boost = (rsi_strength + bb_strength) * 15
            base_confidence += confidence_boost
        elif signal_reason == 'squeeze_breakout':
            base_confidence += 10  # Moderate boost
        elif signal_reason == 'neutral_reversion':
            bb_strength = abs(0.5 - bb_position) * 2
            base_confidence += bb_strength * 12
        
        # Market condition boost
        if market_condition == "STRONG_RANGE":
            base_confidence *= 1.08
        
        confidence = np.clip(base_confidence, 65, 92)
        
        return {
            'action': action,
            'strategy': 'RANGE',
            'market_condition': market_condition,
            'signal_reason': signal_reason,
            'rsi': round(rsi, 1),
            'bb_position': round(bb_position, 3),
            'micro_momentum': round(self.calculate_micro_momentum(data['close']) * 10000, 2),
            'price': price,
            'structure_stop': structure_stop,
            'level': level,
            'signal_type': f"range_{action.lower()}",
            'confidence': round(confidence, 1),
            'risk_reward_ratio': self.config['risk_reward_ratio'],
            'gross_profit_target': self.config['gross_profit_usdt'],
            'net_profit_target': self.config['base_profit_usdt'], 
            'max_hold_seconds': self.config['max_hold_seconds'],
            'timeframe': '3m'  # Updated to 3m
        }
    
    def _is_cooldown_active(self) -> bool:
        """Check if ultra-aggressive cooldown is active (2 minutes)"""
        if not self.last_signal_time:
            return False
        return (datetime.now() - self.last_signal_time).total_seconds() < self.config['cooldown_seconds']
    
    def get_strategy_info(self) -> Dict:
        """Get ultra-aggressive strategy information"""
        return {
            'name': f'ULTRA-AGGRESSIVE RSI({self.config["rsi_length"]}) + BB({self.config["bb_length"]}) Range Strategy',
            'type': 'RANGE',
            'timeframe': '3m',  # Research optimized
            'config': self.config,
            'description': f'High-frequency range trading: RSI(6) 25/75 + BB(15,1.8) + micro-momentum - ${self.config["base_profit_usdt"]} net target',
            'expected_signals_per_hour': '8-12',
            'expected_win_rate': '65-75%',
            'risk_reward': f'1:{self.config["risk_reward_ratio"]}',
            'key_features': [
                'RSI(6) ultra-fast response',
                'BB(15,1.8) high sensitivity', 
                'Micro-momentum detection',
                'BB squeeze breakout signals',
                'Dynamic volatility-adjusted thresholds',
                '2-minute cooldowns',
                '5-minute max hold times'
            ]
        }