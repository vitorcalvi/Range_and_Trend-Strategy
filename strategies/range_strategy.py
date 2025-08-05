import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Optional

class RangeStrategy:
    """Optimized Range Strategy - Dynamic Sizing & Corrected Break-Even"""
    
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
            "bb_length": 20,
            "bb_std": 2.0,
            
            # Optimized RSI levels for quality signals
            "oversold": 32,           # Slightly less extreme for more opportunities
            "overbought": 68,         # Slightly less extreme for more opportunities
            "rsi_momentum_threshold": 2.0,  # RSI momentum for confirmation
            
            # CORRECTED: 1.5R setup parameters
            "risk_reward_ratio": 1.5,     # 1.5R target
            "risk_percentage": 0.02,      # 2% position risk (1R)
            "min_confidence": 72,         # Optimized confidence threshold
            
            # Dynamic position sizing parameters
            "fee_target_percentage": 0.03,  # Fees should be 3% of expected gross
            "min_position_usdt": 2500,      # Minimum for fee efficiency
            "max_position_usdt": 7500,      # Maximum for safety
            
            # Timing optimizations
            "cooldown_seconds": 240,      # 4 minutes between signals
            "max_hold_seconds": 480,      # 8 minutes max hold
            "quick_exit_threshold": 0.7,  # 70% of max time for quick exits
            
            # Backward compatibility
            "base_profit_usdt": 35,       # Base profit target (legacy)
            
            # Signal quality filters
            "min_volatility": 0.015,      # Minimum BB width for trading
            "max_volatility": 0.08,       # Maximum BB width (too chaotic)
            "signal_strength_min": 0.65,  # Minimum combined signal strength
            
            # Backward compatibility
            "base_profit_usdt": 35,       # Base profit target (legacy)
        }
        self.last_signal_time = None
        
        # Calculate corrected break-even rate
        self._calculate_breakeven_rate()
        
    def _calculate_breakeven_rate(self):
        """Calculate corrected break-even win rate"""
        # CORRECTED FORMULA: Win_Rate = (Risk + Fee) / (Risk + Reward)
        risk_pct = float(self.config['risk_percentage'])      # 2%
        reward_pct = risk_pct * self.config['risk_reward_ratio']  # 3%
        fee_pct = float(self.total_cost_rate)  # ~0.062%
        
        numerator = risk_pct + fee_pct      # 0.02 + 0.00062 = 0.02062
        denominator = risk_pct + reward_pct  # 0.02 + 0.03 = 0.05
        self.breakeven_rate = numerator / denominator  # 0.41240 = 41.24%
        
    def calculate_rsi(self, prices: pd.Series) -> float:
        """Calculate RSI with momentum component"""
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
    
    def calculate_rsi_momentum(self, prices: pd.Series) -> float:
        """Calculate RSI momentum for confirmation"""
        if len(prices) < 20:
            return 0
        
        rsi_series = []
        for i in range(len(prices) - 15, len(prices)):
            if i >= 14:
                rsi_val = self.calculate_rsi(prices.iloc[:i+1])
                rsi_series.append(rsi_val)
        
        if len(rsi_series) < 5:
            return 0
        
        # Calculate 5-period RSI momentum
        momentum = rsi_series[-1] - rsi_series[-5]
        return momentum
    
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
    
    def calculate_signal_strength(self, rsi: float, bb_position: float, band_width: float, 
                                rsi_momentum: float) -> float:
        """Calculate comprehensive signal strength"""
        
        # RSI extremity strength
        if rsi <= self.config['oversold']:
            rsi_strength = (self.config['oversold'] - rsi) / self.config['oversold']
        elif rsi >= self.config['overbought']:
            rsi_strength = (rsi - self.config['overbought']) / (100 - self.config['overbought'])
        else:
            rsi_strength = 0
        
        # RSI momentum confirmation (opposite direction to price)
        momentum_strength = 0
        if rsi <= self.config['oversold'] and rsi_momentum < -self.config['rsi_momentum_threshold']:
            momentum_strength = min(abs(rsi_momentum) / 10, 1.0)
        elif rsi >= self.config['overbought'] and rsi_momentum > self.config['rsi_momentum_threshold']:
            momentum_strength = min(abs(rsi_momentum) / 10, 1.0)
        
        # BB position strength
        bb_strength = abs(bb_position - 0.5) * 2
        
        # Volatility component (sweet spot for ranging)
        vol_strength = 0
        if self.config['min_volatility'] <= band_width <= self.config['max_volatility']:
            # Optimal volatility range
            optimal_vol = (self.config['min_volatility'] + self.config['max_volatility']) / 2
            vol_strength = 1 - abs(band_width - optimal_vol) / optimal_vol
        
        # Combined strength with weights
        overall_strength = (rsi_strength * 0.4 + momentum_strength * 0.2 + 
                          bb_strength * 0.25 + vol_strength * 0.15)
        
        return np.clip(overall_strength, 0, 1)
    
    def calculate_dynamic_position_size(self, balance: float, entry_price: float, 
                                      stop_price: float, signal_strength: float) -> float:
        """Calculate dynamic position size for fee efficiency"""
        if balance <= 0 or entry_price <= 0 or stop_price <= 0:
            return 0
        
        # Calculate risk distance
        risk_distance = abs(entry_price - stop_price) / entry_price
        reward_distance = risk_distance * self.config['risk_reward_ratio']
        
        # Calculate position size for target fee efficiency
        # Target: fees = fee_target_percentage × expected_gross_profit
        fee_target_pct = self.config['fee_target_percentage']
        
        # Expected gross profit = position_size × reward_distance
        # Fee cost = position_size × total_cost_rate
        # Target: position_size × total_cost_rate = fee_target_pct × position_size × reward_distance
        # This is always satisfied, so we optimize based on other factors
        
        # Base position size for fee efficiency
        min_gross_profit = 1000 / fee_target_pct  # $1000 / 3% = $33,333 gross needed
        base_position = min_gross_profit / reward_distance if reward_distance > 0 else 0
        
        # Adjust based on signal strength
        strength_multiplier = 0.7 + (signal_strength * 0.6)  # 0.7 to 1.3 range
        adjusted_position = base_position * strength_multiplier
        
        # Apply limits
        min_position = self.config['min_position_usdt']
        max_position = min(self.config['max_position_usdt'], balance * 0.25)
        
        target_position = np.clip(adjusted_position, min_position, max_position)
        
        return target_position / entry_price
    
    def generate_signal(self, data: pd.DataFrame, market_condition: str) -> Optional[Dict]:
        """Generate optimized signals with dynamic sizing consideration"""
        if len(data) < 30 or self._is_cooldown_active():
            return None
        
        # Only trade in ranging markets
        if market_condition not in ["STRONG_RANGE", "WEAK_RANGE"]:
            return None
        
        rsi = self.calculate_rsi(data['close'])
        rsi_momentum = self.calculate_rsi_momentum(data['close'])
        bb_position, band_width = self.calculate_bollinger_position(data['close'])
        signal_strength = self.calculate_signal_strength(rsi, bb_position, band_width, rsi_momentum)
        price = data['close'].iloc[-1]
        
        if pd.isna(rsi) or band_width == 0:
            return None
        
        # Volatility filter
        if not (self.config['min_volatility'] <= band_width <= self.config['max_volatility']):
            return None
        
        # Signal strength filter
        if signal_strength < self.config['signal_strength_min']:
            return None
        
        signal = None
        
        # Optimized oversold signal
        if (rsi <= self.config['oversold'] or bb_position <= 0.12) and rsi_momentum < 0:
            signal = self._create_optimized_signal('BUY', rsi, bb_position, price, data, 
                                                 market_condition, signal_strength, rsi_momentum)
            
        # Optimized overbought signal
        elif (rsi >= self.config['overbought'] or bb_position >= 0.88) and rsi_momentum > 0:
            signal = self._create_optimized_signal('SELL', rsi, bb_position, price, data, 
                                                 market_condition, signal_strength, rsi_momentum)
        
        if signal:
            self.last_signal_time = datetime.now()
        
        return signal
    
    def _create_optimized_signal(self, action: str, rsi: float, bb_position: float, price: float, 
                               data: pd.DataFrame, market_condition: str, signal_strength: float,
                               rsi_momentum: float) -> Dict:
        """Create optimized signal with dynamic sizing and fee validation"""
        
        # Use longer window for better structure identification
        window = data.tail(60)
        
        if action == 'BUY':
            # More precise stop placement
            recent_lows = window['low'].rolling(10).min()
            structure_stop = recent_lows.iloc[-3:].min() * 0.997  # Use 3-period low with buffer
            level = structure_stop / 0.997  # Remove buffer for level
        else:
            # More precise stop placement
            recent_highs = window['high'].rolling(10).max()
            structure_stop = recent_highs.iloc[-3:].max() * 1.003  # Use 3-period high with buffer
            level = structure_stop / 1.003  # Remove buffer for level
        
        # Validate risk distance for 1.5R setup
        risk_distance = abs(price - structure_stop) / price
        if not (0.012 <= risk_distance <= 0.035):  # 1.2% to 3.5% risk range
            return None
        
        # Calculate targets and fee efficiency
        reward_distance = risk_distance * self.config['risk_reward_ratio']
        
        if action == 'BUY':
            target_price = price + (price * reward_distance)
        else:
            target_price = price - (price * reward_distance)
        
        # Estimate fee efficiency (assuming $5000 position)
        estimated_position = 5000
        fee_cost = estimated_position * float(self.total_cost_rate)
        gross_profit_target = estimated_position * reward_distance
        fee_efficiency_ratio = gross_profit_target / fee_cost if fee_cost > 0 else 0
        
        # Fee efficiency validation
        if fee_efficiency_ratio < 15:  # Require 15x fee efficiency minimum
            return None
        
        # Optimized confidence calculation
        base_confidence = 72
        
        # RSI extremity bonus
        if action == 'BUY':
            rsi_bonus = max(0, (32 - rsi) * 0.7)
        else:
            rsi_bonus = max(0, (rsi - 68) * 0.7)
        
        # RSI momentum bonus (confirmation)
        momentum_bonus = min(abs(rsi_momentum) * 0.3, 8)
        
        # BB position bonus
        bb_bonus = abs(bb_position - 0.5) * 15
        
        # Signal strength bonus
        strength_bonus = signal_strength * 12
        
        # Market condition bonus
        condition_bonus = 6 if market_condition == "STRONG_RANGE" else 3
        
        confidence = base_confidence + rsi_bonus + momentum_bonus + bb_bonus + strength_bonus + condition_bonus
        confidence = np.clip(confidence, 72, 92)
        
        return {
            'action': action,
            'strategy': 'RANGE',
            'setup_type': 'Optimized 1.5R Dynamic',
            'market_condition': market_condition,
            'rsi': round(rsi, 1),
            'rsi_momentum': round(rsi_momentum, 1),
            'bb_position': round(bb_position, 2),
            'bb_width': round(data['close'].rolling(20).std().iloc[-1] / data['close'].iloc[-1] * 2, 4),
            'signal_strength': round(signal_strength, 2),
            'price': price,
            'structure_stop': structure_stop,
            'target_price': target_price,
            'level': level,
            'risk_reward_ratio': self.config['risk_reward_ratio'],
            'risk_percentage': round(risk_distance * 100, 2),
            'reward_percentage': round(reward_distance * 100, 2),
            'signal_type': f"optimized_range_{action.lower()}",
            'confidence': round(confidence, 1),
            'max_hold_seconds': self.config['max_hold_seconds'],
            'timeframe': '1m',
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
                'fee_model': f"Blended: {float(self.blended_fee_rate)*100:.3f}% + {float(self.slippage_rate)*100:.2f}% slippage"
            },
            'dynamic_sizing': {
                'strength_multiplier': 0.7 + (signal_strength * 0.6),
                'min_position': f"${self.config['min_position_usdt']}",
                'max_position': f"${self.config['max_position_usdt']}",
                'fee_target': f"{self.config['fee_target_percentage']*100:.1f}%"
            }
        }
        
    def _is_cooldown_active(self) -> bool:
        """Check if cooldown is active"""
        if not self.last_signal_time:
            return False
        elapsed = (datetime.now() - self.last_signal_time).total_seconds()
        return elapsed < self.config['cooldown_seconds']
    
    def get_strategy_info(self) -> Dict:
        """Get comprehensive strategy information"""
        return {
            'name': 'Optimized Range Strategy',
            'type': 'RANGE',
            'setup': 'Dynamic 1.5R with Fee Optimization',
            'timeframe': '1m',
            'config': self.config,
            'corrected_breakeven': {
                'win_rate_required': f"{self.breakeven_rate*100:.2f}%",
                'formula': 'Win_Rate = (Risk + Fee) / (Risk + Reward)',
                'improvement': 'Corrected from flawed 40.1% calculation'
            },
            'fee_model': {
                'maker_fee': f"{float(self.maker_fee_rate)*100:.3f}%",
                'taker_fee': f"{float(self.taker_fee_rate)*100:.3f}%",
                'blended_fee': f"{float(self.blended_fee_rate)*100:.3f}%",
                'maker_assumption': f"{float(self.maker_fill_ratio)*100:.0f}%",
                'total_cost': f"{float(self.total_cost_rate)*100:.3f}%"
            },
            'dynamic_sizing': {
                'fee_target': f"{self.config['fee_target_percentage']*100:.1f}% of gross profit",
                'position_range': f"${self.config['min_position_usdt']} - ${self.config['max_position_usdt']}",
                'strength_based': 'Position size adjusts based on signal strength'
            },
            'optimizations': {
                'limit_first_orders': 'Attempts maker fills before taker',
                'rsi_momentum_confirmation': 'RSI direction confirms setup',
                'volatility_filtering': 'Optimal BB width range',
                'dynamic_stops': '3-period structure levels',
                'fee_efficiency_validation': '15x minimum profit-to-fee ratio'
            },
            'description': f'Fee-optimized range mean reversion: RSI({self.config["rsi_length"]}) + BB({self.config["bb_length"]}) with dynamic 1.5R targets and corrected {self.breakeven_rate*100:.2f}% break-even'
        }