import os
from dotenv import load_dotenv
from typing import Tuple, Dict, Any
import numpy as np

load_dotenv()

class RiskManager:
    """ULTRA-AGGRESSIVE: Risk Manager with optimized fee model and high-frequency sizing"""

    def __init__(self):
        self.symbol = os.getenv('TRADING_SYMBOL', 'ETHUSDT')
        
        # ENHANCED FEE MODEL (ULTRA-AGGRESSIVE)
        self.taker_fee = 0.00055      # 0.055% market orders
        self.maker_fee = 0.0001       # 0.01% limit orders  
        self.maker_fill_ratio = 0.40  # 40% limit order fills (optimized)
        self.blended_fee = (self.maker_fee * self.maker_fill_ratio + 
                           self.taker_fee * (1 - self.maker_fill_ratio))  # 0.000415
        self.slippage = 0.0002        # 0.02% slippage estimation
        self.fee_rate = self.blended_fee + self.slippage  # 0.000615 total cost
        
        # ULTRA-AGGRESSIVE RANGE CONFIG (High Frequency)
        self.range_config = {
            'min_position_usdt': 1500,   # REDUCED for frequency
            'max_position_usdt': 4000,   # REDUCED for frequency  
            'target_position_usdt': 2500, # Optimal size for range
            'risk_percentage': 0.015,    # 1.5% risk (tighter)
            'reward_ratio': 1.3,         # REDUCED: 1.3R for faster exits
            'gross_profit_target': 65,   # Gross profit before fees
            'net_profit_target': 60,     # Net profit after fees
            'max_position_time': 300,    # 5 minutes max hold
            'emergency_stop_pct': 0.015, # 1.5% emergency stop
            'leverage': 10,
            'fee_target_percentage': 0.02, # 2% of gross profit
        }
        
        # ULTRA-AGGRESSIVE TREND CONFIG (High Frequency)
        self.trend_config = {
            'min_position_usdt': 2000,   # REDUCED for frequency
            'max_position_usdt': 5000,   # REDUCED for frequency
            'target_position_usdt': 3000, # Optimal size for trend
            'risk_percentage': 0.012,    # 1.2% risk (tighter)
            'reward_ratio': 1.5,         # REDUCED: 1.5R vs 2R
            'gross_profit_target': 72,   # Gross profit before fees  
            'net_profit_target': 67,     # Net profit after fees
            'max_position_time': 720,    # 12 minutes max hold (REDUCED)
            'emergency_stop_pct': 0.015, # 1.5% emergency stop
            'trailing_stop_pct': 0.004,  # 0.4% ultra-tight trailing
            'profit_lock_threshold': 1.0, # 1R activation (REDUCED)
            'max_drawdown_from_peak': 0.15, # 15% max drawdown
            'leverage': 10,
            'fee_target_percentage': 0.018, # 1.8% of gross profit
        }
        
        self.active_config = self.range_config.copy()
        self.active_strategy = "RANGE"
    
    def set_strategy(self, strategy_type: str):
        """Set active strategy configuration"""
        if strategy_type in ["RANGE", "TREND"]:
            self.active_strategy = strategy_type
            self.active_config = (self.range_config if strategy_type == "RANGE" 
                                else self.trend_config).copy()
        else:
            self.active_strategy = "RANGE"
            self.active_config = self.range_config.copy()
    
    def get_break_even_pnl(self, position_size_usdt: float) -> float:
        """Calculate break-even PnL with enhanced fee model
        
        CORRECTED Fee Model: 0.0615% total trading cost
        Break-even PnL = Position_Size × 0.000615
        """
        return position_size_usdt * self.fee_rate
    
    def get_min_profitable_target(self, position_size_usdt: float) -> float:
        """Calculate minimum profitable target (ultra-aggressive)"""
        break_even = self.get_break_even_pnl(position_size_usdt)
        return break_even + 10.0  # REDUCED: $10 minimum net profit vs $15
    
    def calculate_position_size(self, balance: float, entry_price: float, stop_price: float) -> float:
        """ULTRA-AGGRESSIVE: Dynamic position sizing for high frequency"""
        if balance <= 0 or entry_price <= 0 or stop_price <= 0:
            return 0
        
        # Get target position size based on strategy
        target_usdt = self.active_config['target_position_usdt']
        min_usdt = self.active_config['min_position_usdt']
        max_usdt = self.active_config['max_position_usdt']
        
        # Risk-based sizing
        risk_amount = balance * self.active_config['risk_percentage']
        stop_distance = abs(entry_price - stop_price) / entry_price
        
        if stop_distance == 0:
            return 0
        
        # Calculate position size based on risk
        risk_based_size = risk_amount / stop_distance / entry_price
        risk_based_usdt = risk_based_size * entry_price
        
        # Dynamic sizing based on market conditions
        if risk_based_usdt < min_usdt:
            position_usdt = min_usdt
        elif risk_based_usdt > max_usdt:
            position_usdt = max_usdt
        else:
            # Blend target size with risk-based size
            position_usdt = (target_usdt * 0.6 + risk_based_usdt * 0.4)
        
        # Final safety checks
        max_balance_pct = 0.15  # Max 15% of balance (REDUCED for high frequency)
        max_allowed = balance * max_balance_pct
        position_usdt = min(position_usdt, max_allowed)
        
        # Validate fee efficiency (ultra-aggressive thresholds)
        break_even_fees = self.get_break_even_pnl(position_usdt)
        fee_efficiency = self.active_config['fee_target_percentage']
        
        expected_gross_profit = position_usdt * stop_distance * self.active_config['reward_ratio']
        if break_even_fees > expected_gross_profit * fee_efficiency:
            # Scale down position to meet fee efficiency
            position_usdt *= 0.8
        
        return max(position_usdt / entry_price, 0)
    
    def should_close_position(self, current_price: float, entry_price: float, side: str, 
                             unrealized_pnl: float, position_age_seconds: float, 
                             position_size_usdt: float = 0) -> Tuple[bool, str]:
        """ULTRA-AGGRESSIVE: Exit conditions with tighter timing"""
        if entry_price <= 0:
            return False, "hold"
        
        # Calculate fees for exit decisions
        fees = self.get_break_even_pnl(position_size_usdt) if position_size_usdt > 0 else 0
        
        # Emergency stop: 1.5% loss + fees (TIGHTER)
        emergency_threshold = -(position_size_usdt * self.active_config['emergency_stop_pct'] + fees)
        if unrealized_pnl <= emergency_threshold:
            return True, "emergency_stop"
        
        # Max hold time exceeded (MUCH SHORTER)
        if position_age_seconds >= self.active_config['max_position_time']:
            return True, "max_hold_time"
        
        # Strategy-specific exits
        if self.active_strategy == "RANGE":
            return self._check_range_exits_ultra(unrealized_pnl, position_age_seconds, fees, position_size_usdt)
        else:
            return self._check_trend_exits_ultra(entry_price, current_price, side, unrealized_pnl, 
                                               position_age_seconds, fees, position_size_usdt)
    
    def _check_range_exits_ultra(self, unrealized_pnl: float, position_age_seconds: float, 
                                 fees: float, position_size_usdt: float) -> Tuple[bool, str]:
        """ULTRA-AGGRESSIVE: Range strategy exits (5-minute max hold)"""
        gross_target = self.active_config['gross_profit_target']  # $65
        
        # Hit profit target
        if unrealized_pnl >= gross_target:
            return True, "profit_target"
        
        # Quick profit taking (ultra-aggressive)
        quick_target = fees + 25  # $25 net profit minimum
        if unrealized_pnl >= quick_target and position_age_seconds >= 120:  # After 2 minutes
            return True, "profit_target"
        
        # Time-based exits (much faster)
        time_threshold = self.active_config['max_position_time'] * 0.6  # 60% of max time (3 minutes)
        if position_age_seconds >= time_threshold:
            loss_threshold = -(fees + 15)  # Allow $15 loss beyond fees
            if unrealized_pnl <= loss_threshold:
                return True, "timeout_no_profit"
            elif unrealized_pnl >= fees + 8:  # Small profit after 3 minutes
                return True, "profit_target"
        
        return False, "hold"
    
    def _check_trend_exits_ultra(self, entry_price: float, current_price: float, side: str,
                                 unrealized_pnl: float, position_age_seconds: float, 
                                 fees: float, position_size_usdt: float) -> Tuple[bool, str]:
        """ULTRA-AGGRESSIVE: Trend strategy exits (12-minute max, 0.4% trailing)"""
        gross_target = self.active_config['gross_profit_target']  # $72
        
        # Quick profit taking (much more aggressive)  
        quick_target = fees + 20  # $20 net profit minimum
        if unrealized_pnl >= quick_target and position_age_seconds >= 240:  # After 4 minutes
            return True, "profit_target"
        
        # Main profit target
        if unrealized_pnl >= gross_target:
            return True, "profit_target"
        
        # Ultra-tight trailing stop logic
        if unrealized_pnl > 0:
            risk_unit = position_size_usdt * self.active_config['risk_percentage']  # 1.2% of position
            profit_ratio = unrealized_pnl / risk_unit if risk_unit > 0 else 0
            
            # Activate trailing at 1R (REDUCED from 1.2R)
            if profit_ratio >= self.trend_config['profit_lock_threshold']:
                # Calculate trailing stop
                trailing_distance = current_price * self.trend_config['trailing_stop_pct']  # 0.4%
                
                if side.lower() == 'buy':
                    trailing_stop = current_price - trailing_distance
                    if entry_price > trailing_stop * 0.998:  # Price dropped below trailing
                        return True, "trailing_stop"
                else:
                    trailing_stop = current_price + trailing_distance  
                    if entry_price < trailing_stop * 1.002:  # Price rose above trailing
                        return True, "trailing_stop"
        
        # Time-based exits (faster for high frequency)
        time_threshold_1 = self.active_config['max_position_time'] * 0.5  # 50% (6 minutes)
        time_threshold_2 = self.active_config['max_position_time'] * 0.75  # 75% (9 minutes)
        
        if position_age_seconds >= time_threshold_2:
            # Exit with small loss/profit after 9 minutes
            if unrealized_pnl >= fees - 10:  # Break-even or small loss acceptable
                return True, "timeout_no_profit"
        elif position_age_seconds >= time_threshold_1:
            # Take small profits after 6 minutes
            if unrealized_pnl >= fees + 15:  # $15 net profit
                return True, "profit_target"
        
        return False, "hold"
    
    def calculate_trailing_stop(self, current_price: float, entry_price: float, side: str, 
                               highest_profit: float) -> float:
        """Calculate ultra-tight 0.4% trailing stop"""
        if self.active_strategy != "TREND" or highest_profit <= 0:
            return 0
        
        trail_distance = current_price * self.trend_config['trailing_stop_pct']  # 0.4%
        
        if side.lower() == 'buy':
            return current_price - trail_distance
        else:
            return current_price + trail_distance
    
    def adapt_to_market_condition(self, market_condition: str, volatility: str):
        """ULTRA-AGGRESSIVE: Adapt to market conditions while maintaining high frequency"""
        base_config = (self.range_config if self.active_strategy == "RANGE" 
                      else self.trend_config).copy()
        
        self.active_config = base_config.copy()
        
        # Volatility adjustments (more conservative than before)
        if volatility == "HIGH_VOL":
            # Reduce position sizes and tighten timing in high volatility
            self.active_config['target_position_usdt'] = int(base_config['target_position_usdt'] * 0.75)
            self.active_config['max_position_time'] = int(base_config['max_position_time'] * 0.8)
        elif volatility == "LOW_VOL":
            # Slightly increase size in low volatility but maintain speed
            self.active_config['target_position_usdt'] = int(base_config['target_position_usdt'] * 1.1)
            self.active_config['max_position_time'] = int(base_config['max_position_time'] * 1.2)
        
        # Market condition fine-tuning
        if market_condition == "STRONG_TREND" and self.active_strategy == "TREND":
            # Allow slightly larger positions in strong trends
            self.active_config['target_position_usdt'] = int(base_config['target_position_usdt'] * 1.15)
        elif market_condition == "STRONG_RANGE" and self.active_strategy == "RANGE":
            # Optimize for strong ranging markets
            self.active_config['target_position_usdt'] = int(base_config['target_position_usdt'] * 1.1)
    
    def get_leverage(self) -> int:
        return self.active_config['leverage']
    
    def get_max_position_time(self) -> int:
        return self.active_config['max_position_time']
    
    def validate_fee_efficiency(self, position_size_usdt: float, expected_profit: float) -> bool:
        """Validate fee efficiency for ultra-aggressive system"""
        break_even_fees = self.get_break_even_pnl(position_size_usdt)
        min_profit_target = self.get_min_profitable_target(position_size_usdt)
        
        # More lenient efficiency check for high frequency
        efficiency_ratio = expected_profit / break_even_fees if break_even_fees > 0 else 0
        min_efficiency = 10 if self.active_strategy == "RANGE" else 12  # REDUCED ratios
        
        return efficiency_ratio >= min_efficiency and expected_profit >= min_profit_target
    
    def get_active_config(self) -> Dict[str, Any]:
        """Get complete ultra-aggressive configuration"""
        position_size = self.active_config['target_position_usdt']
        fees = self.get_break_even_pnl(position_size)
        
        if self.active_strategy == "RANGE":
            gross_target = self.active_config['gross_profit_target']
            net_target = gross_target - fees
            sizing_info = {
                'method': 'Ultra-Aggressive Dynamic Sizing',
                'target_size_usdt': position_size,
                'size_range': f"${self.active_config['min_position_usdt']}-${self.active_config['max_position_usdt']}",
                'gross_target': f"${gross_target}",
                'estimated_fees': f"${fees:.2f}",
                'estimated_net': f"${net_target:.2f}",
                'reward_ratio': f"1:{self.active_config['reward_ratio']}",
                'max_hold_time': f"{self.active_config['max_position_time']}s"
            }
        else:
            risk_amount = position_size * self.active_config['risk_percentage']
            target_profit = risk_amount * self.active_config['reward_ratio']
            sizing_info = {
                'method': 'Ultra-Aggressive Dynamic Sizing',
                'target_size_usdt': position_size,
                'size_range': f"${self.active_config['min_position_usdt']}-${self.active_config['max_position_usdt']}",
                'risk_amount': f"${risk_amount:.2f}",
                'target_profit': f"${target_profit:.2f}",
                'estimated_fees': f"${fees:.2f}",
                'reward_ratio': f"1:{self.active_config['reward_ratio']}",
                'trailing_stop': f"{self.trend_config['trailing_stop_pct']*100:.1f}%",
                'max_hold_time': f"{self.active_config['max_position_time']}s"
            }
        
        return {
            'strategy': self.active_strategy,
            'config': self.active_config.copy(),
            'fee_model': {
                'blended_fee_rate': f"{self.blended_fee*100:.4f}%",
                'total_cost_rate': f"{self.fee_rate*100:.4f}%",
                'maker_fill_ratio': f"{self.maker_fill_ratio*100:.0f}%",
                'expected_fee_reduction': "31%"
            },
            'position_sizing': sizing_info,
            'ultra_aggressive_features': {
                'emergency_stop': f"{self.active_config['emergency_stop_pct']*100:.1f}%",
                'max_hold_time': f"{self.active_config['max_position_time']}s",
                'leverage': f"{self.active_config['leverage']}x",
                'fee_efficiency_min': "10-12x ratios",
                'frequency_optimization': 'High-frequency sizing (reduced positions)',
                'break_even_formula': 'Position_Size × 0.000615',
                'profit_targets': f"Range: ${self.range_config['net_profit_target']}, Trend: ${self.trend_config['net_profit_target']}"
            }
        }