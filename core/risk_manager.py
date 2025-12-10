import os
from dotenv import load_dotenv
from typing import Tuple, Dict, Any
import numpy as np
from core.fee_calculator import FeeCalculator

load_dotenv()

class RiskManager:
    """Fixed Risk Manager with correct fee model and realistic parameters"""

    def __init__(self):
        self.symbol = os.getenv('TRADING_SYMBOL')
        self.fee_calc = FeeCalculator()
        
        # RANGE CONFIG - Adjusted for realistic 3m testing
        self.range_config = {
            'min_position_usdt': 1000,    # Reduced for testing
            'max_position_usdt': 3000,    # Reasonable max
            'target_position_usdt': 2000, # Standard position
            'risk_percentage': 0.02,      # 2% risk (realistic)
            'reward_ratio': 2.0,          # 2R needed for profitability
            'max_position_time': 300,     # 5 minutes
            'emergency_stop_pct': 0.03,   # 3% emergency (realistic for crypto)
            'leverage': 3,                # Reduced from 10x
        }
        
        # TREND CONFIG - Adjusted for realistic 3m testing
        self.trend_config = {
            'min_position_usdt': 1500,
            'max_position_usdt': 3500,
            'target_position_usdt': 2500,
            'risk_percentage': 0.02,      # 2% risk
            'reward_ratio': 2.5,          # 2.5R for trends
            'max_position_time': 600,     # 10 minutes
            'emergency_stop_pct': 0.03,   # 3% emergency
            'trailing_stop_pct': 0.01,    # 1% trailing (realistic)
            'profit_lock_threshold': 1.5, # Lock at 1.5R
            'leverage': 3,
        }
        
        self.active_config = self.range_config.copy()
        self.active_strategy = "RANGE"
    
    def set_strategy(self, strategy_type: str):
        """Set active strategy configuration"""
        if strategy_type == "TREND":
            self.active_strategy = "TREND"
            self.active_config = self.trend_config.copy()
        else:
            self.active_strategy = "RANGE"
            self.active_config = self.range_config.copy()
    
    def calculate_position_size(self, balance: float, entry_price: float, stop_price: float) -> float:
        """Calculate position size with proper risk management"""
        if balance <= 0 or entry_price <= 0 or stop_price <= 0:
            return 0
        
        # Risk-based sizing
        risk_amount = balance * self.active_config['risk_percentage']
        stop_distance = abs(entry_price - stop_price) / entry_price
        
        if stop_distance < 0.005:  # Minimum 0.5% stop distance
            return 0
        
        # Calculate position size based on risk
        position_value = risk_amount / stop_distance
        
        # Apply bounds
        position_value = np.clip(
            position_value,
            self.active_config['min_position_usdt'],
            min(self.active_config['max_position_usdt'], balance * 0.1)  # Max 10% of balance
        )
        
        # Validate profitability
        expected_profit = position_value * stop_distance * self.active_config['reward_ratio']
        if not self.fee_calc.validate_profitability(position_value, expected_profit):
            return 0  # Skip unprofitable trades
        
        return position_value / entry_price
    
    def should_close_position(self, current_price: float, entry_price: float, side: str,
                             unrealized_pnl: float, position_age_seconds: float,
                             position_size_usdt: float = 0) -> Tuple[bool, str]:
        """Simplified exit logic with correct fee consideration"""
        if entry_price <= 0:
            return False, "hold"
        
        # Get real costs
        breakeven_pnl = self.fee_calc.get_breakeven_pnl(position_size_usdt)
        min_target = self.fee_calc.get_minimum_profit_target(position_size_usdt)
        
        # Emergency stop (account for fees)
        emergency_loss = -(position_size_usdt * self.active_config['emergency_stop_pct'] + breakeven_pnl)
        if unrealized_pnl <= emergency_loss:
            return True, "emergency_stop"
        
        # Max time exit
        if position_age_seconds >= self.active_config['max_position_time']:
            return True, "max_hold_time"
        
        # Profit target hit
        if unrealized_pnl >= min_target:
            return True, "profit_target"
        
        # Strategy-specific exits
        if self.active_strategy == "TREND" and unrealized_pnl > breakeven_pnl:
            # Simple trailing stop
            price_change = abs(current_price - entry_price) / entry_price
            if price_change > self.trend_config['trailing_stop_pct']:
                profit_ratio = unrealized_pnl / (position_size_usdt * self.active_config['risk_percentage'])
                if profit_ratio >= self.trend_config['profit_lock_threshold']:
                    # Check if price reversed
                    if side.lower() == 'buy':
                        if current_price < entry_price * (1 + price_change - self.trend_config['trailing_stop_pct']):
                            return True, "trailing_stop"
                    else:
                        if current_price > entry_price * (1 - price_change + self.trend_config['trailing_stop_pct']):
                            return True, "trailing_stop"
        
        # Time-based reduced profit taking
        if position_age_seconds > self.active_config['max_position_time'] * 0.7:
            if unrealized_pnl > breakeven_pnl * 1.5:  # At least 50% above breakeven
                return True, "timeout_profit"
        
        return False, "hold"
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary with correct calculations"""
        position_size = self.active_config['target_position_usdt']
        breakeven = self.fee_calc.get_breakeven_pnl(position_size)
        min_target = self.fee_calc.get_minimum_profit_target(position_size)
        required_move = self.fee_calc.get_required_price_move(position_size, min_target)
        
        # Calculate correct breakeven win rates
        if self.active_strategy == "RANGE":
            breakeven_wr = self.fee_calc.calculate_breakeven_rate(0.02, 2.0)
        else:
            breakeven_wr = self.fee_calc.calculate_breakeven_rate(0.02, 2.5)
        
        return {
            'strategy': self.active_strategy,
            'position_size': position_size,
            'breakeven_fees': round(breakeven, 2),
            'minimum_target': round(min_target, 2),
            'required_move_pct': round(required_move, 2),
            'breakeven_winrate': round(breakeven_wr * 100, 1),
            'risk_reward': f"1:{self.active_config['reward_ratio']}",
            'max_hold_time': f"{self.active_config['max_position_time']}s",
            'emergency_stop': f"{self.active_config['emergency_stop_pct']*100}%",
            'fee_model': {
                'round_trip': f"{float(self.fee_calc.round_trip_cost)*100:.1f}%",
                'per_side': f"{float(self.fee_calc.total_cost_rate)*100:.1f}%"
            }
        }