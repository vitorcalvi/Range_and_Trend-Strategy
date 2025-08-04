import os
from dotenv import load_dotenv
from typing import Tuple, Dict, Any

load_dotenv()

class RiskManager:
    """Streamlined Risk Manager - $2,381 Position Sizing (88% Lower Exposure)"""

    def __init__(self):
        self.symbol = os.getenv('TRADING_SYMBOL', 'ETHUSDT')
        self.fee_rate = 0.0011  # 0.11% round-trip fees
        
        # Optimized configurations - both use $2,381 positions
        self.range_config = {
            'fixed_position_usdt': 2381,
            'gross_profit_target': 65,
            'max_position_time': 900,      # 15 minutes
            'emergency_stop_pct': 0.02,    # 2% stop loss
            'leverage': 10
        }
        
        self.trend_config = {
            'fixed_position_usdt': 2381,
            'risk_reward_ratio': 3.0,
            'max_position_time': 14400,    # 4 hours
            'emergency_stop_pct': 0.02,
            'trailing_stop_pct': 0.005,
            'profit_lock_threshold': 1.5,
            'leverage': 10
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
        """Calculate break-even PnL to cover fees
        
        Fee model: 0.11% round-trip Bybit market orders
        Break-even PnL (B) = position_size × 0.0011
        """
        return position_size_usdt * self.fee_rate
    
    def get_min_profitable_target(self, position_size_usdt: float) -> float:
        """Calculate minimum profitable target"""
        return self.get_break_even_pnl(position_size_usdt) + 15.0  # $15 minimum net profit
    
    def validate_trade(self, signal: Dict[str, Any], balance: float, current_price: float) -> Tuple[bool, str]:
        """Validate trade parameters"""
        if not signal or not signal.get('action') or not signal.get('structure_stop'):
            return False, "Invalid signal"
        
        if balance <= 0 or current_price <= 0:
            return False, "Invalid market data"
        
        # Check stop distance is reasonable
        stop_distance = abs(current_price - signal['structure_stop']) / current_price
        min_stop, max_stop = (0.0005, 0.015) if self.active_strategy == "RANGE" else (0.003, 0.025)
        
        if not (min_stop <= stop_distance <= max_stop):
            return False, f"Invalid stop distance for {self.active_strategy.lower()} strategy"
        
        # Check sufficient balance for $2,381 strategy
        if min(self.active_config['fixed_position_usdt'], balance * 0.5) < 500:
            return False, "Insufficient balance for optimized position sizing"
        
        return True, "Valid"
    
    def calculate_position_size(self, balance: float, entry_price: float, stop_price: float) -> float:
        """Calculate position size - optimized for $2,381 target"""
        if balance <= 0 or entry_price <= 0 or stop_price <= 0:
            return 0
        
        target_usdt = self.active_config['fixed_position_usdt']
        max_usdt = min(target_usdt, balance * 0.4)
        
        if self.active_strategy == "RANGE":
            # Fixed sizing for range
            return max_usdt / entry_price
        
        # Risk-based sizing for trend with $2,381 cap
        stop_distance = abs(entry_price - stop_price)
        if stop_distance == 0:
            return max_usdt / entry_price
        
        risk_amount = balance * 0.015  # 1.5% risk
        risk_based_qty = risk_amount / stop_distance
        max_qty = max_usdt / entry_price
        
        return round(min(risk_based_qty, max_qty), 6)
    
    def should_close_position(self, current_price: float, entry_price: float, side: str, 
                             unrealized_pnl: float, position_age_seconds: float, 
                             position_size_usdt: float = 0) -> Tuple[bool, str]:
        """Check if position should be closed"""
        if entry_price <= 0:
            return False, "hold"
        
        # Emergency stop: loss exceeds 2% + fees
        fees = self.get_break_even_pnl(position_size_usdt) if position_size_usdt > 0 else 0
        emergency_threshold = -(position_size_usdt * self.active_config['emergency_stop_pct'] + fees)
        
        if unrealized_pnl <= emergency_threshold:
            return True, "emergency_stop"
        
        # Max hold time exceeded
        if position_age_seconds >= self.active_config['max_position_time']:
            return True, "max_hold_time"
        
        # Strategy-specific exits
        if self.active_strategy == "RANGE":
            return self._check_range_exits(unrealized_pnl, position_age_seconds, fees)
        else:
            return self._check_trend_exits(entry_price, unrealized_pnl)
    
    def _check_range_exits(self, unrealized_pnl: float, position_age_seconds: float, fees: float) -> Tuple[bool, str]:
        """Range strategy exit conditions"""
        # Hit profit target
        if unrealized_pnl >= self.active_config['gross_profit_target']:
            return True, "profit_target"
        
        # Timeout with significant loss
        timeout_threshold = self.active_config['max_position_time'] * 0.7
        loss_threshold = -(fees + 8)  # Allow $8 loss beyond fees
        
        if position_age_seconds >= timeout_threshold and unrealized_pnl <= loss_threshold:
            return True, "timeout_no_profit"
        
        return False, "hold"
    
    def _check_trend_exits(self, entry_price: float, unrealized_pnl: float) -> Tuple[bool, str]:
        """Trend strategy exit conditions"""
        if unrealized_pnl <= 0:
            return False, "hold"
        
        # Calculate profit in terms of risk units (2% of entry = 1 risk unit)
        risk_unit = entry_price * 0.02
        profit_ratio = unrealized_pnl / risk_unit
        
        # Lock profits and check for 3:1 target
        if profit_ratio >= self.trend_config['profit_lock_threshold']:
            target_profit = risk_unit * self.trend_config['risk_reward_ratio']  # 3x risk
            if unrealized_pnl >= target_profit:
                return True, "profit_target"
        
        return False, "hold"
    
    def calculate_trailing_stop(self, current_price: float, entry_price: float, side: str, 
                               highest_profit: float) -> float:
        """Calculate trailing stop for trend strategy"""
        if self.active_strategy != "TREND":
            return 0
        
        trail_distance = current_price * self.trend_config['trailing_stop_pct']
        return (current_price - trail_distance if side.lower() == 'buy' 
                else current_price + trail_distance)
    
    def adapt_to_market_condition(self, market_condition: str, volatility: str):
        """Adapt position size based on market volatility"""
        base_config = (self.range_config if self.active_strategy == "RANGE" 
                      else self.trend_config).copy()
        
        # Conservative volatility adjustments
        vol_multipliers = {"HIGH_VOL": 0.8, "LOW_VOL": 1.1, "NORMAL": 1.0}
        multiplier = vol_multipliers.get(volatility, 1.0)
        
        self.active_config = base_config.copy()
        self.active_config['fixed_position_usdt'] = int(base_config['fixed_position_usdt'] * multiplier)
    
    def get_leverage(self) -> int:
        return self.active_config['leverage']
    
    def get_max_position_time(self) -> int:
        return self.active_config['max_position_time']
    
    def get_active_config(self) -> Dict[str, Any]:
        """Get complete active configuration"""
        position_size = self.active_config['fixed_position_usdt']
        fees = self.get_break_even_pnl(position_size)
        
        if self.active_strategy == "RANGE":
            gross_target = self.active_config['gross_profit_target']
            net_target = gross_target - fees
            sizing_info = {
                'method': 'Optimized Fixed Size',
                'size_usdt': position_size,
                'gross_target': f"${gross_target}",
                'estimated_fees': f"${fees:.2f}",
                'estimated_net': f"${net_target:.2f}",
                'hold_time': f"{self.active_config['max_position_time']}s"
            }
        else:
            risk_amount = position_size * 0.02
            target_profit = risk_amount * self.trend_config['risk_reward_ratio']
            sizing_info = {
                'method': 'Optimized Risk-Based',
                'max_size_usdt': position_size,
                'risk_amount': f"${risk_amount:.2f}",
                'target_profit': f"${target_profit:.2f}",
                'estimated_fees': f"${fees:.2f}",
                'risk_reward': f"1:{self.trend_config['risk_reward_ratio']}",
                'hold_time': f"{self.active_config['max_position_time']}s"
            }
        
        return {
            'strategy': self.active_strategy,
            'config': self.active_config.copy(),
            'fee_rate': self.fee_rate,
            'position_sizing': sizing_info,
            'risk_limits': {
                'emergency_stop': f"{self.active_config['emergency_stop_pct']*100:.1f}%",
                'max_hold_time': f"{self.active_config['max_position_time']}s",
                'leverage': f"{self.active_config['leverage']}x",
                'fee_rate': f"{self.fee_rate*100:.2f}%",
                'position_size': f"${self.active_config['fixed_position_usdt']}",
                'optimization': '88% lower exposure vs original'
            }
        }