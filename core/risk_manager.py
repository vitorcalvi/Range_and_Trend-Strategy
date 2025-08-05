import os
from dotenv import load_dotenv
from typing import Tuple, Dict, Any

load_dotenv()

class RiskManager:
    """FIXED: Risk Manager with correct $2,381 position sizing"""

    def __init__(self):
        self.symbol = os.getenv('TRADING_SYMBOL', 'ETHUSDT')
        self.fee_rate = 0.0011  # 0.11% round-trip fees
        
        # FIXED: Enforced $2,381 position sizing
        self.range_config = {
            'fixed_position_usdt': 2381,
            'gross_profit_target': 65,
            'max_position_time': 900,      # 15 minutes
            'emergency_stop_pct': 0.02,    # 2% stop loss
            'leverage': 10
        }
        
        self.trend_config = {
            'fixed_position_usdt': 2381,   # FIXED: Same as range
            'risk_reward_ratio': 3.0,
            'max_position_time': 7200,     # FIXED: 2 hours (was 4)
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
    
    def calculate_position_size(self, balance: float, entry_price: float, stop_price: float) -> float:
        """FIXED: Calculate position size - enforce $2,381 target"""
        if balance <= 0 or entry_price <= 0 or stop_price <= 0:
            return 0
        
        # FIXED: Always use exactly $2,381 regardless of strategy
        target_usdt = 2381
        
        # Safety check - don't use more than 40% of balance
        max_usdt = min(target_usdt, balance * 0.4)
        
        # FIXED: For both strategies, use fixed sizing
        if max_usdt < 1000:  # If balance too small for $2,381
            return 0
        
        return max_usdt / entry_price
    
    def should_close_position(self, current_price: float, entry_price: float, side: str, 
                             unrealized_pnl: float, position_age_seconds: float, 
                             position_size_usdt: float = 0) -> Tuple[bool, str]:
        """FIXED: Better exit conditions for trend strategy"""
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
            return self._check_trend_exits(entry_price, unrealized_pnl, position_age_seconds, fees)
    
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
    
    def _check_trend_exits(self, entry_price: float, unrealized_pnl: float, 
                          position_age_seconds: float, fees: float) -> Tuple[bool, str]:
        """FIXED: Improved trend strategy exit conditions"""
        
        # FIXED: Take smaller profits earlier to avoid max hold time
        quick_profit_threshold = fees + 25  # $25 net profit
        if unrealized_pnl >= quick_profit_threshold and position_age_seconds >= 1800:  # After 30 minutes
            return True, "profit_target"
        
        # Original profit target (3:1 risk/reward)
        if unrealized_pnl > 0:
            risk_unit = entry_price * 0.02  # 2% risk
            profit_ratio = unrealized_pnl / risk_unit
            
            if profit_ratio >= self.trend_config['profit_lock_threshold']:
                target_profit = risk_unit * self.trend_config['risk_reward_ratio']  # 3x risk
                if unrealized_pnl >= target_profit:
                    return True, "profit_target"
        
        # FIXED: Exit with small loss before max hold time if trend fails
        if position_age_seconds >= self.active_config['max_position_time'] * 0.8:  # 80% of max time
            small_loss_threshold = -(fees + 15)  # Allow $15 loss beyond fees
            if unrealized_pnl <= small_loss_threshold:
                return True, "timeout_no_profit"
        
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
        """FIXED: Keep position size at $2,381 regardless of conditions"""
        base_config = (self.range_config if self.active_strategy == "RANGE" 
                      else self.trend_config).copy()
        
        # FIXED: No position size adjustments - always use $2,381
        self.active_config = base_config.copy()
        
        # Only adjust timeouts slightly for volatility
        if volatility == "HIGH_VOL":
            self.active_config['max_position_time'] = int(base_config['max_position_time'] * 0.8)
        elif volatility == "LOW_VOL":
            self.active_config['max_position_time'] = int(base_config['max_position_time'] * 1.2)
    
    def get_leverage(self) -> int:
        return self.active_config['leverage']
    
    def get_max_position_time(self) -> int:
        return self.active_config['max_position_time']
    
    def get_active_config(self) -> Dict[str, Any]:
        """Get complete active configuration"""
        position_size = 2381  # FIXED: Always $2,381
        fees = self.get_break_even_pnl(position_size)
        
        if self.active_strategy == "RANGE":
            gross_target = self.active_config['gross_profit_target']
            net_target = gross_target - fees
            sizing_info = {
                'method': 'Fixed Size (CORRECTED)',
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
                'method': 'Fixed Size (CORRECTED)',
                'size_usdt': position_size,
                'risk_amount': f"${risk_amount:.2f}",
                'target_profit': f"${target_profit:.2f}",
                'estimated_fees': f"${fees:.2f}",
                'risk_reward': f"1:{self.trend_config['risk_reward_ratio']}",
                'hold_time': f"{self.active_config['max_position_time']}s (REDUCED)"
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
                'position_size': f"$2,381 (FIXED)",
                'optimization': 'Enforced $2,381 sizing + 10min trade cooldowns'
            }
        }