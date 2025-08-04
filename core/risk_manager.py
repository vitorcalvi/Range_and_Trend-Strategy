import os
from dotenv import load_dotenv
from typing import Tuple, Dict, Any
from decimal import Decimal

load_dotenv()

class RiskManager:
    """FIXED Risk Manager - Correct method signature and fee calculations"""

    def __init__(self):
        self.symbol = os.getenv('TRADING_SYMBOL')
        self.fee_rate = Decimal('0.0011')  # 0.11% round-trip fees
        
        # FIXED: Loosened emergency stops
        self.range_config = {
            'fixed_position_usdt': 9091,
            'gross_profit_target': 50,
            'max_position_time': 300,  # 5 minutes
            'emergency_stop_pct': 0.015,  # FIXED: 1.5%
            'leverage': 10
        }
        
        self.trend_config = {
            'fixed_position_usdt': 12000,
            'risk_reward_ratio': 2.5,
            'max_position_time': 3600,
            'emergency_stop_pct': 0.02,  # FIXED: 2%
            'trailing_stop_pct': 0.005,
            'profit_lock_threshold': 1.5,
            'leverage': 10
        }
        
        self.active_config = self.range_config.copy()
        self.active_strategy = "RANGE"
    
    def set_strategy(self, strategy_type: str):
        """Set active strategy configuration"""
        strategy_type = strategy_type if strategy_type in ["RANGE", "TREND"] else "RANGE"
        self.active_config = (self.range_config if strategy_type == "RANGE" 
                             else self.trend_config).copy()
        self.active_strategy = strategy_type
    
    def validate_trade(self, signal: Dict[str, Any], balance: float, current_price: float) -> Tuple[bool, str]:
        """FIXED: Loosened validation rules"""
        if not signal or not signal.get('action') or not signal.get('structure_stop'):
            return False, "Invalid signal"
        
        if balance <= 0 or current_price <= 0:
            return False, "Invalid market data"
        
        # FIXED: Loosened stop distance validation
        stop_distance = abs(current_price - signal['structure_stop']) / current_price
        min_stop, max_stop = (0.0003, 0.01) if self.active_strategy == "RANGE" else (0.002, 0.03)
        
        if not (min_stop <= stop_distance <= max_stop):
            return False, f"Invalid stop distance for {self.active_strategy.lower()} strategy"
        
        max_position = min(self.active_config['fixed_position_usdt'], balance * 0.8)
        if max_position < 100:
            return False, "Insufficient balance"
        
        return True, "Valid"
    
    def get_break_even_pnl(self, position_size_usdt: float) -> Decimal:
        """Calculate break-even PnL needed to cover fees"""
        # Fee model: B = P × 0.0011
        return Decimal(str(position_size_usdt)) * self.fee_rate
    
    def get_min_profitable_target(self, position_size_usdt: float) -> float:
        """Calculate minimum profitable target"""
        break_even_fees = float(self.get_break_even_pnl(position_size_usdt))
        min_net_profit = 20  # FIXED: Reduced from $30 to $20
        return min_net_profit + break_even_fees
    
    def calculate_position_size(self, balance: float, entry_price: float, stop_price: float) -> float:
        """Calculate position size based on active strategy"""
        if balance <= 0 or entry_price <= 0 or stop_price <= 0:
            return 0
        
        max_position_usdt = min(self.active_config['fixed_position_usdt'], balance * 0.5)
        
        if self.active_strategy == "RANGE":
            position_size = max_position_usdt / entry_price
        else:  # TREND
            risk_per_trade = balance * 0.02
            stop_distance = abs(entry_price - stop_price)
            
            if stop_distance > 0:
                position_size = risk_per_trade / stop_distance
                max_size = max_position_usdt / entry_price
                position_size = min(position_size, max_size)
            else:
                position_size = max_position_usdt / entry_price
        
        return round(max(position_size, 0), 6)
    
    def should_close_position(self, current_price: float, entry_price: float, side: str, 
                             unrealized_pnl: float, position_age_seconds: float, 
                             position_size_usdt: float = 0) -> Tuple[bool, str]:
        """FIXED: Corrected method signature with position_size_usdt parameter"""
        if entry_price <= 0:
            return False, "hold"
        
        # FIXED: Emergency stop based on percentage of position value
        if position_size_usdt > 0:
            emergency_loss = -(position_size_usdt * self.active_config['emergency_stop_pct'])
        else:
            # Fallback to entry price based calculation
            position_value = entry_price
            emergency_loss = -(position_value * self.active_config['emergency_stop_pct'])
        
        if unrealized_pnl <= emergency_loss:
            return True, "emergency_stop"
        
        # Max hold time
        if position_age_seconds >= self.active_config['max_position_time']:
            return True, "max_hold_time"
        
        # Strategy-specific exits
        return (self._check_range_exits(unrealized_pnl, position_age_seconds, position_size_usdt) 
                if self.active_strategy == "RANGE" 
                else self._check_trend_exits(current_price, entry_price, side, unrealized_pnl, position_age_seconds))
    
    def _check_range_exits(self, unrealized_pnl: float, position_age_seconds: float, 
                          position_size_usdt: float = 0) -> Tuple[bool, str]:
        """Range exits using gross profit target"""
        profit_target = self.active_config['gross_profit_target']
        
        if unrealized_pnl >= profit_target:
            return True, "profit_target"
        
        # FIXED: Less aggressive timeout exit
        timeout_threshold = self.active_config['max_position_time'] * 0.8
        if position_age_seconds >= timeout_threshold and unrealized_pnl <= -20:
            return True, "timeout_no_profit"
        
        return False, "hold"
    
    def _check_trend_exits(self, current_price: float, entry_price: float, side: str, 
                          unrealized_pnl: float, position_age_seconds: float) -> Tuple[bool, str]:
        """Trend exits using unrealized PnL directly"""
        if unrealized_pnl > 0:
            profit_ratio = unrealized_pnl / (entry_price * 0.02)
            
            if profit_ratio >= self.trend_config['profit_lock_threshold']:
                target_profit = entry_price * 0.02 * self.trend_config['risk_reward_ratio']
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
        """Adapt risk parameters based on market conditions"""
        base_config = (self.range_config if self.active_strategy == "RANGE" 
                      else self.trend_config).copy()
        
        vol_multipliers = {"HIGH_VOL": 0.7, "LOW_VOL": 1.2, "NORMAL": 1.0}
        volatility_multiplier = vol_multipliers.get(volatility, 1.0)
        
        self.active_config = base_config.copy()
        original_size = self.active_config['fixed_position_usdt']
        self.active_config['fixed_position_usdt'] = int(original_size * volatility_multiplier)
    
    def get_leverage(self) -> int:
        return self.active_config['leverage']
    
    def get_max_position_time(self) -> int:
        return self.active_config['max_position_time']
    
    def get_active_config(self) -> Dict[str, Any]:
        return {
            'strategy': self.active_strategy,
            'config': self.active_config.copy(),
            'fee_rate': float(self.fee_rate),
            'position_sizing': self._get_position_sizing_info(),
            'risk_limits': self._get_risk_limits()
        }
    
    def _get_position_sizing_info(self) -> Dict[str, Any]:
        if self.active_strategy == "RANGE":
            gross_target = self.range_config['gross_profit_target']
            estimated_net = gross_target - (self.range_config['fixed_position_usdt'] * float(self.fee_rate))
            return {
                'method': 'Fixed Size',
                'size_usdt': self.active_config['fixed_position_usdt'],
                'gross_target': f"${gross_target} USDT",
                'estimated_net': f"${estimated_net:.2f} USDT",
                'hold_time': f"{self.active_config['max_position_time']}s"
            }
        else:
            return {
                'method': 'Risk-Based',
                'max_size_usdt': self.active_config['fixed_position_usdt'],
                'risk_reward': f"1:{self.trend_config['risk_reward_ratio']}",
                'hold_time': f"{self.active_config['max_position_time']}s"
            }
    
    def _get_risk_limits(self) -> Dict[str, Any]:
        return {
            'emergency_stop': f"{self.active_config['emergency_stop_pct']*100:.1f}%",
            'max_hold_time': f"{self.active_config['max_position_time']}s",
            'leverage': f"{self.active_config['leverage']}x",
            'fee_rate': f"{float(self.fee_rate)*100:.2f}%",
            'trailing_stop': (f"{self.trend_config.get('trailing_stop_pct', 0)*100:.1f}%" 
                            if self.active_strategy == "TREND" else "None")
        }