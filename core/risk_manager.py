import os
from dotenv import load_dotenv
from typing import Tuple, Dict, Any

load_dotenv()

class RiskManager:
    """Dual Strategy Risk Manager with FIXED fee calculations"""

    def __init__(self):
        self.symbol = os.getenv('TRADING_SYMBOL')
        self.fee_rate = 0.0011  # 0.11% round-trip fee (0.055% taker x 2)
        
        # Strategy configurations - FIXED profit targets
        self.range_config = {
            'fixed_position_usdt': 9091,
            'base_profit_usdt': 35,  # FIXED: Increased from 15 to 35
            'max_position_time': 180,
            'emergency_stop_pct': 0.006,
            'leverage': 10
        }
        
        self.trend_config = {
            'fixed_position_usdt': 12000,
            'risk_reward_ratio': 2.5,
            'max_position_time': 3600,
            'emergency_stop_pct': 0.01,
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
    
    def calculate_fee_adjusted_profit_target(self, position_size_usdt: float, base_profit: float = None) -> float:
        """Calculate profit target including fees"""
        if base_profit is None:
            base_profit = self.active_config.get('base_profit_usdt', 35)  # FIXED: default to 35
        
        fee_cost = position_size_usdt * self.fee_rate
        return base_profit + fee_cost
    
    def calculate_net_pnl(self, gross_pnl: float, position_size_usdt: float) -> float:
        """FIXED: Calculate net PnL after fees"""
        fee_cost = position_size_usdt * self.fee_rate
        return gross_pnl - fee_cost
    
    def validate_trade_profitability(self, position_size_usdt: float, expected_profit: float) -> bool:
        """FIXED: Validate trade can be profitable after fees"""
        fee_cost = position_size_usdt * self.fee_rate
        min_required_profit = self.active_config.get('base_profit_usdt', 35)
        gross_profit_needed = min_required_profit + fee_cost
        
        return expected_profit >= gross_profit_needed
    
    def validate_trade(self, signal: Dict[str, Any], balance: float, current_price: float) -> Tuple[bool, str]:
        """Validate trade with strategy-specific rules"""
        if not signal or not signal.get('action') or not signal.get('structure_stop'):
            return False, "Invalid signal"
        
        if balance <= 0 or current_price <= 0:
            return False, "Invalid market data"
        
        # Stop distance validation
        stop_distance = abs(current_price - signal['structure_stop']) / current_price
        
        min_stop, max_stop = (0.0005, 0.005) if self.active_strategy == "RANGE" else (0.003, 0.02)
        
        if not (min_stop <= stop_distance <= max_stop):
            return False, f"Invalid stop distance for {self.active_strategy.lower()} strategy"
        
        # Position size validation
        max_position = min(self.active_config['fixed_position_usdt'], balance * 0.8)
        if max_position < 100:
            return False, "Insufficient balance"
        
        # FIXED: Profitability validation
        expected_profit = max_position * stop_distance * 2  # Assume 2:1 RR minimum
        if not self.validate_trade_profitability(max_position, expected_profit):
            return False, f"Trade not profitable after fees (need ${self.calculate_fee_adjusted_profit_target(max_position):.2f} gross)"
        
        return True, "Valid"
    
    def calculate_position_size(self, balance: float, entry_price: float, stop_price: float) -> float:
        """Calculate position size based on active strategy"""
        if balance <= 0 or entry_price <= 0 or stop_price <= 0:
            return 0
        
        max_position_usdt = min(self.active_config['fixed_position_usdt'], balance * 0.5)
        
        if self.active_strategy == "RANGE":
            position_size = max_position_usdt / entry_price
        else:  # TREND
            risk_per_trade = balance * 0.02  # 2% risk per trade
            stop_distance = abs(entry_price - stop_price)
            
            if stop_distance > 0:
                position_size = risk_per_trade / stop_distance
                max_size = max_position_usdt / entry_price
                position_size = min(position_size, max_size)
            else:
                position_size = max_position_usdt / entry_price
        
        return round(max(position_size, 0), 6)
    
    def should_close_position(self, current_price: float, entry_price: float, side: str, 
                             unrealized_pnl: float, position_age_seconds: float, position_size_usdt: float = None) -> Tuple[bool, str]:
        """FIXED: Determine if position should be closed"""
        if entry_price <= 0:
            return False, "hold"
        
        # Calculate net PnL if position size provided
        net_pnl = unrealized_pnl
        if position_size_usdt:
            net_pnl = self.calculate_net_pnl(unrealized_pnl, position_size_usdt)
        
        # Emergency stop based on gross PnL percentage
        pnl_pct = unrealized_pnl / entry_price if entry_price > 0 else 0
        if pnl_pct <= -self.active_config['emergency_stop_pct']:
            return True, "emergency_stop"
        
        # Max hold time - FIXED: Use configured time, not hardcoded
        if position_age_seconds >= self.active_config['max_position_time']:
            return True, "max_hold_time"
        
        # Strategy-specific exits
        return (self._check_range_exits(net_pnl, position_age_seconds, position_size_usdt) 
                if self.active_strategy == "RANGE" 
                else self._check_trend_exits(current_price, entry_price, side, net_pnl, position_age_seconds))
    
    def _check_range_exits(self, net_pnl: float, position_age_seconds: float, position_size_usdt: float = None) -> Tuple[bool, str]:
        """FIXED: Range strategy specific exits"""
        # Use provided position size or default
        position_size = position_size_usdt or self.active_config['fixed_position_usdt']
        profit_target = self.active_config['base_profit_usdt']  # Net profit target
        
        # Check if we hit our net profit target
        if net_pnl >= profit_target:
            return True, "profit_target"
        
        # FIXED: Use configured timeout instead of hardcoded 120s
        # Only exit early if we're well past halfway point and losing money
        timeout_threshold = self.active_config['max_position_time'] * 0.75  # 75% of max time
        if position_age_seconds >= timeout_threshold and net_pnl <= -5:  # Losing more than $5
            return True, "timeout_no_profit"
        
        return False, "hold"
    
    def _check_trend_exits(self, current_price: float, entry_price: float, side: str, 
                          net_pnl: float, position_age_seconds: float) -> Tuple[bool, str]:
        """FIXED: Trend strategy specific exits"""
        if net_pnl > 0:
            profit_ratio = net_pnl / (entry_price * 0.02)
            
            if profit_ratio >= self.trend_config['profit_lock_threshold']:
                # Calculate fee-adjusted target
                position_size = self.active_config['fixed_position_usdt']
                fee_cost = position_size * self.fee_rate
                target_profit = entry_price * 0.02 * self.trend_config['risk_reward_ratio']
                
                if net_pnl >= target_profit:
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
        
        # Volatility adjustment
        vol_multipliers = {"HIGH_VOL": 0.7, "LOW_VOL": 1.2, "NORMAL": 1.0}
        volatility_multiplier = vol_multipliers.get(volatility, 1.0)
        
        # Apply adjustment
        self.active_config = base_config.copy()
        original_size = self.active_config['fixed_position_usdt']
        self.active_config['fixed_position_usdt'] = int(original_size * volatility_multiplier)
    
    def get_leverage(self) -> int:
        """Get leverage setting"""
        return self.active_config['leverage']
    
    def get_max_position_time(self) -> int:
        """Get maximum position time"""
        return self.active_config['max_position_time']
    
    def get_active_config(self) -> Dict[str, Any]:
        """Get current strategy configuration"""
        return {
            'strategy': self.active_strategy,
            'config': self.active_config.copy(),
            'fee_rate': self.fee_rate,
            'position_sizing': self._get_position_sizing_info(),
            'risk_limits': self._get_risk_limits()
        }
    
    def _get_position_sizing_info(self) -> Dict[str, Any]:
        """Get position sizing information"""
        if self.active_strategy == "RANGE":
            profit_target = self.calculate_fee_adjusted_profit_target(self.range_config['fixed_position_usdt'])
            net_profit = self.range_config['base_profit_usdt']
            return {
                'method': 'Fixed Size',
                'size_usdt': self.active_config['fixed_position_usdt'],
                'gross_target': f"${profit_target:.2f} USDT",
                'net_profit': f"${net_profit:.2f} USDT",
                'hold_time': f"{self.active_config['max_position_time']}s"
            }
        else:
            return {
                'method': 'Risk-Based',
                'max_size_usdt': self.active_config['fixed_position_usdt'],
                'risk_reward': f"1:{self.trend_config['risk_reward_ratio']} (fee-adjusted)",
                'hold_time': f"{self.active_config['max_position_time']}s"
            }
    
    def _get_risk_limits(self) -> Dict[str, Any]:
        """Get risk limit information"""
        return {
            'emergency_stop': f"{self.active_config['emergency_stop_pct']*100:.1f}%",
            'max_hold_time': f"{self.active_config['max_position_time']}s",
            'leverage': f"{self.active_config['leverage']}x",
            'fee_rate': f"{self.fee_rate*100:.2f}%",
            'trailing_stop': (f"{self.trend_config.get('trailing_stop_pct', 0)*100:.1f}%" 
                            if self.active_strategy == "TREND" else "None")
        }