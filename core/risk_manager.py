import os
from dotenv import load_dotenv
from typing import Tuple, Dict, Any

load_dotenv()

class RiskManager:
    """3-Minute Trading Risk Manager - $3,817 position sizing"""

    def __init__(self):
        self.symbol = os.getenv('TRADING_SYMBOL', 'ETHUSDT')
        self.fee_rate = 0.0011  # 0.11% round-trip fees
        
        # 3-Minute Trading: $3,817 position sizing
        self.range_config = {
            'fixed_position_usdt': 3817,    # 3m timeframe sizing
            'gross_profit_target': 92,      # 2.4% target (3817 * 0.024)
            'max_position_time': 2700,      # 45 minutes max hold
            'emergency_stop_pct': 0.012,    # 1.2% stop loss
            'leverage': 10
        }
        
        self.trend_config = {
            'fixed_position_usdt': 3817,    # Same as range for 3m
            'risk_reward_ratio': 2.5,       # 1:2.5 for 3m
            'max_position_time': 2700,      # 45 minutes max
            'emergency_stop_pct': 0.012,    # 1.2% stop loss
            'trailing_stop_pct': 0.008,     # 0.8% trailing
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
        """Calculate minimum profitable target for 3m trading"""
        return self.get_break_even_pnl(position_size_usdt) + 25.0  # $25 minimum net profit
    
    def calculate_position_size(self, balance: float, entry_price: float, stop_price: float) -> float:
        """3-Minute Trading: Calculate position size - enforce $3,817 target"""
        if balance <= 0 or entry_price <= 0 or stop_price <= 0:
            return 0
        
        # 3-minute trading: $3,817 position size
        target_usdt = 3817
        
        # Safety check - don't use more than 50% of balance for 3m trading
        max_usdt = min(target_usdt, balance * 0.5)
        
        if max_usdt < 2000:  # Minimum for 3m trading
            return 0
        
        return max_usdt / entry_price
    
    def should_close_position(self, current_price: float, entry_price: float, side: str, 
                             unrealized_pnl: float, position_age_seconds: float, 
                             position_size_usdt: float = 0) -> Tuple[bool, str]:
        """3-Minute Trading: Faster exit conditions"""
        if entry_price <= 0:
            return False, "hold"
        
        # Emergency stop: 1.2% + fees for 3m trading
        fees = self.get_break_even_pnl(position_size_usdt) if position_size_usdt > 0 else 0
        emergency_threshold = -(position_size_usdt * self.active_config['emergency_stop_pct'] + fees)
        
        if unrealized_pnl <= emergency_threshold:
            return True, "emergency_stop"
        
        # Max hold time: 45 minutes for 3m trading
        if position_age_seconds >= self.active_config['max_position_time']:
            return True, "max_hold_time"
        
        # Strategy-specific exits
        if self.active_strategy == "RANGE":
            return self._check_range_exits_3m(unrealized_pnl, position_age_seconds, fees)
        else:
            return self._check_trend_exits_3m(entry_price, unrealized_pnl, position_age_seconds, fees)
    
    def _check_range_exits_3m(self, unrealized_pnl: float, position_age_seconds: float, fees: float) -> Tuple[bool, str]:
        """3-Minute range strategy exit conditions"""
        # Quick profit target: $92 gross (2.4% of $3,817)
        if unrealized_pnl >= self.active_config['gross_profit_target']:
            return True, "profit_target"
        
        # Fast timeout for 3m: exit if losing after 30 minutes
        timeout_threshold = 1800  # 30 minutes
        loss_threshold = -(fees + 15)  # Allow $15 loss beyond fees
        
        if position_age_seconds >= timeout_threshold and unrealized_pnl <= loss_threshold:
            return True, "timeout_no_profit"
        
        return False, "hold"
    
    def _check_trend_exits_3m(self, entry_price: float, unrealized_pnl: float, 
                             position_age_seconds: float, fees: float) -> Tuple[bool, str]:
        """3-Minute trend strategy exit conditions"""
        
        # Quick profit for 3m: Take $50+ net profit after 15 minutes
        quick_profit_threshold = fees + 50
        if unrealized_pnl >= quick_profit_threshold and position_age_seconds >= 900:  # 15 minutes
            return True, "profit_target"
        
        # 2.5:1 risk/reward for 3m trading
        if unrealized_pnl > 0:
            risk_unit = entry_price * 0.012  # 1.2% risk
            target_profit = risk_unit * self.trend_config['risk_reward_ratio']  # 2.5x
            if unrealized_pnl >= target_profit:
                return True, "profit_target"
        
        # Fast exit for failed 3m trends: 70% of max time
        if position_age_seconds >= self.active_config['max_position_time'] * 0.7:  # ~30 minutes
            small_loss_threshold = -(fees + 20)  # Allow $20 loss beyond fees
            if unrealized_pnl <= small_loss_threshold:
                return True, "timeout_no_profit"
        
        return False, "hold"
    
    def calculate_trailing_stop(self, current_price: float, entry_price: float, side: str, 
                               highest_profit: float) -> float:
        """Calculate trailing stop for 3m trend strategy"""
        if self.active_strategy != "TREND":
            return 0
        
        trail_distance = current_price * self.trend_config['trailing_stop_pct']  # 0.8%
        return (current_price - trail_distance if side.lower() == 'buy' 
                else current_price + trail_distance)
    
    def adapt_to_market_condition(self, market_condition: str, volatility: str):
        """3-Minute Trading: Keep position size fixed at $3,817"""
        base_config = (self.range_config if self.active_strategy == "RANGE" 
                      else self.trend_config).copy()
        
        # Keep $3,817 position size for 3m trading
        self.active_config = base_config.copy()
        
        # Adjust timeouts for volatility in 3m trading
        if volatility == "HIGH_VOL":
            self.active_config['max_position_time'] = int(base_config['max_position_time'] * 0.7)  # 30 min
        elif volatility == "LOW_VOL":
            self.active_config['max_position_time'] = int(base_config['max_position_time'] * 1.3)  # 60 min
    
    def get_leverage(self) -> int:
        return self.active_config['leverage']
    
    def get_max_position_time(self) -> int:
        return self.active_config['max_position_time']
    
    def get_active_config(self) -> Dict[str, Any]:
        """Get 3-minute trading configuration"""
        position_size = 3817  # 3-minute trading size
        fees = self.get_break_even_pnl(position_size)
        
        if self.active_strategy == "RANGE":
            gross_target = self.active_config['gross_profit_target']
            net_target = gross_target - fees
            sizing_info = {
                'method': '3-Minute Fixed Size',
                'size_usdt': position_size,
                'gross_target': f"${gross_target}",
                'estimated_fees': f"${fees:.2f}",
                'estimated_net': f"${net_target:.2f}",
                'hold_time': f"{self.active_config['max_position_time']}s (45min max)"
            }
        else:
            risk_amount = position_size * 0.012  # 1.2% risk
            target_profit = risk_amount * self.trend_config['risk_reward_ratio']
            sizing_info = {
                'method': '3-Minute Fixed Size',
                'size_usdt': position_size,
                'risk_amount': f"${risk_amount:.2f}",
                'target_profit': f"${target_profit:.2f}",
                'estimated_fees': f"${fees:.2f}",
                'risk_reward': f"1:{self.trend_config['risk_reward_ratio']}",
                'hold_time': f"{self.active_config['max_position_time']}s (45min max)"
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
                'position_size': f"$3,817 (3-MINUTE)",
                'optimization': '3-minute timeframe + fast exits'
            }
        }