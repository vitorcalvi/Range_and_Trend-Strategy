import os
from dotenv import load_dotenv
from typing import Tuple, Dict, Any

load_dotenv()

class RiskManager:
    """Dual Strategy Risk Manager"""

    def __init__(self):
        self.symbol = os.getenv('TRADING_SYMBOL')
        
        # Strategy configurations
        self.range_config = {
            'fixed_position_usdt': 9091,
            'profit_threshold_usdt': 15,
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
        
        self.active_config = self.range_config
        self.active_strategy = "RANGE"
    
    def set_strategy(self, strategy_type: str):
        """Set active strategy configuration"""
        if strategy_type not in ["RANGE", "TREND"]:
            strategy_type = "RANGE"
            
        self.active_config = (self.range_config if strategy_type == "RANGE" 
                             else self.trend_config).copy()
        self.active_strategy = strategy_type
    
    def validate_trade(self, signal: Dict[str, Any], balance: float, current_price: float) -> Tuple[bool, str]:
        """Validate trade with strategy-specific rules"""
        if not signal or not signal.get('action') or not signal.get('structure_stop'):
            return False, "Invalid signal"
        
        if balance <= 0 or current_price <= 0:
            return False, "Invalid market data"
        
        # Stop distance validation
        stop_distance = abs(current_price - signal['structure_stop']) / current_price
        
        if self.active_strategy == "RANGE":
            if stop_distance < 0.0005 or stop_distance > 0.005:
                return False, "Invalid stop distance for range strategy"
        else:  # TREND
            if stop_distance < 0.003 or stop_distance > 0.02:
                return False, "Invalid stop distance for trend strategy"
        
        # Position size validation
        max_position = min(self.active_config['fixed_position_usdt'], balance * 0.8)
        if max_position < 100:
            return False, "Insufficient balance"
        
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
                             unrealized_pnl: float, position_age_seconds: float) -> Tuple[bool, str]:
        """Determine if position should be closed"""
        if entry_price <= 0:
            return False, "hold"
        
        # Emergency stop
        pnl_pct = unrealized_pnl / entry_price if entry_price > 0 else 0
        if pnl_pct <= -self.active_config['emergency_stop_pct']:
            return True, "emergency_stop"
        
        # Max hold time
        if position_age_seconds >= self.active_config['max_position_time']:
            return True, "max_hold_time"
        
        # Strategy-specific exits
        if self.active_strategy == "RANGE":
            return self._check_range_exits(unrealized_pnl, position_age_seconds)
        else:  # TREND
            return self._check_trend_exits(current_price, entry_price, side, unrealized_pnl, position_age_seconds)
    
    def _check_range_exits(self, unrealized_pnl: float, position_age_seconds: float) -> Tuple[bool, str]:
        """Range strategy specific exits"""
        if unrealized_pnl >= self.range_config['profit_threshold_usdt']:
            return True, "profit_target"
        
        if position_age_seconds >= 120 and unrealized_pnl <= 0:
            return True, "timeout_no_profit"
        
        return False, "hold"
    
    def _check_trend_exits(self, current_price: float, entry_price: float, side: str, 
                          unrealized_pnl: float, position_age_seconds: float) -> Tuple[bool, str]:
        """Trend strategy specific exits"""
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
        
        if side.lower() == 'buy':
            return current_price - trail_distance
        else:
            return current_price + trail_distance
    
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
            'position_sizing': self._get_position_sizing_info(),
            'risk_limits': self._get_risk_limits()
        }
    
    def _get_position_sizing_info(self) -> Dict[str, Any]:
        """Get position sizing information"""
        if self.active_strategy == "RANGE":
            return {
                'method': 'Fixed Size',
                'size_usdt': self.range_config['fixed_position_usdt'],
                'profit_target': f"${self.range_config['profit_threshold_usdt']} USDT",
                'hold_time': f"{self.range_config['max_position_time']}s"
            }
        else:
            return {
                'method': 'Risk-Based',
                'max_size_usdt': self.trend_config['fixed_position_usdt'],
                'risk_reward': f"1:{self.trend_config['risk_reward_ratio']}",
                'hold_time': f"{self.trend_config['max_position_time']}s"
            }
    
    def _get_risk_limits(self) -> Dict[str, Any]:
        """Get risk limit information"""
        return {
            'emergency_stop': f"{self.active_config['emergency_stop_pct']*100:.1f}%",
            'max_hold_time': f"{self.active_config['max_position_time']}s",
            'leverage': f"{self.active_config['leverage']}x",
            'trailing_stop': f"{self.trend_config.get('trailing_stop_pct', 0)*100:.1f}%" if self.active_strategy == "TREND" else "None"
        }
    
    def adapt_to_market_condition(self, market_condition: str, volatility: str):
        """Adapt risk parameters based on market conditions"""
        try:
            base_config = (self.range_config if self.active_strategy == "RANGE" 
                          else self.trend_config).copy()
            
            # Calculate volatility multiplier
            if volatility == "HIGH_VOL":
                volatility_multiplier = 0.7
            elif volatility == "LOW_VOL":
                volatility_multiplier = 1.2
            else:
                volatility_multiplier = 1.0
            
            # Apply volatility adjustment
            self.active_config = base_config.copy()
            original_size = self.active_config['fixed_position_usdt']
            self.active_config['fixed_position_usdt'] = int(original_size * volatility_multiplier)
            
        except Exception:
            # Reset to base config on error
            if self.active_strategy == "RANGE":
                self.active_config = self.range_config.copy()
            else:
                self.active_config = self.trend_config.copy()