import os
from dotenv import load_dotenv
from typing import Tuple, Dict, Any

load_dotenv()

class RiskManager:
    """
    Dual Strategy Risk Manager
    
    Manages risk for both Range and Trend strategies with adaptive parameters:
    
    RANGE STRATEGY (1-minute scalping):
    - Fixed position: $9091 USDT 
    - Quick profit targets: $15 USDT
    - Fast exits: 180 seconds max
    - Tight stops: 0.6% emergency
    
    TREND STRATEGY (15-minute following):
    - Larger positions: $12,000-15,000 USDT
    - Higher profit targets: 1:2.5 RR
    - Longer holds: 3600 seconds max
    - Trailing stops: 0.5% trail
    """

    def __init__(self):
        self.symbol = os.getenv('TRADING_SYMBOL')
        
        # Range strategy config (original scalping)
        self.range_config = {
            'fixed_position_usdt': 9091,
            'profit_threshold_usdt': 15,
            'max_position_time': 180,
            'emergency_stop_pct': 0.006,
            'leverage': 10
        }
        
        # Trend strategy config (new trend following)
        self.trend_config = {
            'fixed_position_usdt': 12000,
            'risk_reward_ratio': 2.5,
            'max_position_time': 3600,
            'emergency_stop_pct': 0.01,
            'trailing_stop_pct': 0.005,
            'profit_lock_threshold': 1.5,
            'leverage': 10
        }
        
        # Current active configuration
        self.active_config = self.range_config
        self.active_strategy = "RANGE"
    
    def set_strategy(self, strategy_type: str):
        """Set active strategy configuration with validation"""
        if strategy_type not in ["RANGE", "TREND"]:
            print(f"⚠️ Unknown strategy type: {strategy_type}, defaulting to RANGE")
            strategy_type = "RANGE"
            
        if strategy_type == "RANGE":
            self.active_config = self.range_config.copy()
        else:  # TREND
            self.active_config = self.trend_config.copy()
            
        self.active_strategy = strategy_type
        print(f"🛡️ Risk Manager switched to {strategy_type} strategy")
    
    def validate_trade(self, signal: Dict[str, Any], balance: float, current_price: float) -> Tuple[bool, str]:
        """Validate trade with strategy-specific rules"""
        if not signal or not signal.get('action') or not signal.get('structure_stop'):
            return False, "Invalid signal"
        
        # Basic validation
        if balance <= 0 or current_price <= 0:
            return False, "Invalid market data"
        
        # Stop distance validation
        stop_distance = abs(current_price - signal['structure_stop']) / current_price
        
        if self.active_strategy == "RANGE":
            # Tight stops for scalping
            if stop_distance < 0.0005 or stop_distance > 0.005:
                return False, "Invalid stop distance for range strategy"
        else:  # TREND
            # Wider stops for trend following
            if stop_distance < 0.003 or stop_distance > 0.02:
                return False, "Invalid stop distance for trend strategy"
        
        # Position size validation
        max_position = min(self.active_config['fixed_position_usdt'], balance * 0.8)
        if max_position < 100:  # Minimum position size
            return False, "Insufficient balance"
        
        return True, "Valid"
    
    def calculate_position_size(self, balance: float, entry_price: float, stop_price: float) -> float:
        """Calculate position size based on active strategy"""
        if balance <= 0 or entry_price <= 0 or stop_price <= 0:
            return 0
        
        # Get position sizing parameters
        max_position_usdt = min(self.active_config['fixed_position_usdt'], balance * 0.5)
        
        if self.active_strategy == "RANGE":
            # Fixed size for scalping
            position_size = max_position_usdt / entry_price
        else:  # TREND
            # Risk-based sizing for trend following
            risk_per_trade = balance * 0.02  # 2% risk per trade
            stop_distance = abs(entry_price - stop_price)
            
            if stop_distance > 0:
                position_size = risk_per_trade / stop_distance
                # Cap at maximum position size
                max_size = max_position_usdt / entry_price
                position_size = min(position_size, max_size)
            else:
                position_size = max_position_usdt / entry_price
        
        return round(max(position_size, 0), 6)
    
    def should_close_position(self, current_price: float, entry_price: float, side: str, 
                             unrealized_pnl: float, position_age_seconds: float) -> Tuple[bool, str]:
        """Determine if position should be closed with strategy-specific logic"""
        
        if entry_price <= 0:
            return False, "hold"
        
        # Emergency stop (applies to both strategies)
        pnl_pct = unrealized_pnl / entry_price if entry_price > 0 else 0
        if pnl_pct <= -self.active_config['emergency_stop_pct']:
            return True, "emergency_stop"
        
        # Max hold time (strategy-specific)
        if position_age_seconds >= self.active_config['max_position_time']:
            return True, "max_hold_time"
        
        # Strategy-specific exit logic
        if self.active_strategy == "RANGE":
            return self._check_range_exits(unrealized_pnl, position_age_seconds)
        else:  # TREND
            return self._check_trend_exits(current_price, entry_price, side, unrealized_pnl, position_age_seconds)
    
    def _check_range_exits(self, unrealized_pnl: float, position_age_seconds: float) -> Tuple[bool, str]:
        """Range strategy specific exits - quick scalping"""
        # Quick profit target
        if unrealized_pnl >= self.range_config['profit_threshold_usdt']:
            return True, "profit_target"
        
        # Early exit if holding too long without profit
        if position_age_seconds >= 120 and unrealized_pnl <= 0:
            return True, "timeout_no_profit"
        
        return False, "hold"
    
    def _check_trend_exits(self, current_price: float, entry_price: float, side: str, 
                          unrealized_pnl: float, position_age_seconds: float) -> Tuple[bool, str]:
        """Trend strategy specific exits - trailing and profit locking"""
        
        # Profit lock threshold (lock in profits after good run)
        if unrealized_pnl > 0:
            profit_ratio = unrealized_pnl / (entry_price * 0.02)  # Relative to 2% risk
            
            if profit_ratio >= self.trend_config['profit_lock_threshold']:
                # Could implement trailing stop logic here
                # For now, use simple profit target
                target_profit = entry_price * 0.02 * self.trend_config['risk_reward_ratio']
                if unrealized_pnl >= target_profit:
                    return True, "profit_target"
        
        # Time-based exits for trends
        if position_age_seconds >= self.trend_config['max_position_time']:
            return True, "max_hold_time"
        
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
        """Adapt risk parameters based on market conditions with validation"""
        try:
            # Ensure we have the base config
            if self.active_strategy == "RANGE":
                base_config = self.range_config.copy()
            else:
                base_config = self.trend_config.copy()
            
            # Calculate volatility multiplier
            volatility_multiplier = 1.0
            if volatility == "HIGH_VOL":
                volatility_multiplier = 0.7  # Smaller positions in high volatility
            elif volatility == "LOW_VOL":
                volatility_multiplier = 1.2  # Larger positions in low volatility
            
            # Apply volatility adjustment to position sizing
            self.active_config = base_config.copy()
            original_size = self.active_config['fixed_position_usdt']
            self.active_config['fixed_position_usdt'] = int(original_size * volatility_multiplier)
            
            print(f"📊 Adapted to {market_condition} + {volatility}: Position ${original_size} → ${self.active_config['fixed_position_usdt']}")
            
        except Exception as e:
            print(f"❌ Error adapting to market conditions: {e}")
            # Reset to base config on error
            if self.active_strategy == "RANGE":
                self.active_config = self.range_config.copy()
            else:
                self.active_config = self.trend_config.copy()