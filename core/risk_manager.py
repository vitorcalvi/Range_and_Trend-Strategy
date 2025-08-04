import os
from typing import Tuple, Dict, Any
from dotenv import load_dotenv

load_dotenv()

class RiskManager:
    """Streamlined Risk Manager with proper fee calculation"""

    def __init__(self):
        self.symbol = os.getenv('TRADING_SYMBOL')
        self.fee_rate = 0.0011  # 0.11% round-trip fee
        
        # Optimized configuration for minimum exposure, maximum profitability
        self.config = {
            'RANGE': {
                'position_usdt': 5000,    # Optimal: 0.3% return, $5.5 fees, $9.5 net profit
                'profit_target_usdt': 15,
                'max_hold_time': 180,
                'emergency_stop_pct': 0.006,
                'leverage': 10
            },
            'TREND': {
                'position_usdt': 8000,    # Optimal: Risk-based sizing cap, $8.8 fees on max position
                'risk_reward_ratio': 2.5,
                'max_hold_time': 3600,
                'emergency_stop_pct': 0.01,
                'trailing_stop_pct': 0.005,
                'leverage': 10
            }
        }
        
        self.active_strategy = "RANGE"
    
    def set_strategy(self, strategy_type: str):
        """Set active strategy"""
        self.active_strategy = strategy_type if strategy_type in self.config else "RANGE"
    
    def calculate_fees(self, position_usdt: float) -> float:
        """Calculate round-trip fees (0.11%)"""
        return position_usdt * self.fee_rate
    
    def calculate_break_even_pnl(self, position_usdt: float) -> float:
        """Calculate PnL needed to break even after fees"""
        return self.calculate_fees(position_usdt)
    
    def validate_trade(self, signal: Dict[str, Any], balance: float, current_price: float) -> Tuple[bool, str]:
        """Validate trade with streamlined checks"""
        if not signal or not signal.get('action') or not signal.get('structure_stop') or balance <= 0 or current_price <= 0:
            return False, "Invalid signal or market data"
        
        # Stop distance validation
        stop_distance = abs(current_price - signal['structure_stop']) / current_price
        min_stop, max_stop = (0.0005, 0.005) if self.active_strategy == "RANGE" else (0.003, 0.02)
        
        if not (min_stop <= stop_distance <= max_stop):
            return False, f"Invalid stop distance for {self.active_strategy.lower()} strategy"
        
        # Balance validation
        if min(self.config[self.active_strategy]['position_usdt'], balance * 0.8) < 100:
            return False, "Insufficient balance"
        
        return True, "Valid"
    
    def calculate_position_size(self, balance: float, entry_price: float, stop_price: float) -> float:
        """Calculate position size with fee consideration"""
        if balance <= 0 or entry_price <= 0 or stop_price <= 0:
            return 0
        
        max_position_usdt = min(self.config[self.active_strategy]['position_usdt'], balance * 0.5)
        
        if self.active_strategy == "RANGE":
            # Fixed size for range strategy, adjusted for fees
            position_size = max_position_usdt / entry_price
        else:
            # Risk-based sizing for trend strategy
            risk_per_trade = balance * 0.02  # 2% risk
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
        """Streamlined position exit logic with fee consideration"""
        if entry_price <= 0:
            return False, "hold"
        
        cfg = self.config[self.active_strategy]
        
        # Emergency stop
        if (unrealized_pnl / entry_price) <= -cfg['emergency_stop_pct']:
            return True, "emergency_stop"
        
        # Max hold time
        if position_age_seconds >= cfg['max_hold_time']:
            return True, "max_hold_time"
        
        # Calculate position size for fee calculation
        position_usdt = abs(unrealized_pnl / (current_price - entry_price)) * entry_price if current_price != entry_price else cfg['position_usdt']
        break_even_pnl = self.calculate_break_even_pnl(position_usdt)
        
        # Strategy-specific exits with fee consideration
        if self.active_strategy == "RANGE":
            # Profit target above fees
            if unrealized_pnl >= (cfg['profit_target_usdt'] + break_even_pnl):
                return True, "profit_target"
            # Exit if no progress after 2 minutes
            if position_age_seconds >= 120 and unrealized_pnl <= break_even_pnl:
                return True, "timeout_no_profit"
        else:  # TREND strategy
            if unrealized_pnl > 0:
                risk_amount = entry_price * 0.02  # 2% risk
                target_profit = risk_amount * cfg['risk_reward_ratio'] + break_even_pnl
                if unrealized_pnl >= target_profit:
                    return True, "profit_target"
        
        return False, "hold"
    
    def calculate_trailing_stop(self, current_price: float, entry_price: float, side: str) -> float:
        """Calculate trailing stop for trend strategy"""
        if self.active_strategy != "TREND":
            return 0
        
        trail_distance = current_price * self.config['TREND']['trailing_stop_pct']
        return current_price - trail_distance if side.lower() == 'buy' else current_price + trail_distance
    
    def get_leverage(self) -> int:
        """Get leverage setting"""
        return self.config[self.active_strategy]['leverage']
    
    def get_max_position_time(self) -> int:
        """Get maximum position time"""
        return self.config[self.active_strategy]['max_hold_time']
    
    def adapt_to_market_condition(self, market_condition: str, volatility: str):
        """Adapt position size based on market conditions"""
        try:
            # Volatility adjustment multipliers
            multiplier = {"HIGH_VOL": 0.7, "LOW_VOL": 1.2}.get(volatility, 1.0)
            
            # Apply to both strategies
            for strategy in ['RANGE', 'TREND']:
                base_size = 5000 if strategy == 'RANGE' else 8000
                self.config[strategy]['position_usdt'] = int(base_size * multiplier)
        except:
            # Reset to optimal defaults on error
            self.config['RANGE']['position_usdt'] = 5000
            self.config['TREND']['position_usdt'] = 8000