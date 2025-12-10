from decimal import Decimal
from typing import Dict, Any

class FeeCalculator:
    """Centralized fee calculations with CORRECT Bybit rates"""
    
    def __init__(self):
        # CORRECT BYBIT SPOT FEES
        self.taker_fee_rate = Decimal('0.001')      # 0.1% for spot
        self.maker_fee_rate = Decimal('0.001')      # 0.1% for spot
        self.maker_fill_ratio = Decimal('0.3')      # 30% realistic for aggressive trading
        self.slippage_rate = Decimal('0.003')       # 0.3% realistic slippage
        
        # Blended rate calculation
        self.blended_fee_rate = (self.maker_fill_ratio * self.maker_fee_rate + 
                                 (1 - self.maker_fill_ratio) * self.taker_fee_rate)
        self.total_cost_rate = self.blended_fee_rate + self.slippage_rate  # 0.4% per side
        self.round_trip_cost = self.total_cost_rate * 2  # 0.8% total
    
    def calculate_breakeven_rate(self, risk_pct: float, reward_ratio: float) -> float:
        """Calculate CORRECT break-even win rate"""
        reward_pct = risk_pct * reward_ratio
        round_trip_fee = float(self.round_trip_cost)
        return (risk_pct + round_trip_fee) / (risk_pct + reward_pct)
    
    def get_minimum_profit_target(self, position_size_usdt: float) -> float:
        """Calculate minimum profit target to cover fees + margin"""
        round_trip_fees = float(position_size_usdt * self.round_trip_cost)
        minimum_margin = max(15.0, position_size_usdt * 0.01)  # $15 or 1%, whichever is higher
        return round_trip_fees + minimum_margin
    
    def get_breakeven_pnl(self, position_size_usdt: float) -> float:
        """Calculate breakeven PnL (fees only)"""
        return float(position_size_usdt * self.round_trip_cost)
    
    def validate_profitability(self, position_size_usdt: float, expected_profit: float) -> bool:
        """Check if trade meets minimum profitability threshold"""
        min_target = self.get_minimum_profit_target(position_size_usdt)
        return expected_profit >= min_target
    
    def get_required_price_move(self, position_size_usdt: float, target_profit: float) -> float:
        """Calculate required price movement percentage"""
        gross_needed = target_profit + self.get_breakeven_pnl(position_size_usdt)
        return (gross_needed / position_size_usdt) * 100