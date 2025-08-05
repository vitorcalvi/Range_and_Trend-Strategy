from decimal import Decimal
from typing import Dict, Any

class FeeCalculator:
    """Centralized fee calculations to eliminate duplication"""
    
    def __init__(self):
        self.taker_fee_rate = Decimal('0.00055')    # 0.055%
        self.maker_fee_rate = Decimal('0.0001')     # 0.01%
        self.maker_fill_ratio = Decimal('0.3')      # 30% assumption
        self.slippage_rate = Decimal('0.0002')      # 0.02%
        
        # Blended rate: 30% maker + 70% taker
        self.blended_fee_rate = (self.maker_fill_ratio * self.maker_fee_rate + 
                               (1 - self.maker_fill_ratio) * self.taker_fee_rate)
        self.total_cost_rate = self.blended_fee_rate + self.slippage_rate
    
    def calculate_breakeven_rate(self, risk_pct: float, reward_ratio: float) -> float:
        """Calculate break-even win rate: (Risk + Fee) / (Risk + Reward)"""
        reward_pct = risk_pct * reward_ratio
        fee_pct = float(self.total_cost_rate)
        return (risk_pct + fee_pct) / (risk_pct + reward_pct)
    
    def get_fee_analysis(self, position_size_usdt: float, reward_pct: float) -> Dict[str, Any]:
        """Analyze fee efficiency for position size"""
        if position_size_usdt <= 0 or reward_pct <= 0:
            return {"efficiency_ratio": 0, "fee_percentage": 100}
        
        position_decimal = Decimal(str(position_size_usdt))
        total_cost = position_decimal * self.total_cost_rate
        expected_gross = position_decimal * Decimal(str(reward_pct))
        
        if expected_gross <= 0:
            return {"efficiency_ratio": 0, "fee_percentage": 100}
        
        return {
            "total_cost": float(total_cost),
            "expected_gross": float(expected_gross),
            "efficiency_ratio": float(expected_gross / total_cost),
            "fee_percentage": float(total_cost / expected_gross * 100)
        }
    
    def is_fee_efficient(self, position_size_usdt: float, reward_pct: float, min_ratio: float = 15) -> bool:
        """Check if position meets minimum fee efficiency"""
        analysis = self.get_fee_analysis(position_size_usdt, reward_pct)
        return analysis["efficiency_ratio"] >= min_ratio
    
    def get_order_strategy(self, confidence: float) -> Dict[str, Any]:
        """Get limit-first order strategy based on confidence"""
        if confidence >= 80:
            offset = 0.0005  # 0.05%
        elif confidence >= 70:
            offset = 0.001   # 0.1%
        else:
            offset = 0.0015  # 0.15%
        
        return {
            "limit_offset": offset,
            "expected_maker_rate": float(self.maker_fee_rate),
            "expected_taker_rate": float(self.taker_fee_rate),
            "blended_rate": float(self.blended_fee_rate)
        }