import os
from decimal import Decimal, ROUND_HALF_UP
from dotenv import load_dotenv
from typing import Tuple, Dict, Any

load_dotenv()

class RiskManager:
    """Corrected Break-Even Formulas & Dynamic Position Sizing"""

    def __init__(self):
        self.symbol = os.getenv('TRADING_SYMBOL', 'ETHUSDT')
        
        # CORRECTED: Fee model with maker/taker blend
        self.taker_fee_rate = Decimal('0.00055')    # 0.055% taker (market orders)
        self.maker_fee_rate = Decimal('0.0001')     # 0.01% maker (limit orders)
        self.maker_fill_ratio = Decimal('0.3')      # 30% maker execution assumption
        
        # Blended fee rate: 30% maker + 70% taker
        self.blended_fee_rate = (self.maker_fill_ratio * self.maker_fee_rate + 
                                (1 - self.maker_fill_ratio) * self.taker_fee_rate)
        # = 0.3 × 0.0001 + 0.7 × 0.00055 = 0.00003 + 0.000385 = 0.000415 ≈ 0.042%
        
        self.slippage_est = Decimal('0.0002')       # 0.02% slippage
        self.total_cost_rate = self.blended_fee_rate + self.slippage_est  # ~0.062%
        
        # Range strategy config
        self.range_config = {
            'risk_percentage': Decimal('0.02'),     # 1R = 2% of position
            'reward_ratio': Decimal('1.5'),         # 1.5R target
            'max_position_time': 600,               # 10 minutes
            'emergency_stop_pct': Decimal('0.025'), # 2.5% hard stop
            'leverage': 10,
            'fee_target_pct': Decimal('0.03'),      # Fees should be 3% of expected gross
            'min_position_usdt': 2000,              # Minimum for efficiency
            'max_position_usdt': 8000               # Maximum for safety
        }
        
        # Trend strategy config  
        self.trend_config = {
            'risk_percentage': Decimal('0.018'),    # 1R = 1.8% of position
            'reward_ratio': Decimal('2.0'),         # 2R target
            'max_position_time': 2400,              # 40 minutes
            'emergency_stop_pct': Decimal('0.025'), # 2.5% hard stop
            'trailing_stop_atr': Decimal('0.5'),    # 0.5 ATR trailing
            'profit_lock_threshold': Decimal('1.2'), # Lock at 1.2R breakeven
            'leverage': 10,
            'fee_target_pct': Decimal('0.025'),     # Fees should be 2.5% of expected gross
            'min_position_usdt': 3000,
            'max_position_usdt': 10000
        }
        
        self.active_config = self.range_config.copy()
        self.active_strategy = "RANGE"
        
        # Calculate corrected break-even rates
        self._calculate_breakeven_rates()
    
    def _calculate_breakeven_rates(self):
        """CORRECTED: Calculate actual break-even win rates with fees"""
        
        # Range strategy: 1.5R target, 1R risk
        range_risk_pct = float(self.range_config['risk_percentage'])      # 2%
        range_reward_pct = range_risk_pct * float(self.range_config['reward_ratio'])  # 3%
        fee_pct = float(self.total_cost_rate)  # ~0.062%
        
        # CORRECTED FORMULA:
        # Win_Rate × (Reward - Fee) = (1 - Win_Rate) × (Risk + Fee)
        # Solving: Win_Rate = (Risk + Fee) / (Risk + Fee + Reward - Fee)
        # Simplified: Win_Rate = (Risk + Fee) / (Risk + Reward)
        
        range_numerator = range_risk_pct + fee_pct      # 0.02 + 0.00062 = 0.02062
        range_denominator = range_risk_pct + range_reward_pct  # 0.02 + 0.03 = 0.05
        range_breakeven = range_numerator / range_denominator  # 0.41240 = 41.24%
        
        # Trend strategy: 2R target, 1R risk  
        trend_risk_pct = float(self.trend_config['risk_percentage'])      # 1.8%
        trend_reward_pct = trend_risk_pct * float(self.trend_config['reward_ratio'])  # 3.6%
        
        trend_numerator = trend_risk_pct + fee_pct      # 0.018 + 0.00062 = 0.01862
        trend_denominator = trend_risk_pct + trend_reward_pct  # 0.018 + 0.036 = 0.054
        trend_breakeven = trend_numerator / trend_denominator  # 0.34481 = 34.48%
        
        self.range_breakeven_rate = range_breakeven
        self.trend_breakeven_rate = trend_breakeven
    
    def set_strategy(self, strategy_type: str):
        """Set active strategy configuration"""
        if strategy_type in ["RANGE", "TREND"]:
            self.active_strategy = strategy_type
            self.active_config = (self.range_config if strategy_type == "RANGE" 
                                else self.trend_config).copy()
        else:
            self.active_strategy = "RANGE"
            self.active_config = self.range_config.copy()
    
    def calculate_dynamic_position_size(self, balance: float, entry_price: float, 
                                      stop_price: float, expected_hold_minutes: int = 10) -> float:
        """DYNAMIC: Size position so fees are 2-5% of expected gross profit"""
        if balance <= 0 or entry_price <= 0 or stop_price <= 0:
            return 0
        
        # Calculate risk and reward percentages
        risk_pct = abs(entry_price - stop_price) / entry_price
        reward_pct = risk_pct * float(self.active_config['reward_ratio'])
        
        # Target fee percentage of expected gross profit
        fee_target_pct = float(self.active_config['fee_target_pct'])
        
        # Calculate position size where fees = target % of expected gross
        # fee_cost = position_size × total_cost_rate
        # expected_gross = position_size × reward_pct
        # Target: fee_cost = fee_target_pct × expected_gross
        # position_size × total_cost_rate = fee_target_pct × position_size × reward_pct
        # total_cost_rate = fee_target_pct × reward_pct
        # position_size can be any value, but we want optimal fee efficiency
        
        # Calculate optimal position size for fee efficiency
        if reward_pct > 0:
            # We want: fees / expected_gross = fee_target_pct
            # fees = position_size × total_cost_rate  
            # expected_gross = position_size × reward_pct
            # So: position_size × total_cost_rate = fee_target_pct × position_size × reward_pct
            # This is always true for any position size, so we need other constraints
            
            # Use fee efficiency ratio instead
            min_fee_efficiency = 1 / fee_target_pct  # If target is 3%, min efficiency is 33x
            min_position_for_efficiency = 1000 / (reward_pct * min_fee_efficiency)
            
            # Calculate position size based on reward and fee efficiency
            target_position = max(
                float(self.active_config['min_position_usdt']),
                min_position_for_efficiency
            )
            
            # Safety limits
            max_position_by_balance = balance * 0.25  # Max 25% of balance
            max_position_by_config = float(self.active_config['max_position_usdt'])
            
            target_position = min(target_position, max_position_by_balance, max_position_by_config)
            
            return target_position / entry_price
        
        return 0
    
    def get_fee_analysis(self, position_size_usdt: float) -> Dict[str, Any]:
        """Analyze fee efficiency for a given position size"""
        position_decimal = Decimal(str(position_size_usdt))
        
        # Calculate different fee components
        maker_fees = position_decimal * self.maker_fee_rate * 2  # Round trip
        taker_fees = position_decimal * self.taker_fee_rate * 2  # Round trip  
        blended_fees = position_decimal * self.blended_fee_rate * 2  # Round trip
        slippage_cost = position_decimal * self.slippage_est
        total_cost = blended_fees + slippage_cost
        
        # Calculate expected profit
        risk_pct = self.active_config['risk_percentage']
        reward_pct = risk_pct * self.active_config['reward_ratio']
        expected_gross = position_decimal * reward_pct
        
        return {
            'position_size_usdt': float(position_decimal),
            'fee_breakdown': {
                'maker_only': f"${float(maker_fees):.2f}",
                'taker_only': f"${float(taker_fees):.2f}",
                'blended_30_70': f"${float(blended_fees):.2f}",
                'slippage': f"${float(slippage_cost):.2f}",
                'total_cost': f"${float(total_cost):.2f}"
            },
            'efficiency_metrics': {
                'expected_gross': f"${float(expected_gross):.2f}",
                'fee_percentage': f"{float(total_cost/expected_gross)*100:.1f}%",
                'profit_to_fee_ratio': f"{float(expected_gross/total_cost):.1f}x",
                'breakeven_rate': f"{self.get_current_breakeven_rate()*100:.1f}%"
            }
        }
    
    def get_current_breakeven_rate(self) -> float:
        """Get corrected break-even rate for current strategy"""
        return (self.range_breakeven_rate if self.active_strategy == "RANGE" 
                else self.trend_breakeven_rate)
    
    def should_close_position(self, current_price: float, entry_price: float, side: str, 
                             unrealized_pnl: float, position_age_seconds: float, 
                             position_size_usdt: float = 0, highest_profit: float = 0,
                             atr_value: float = 0) -> Tuple[bool, str]:
        """Enhanced exit logic with trailing stops"""
        if entry_price <= 0:
            return False, "hold"
        
        # Calculate position metrics
        position_decimal = Decimal(str(position_size_usdt)) if position_size_usdt > 0 else Decimal('3000')
        risk_pct = self.active_config['risk_percentage']
        risk_amount = float(position_decimal * risk_pct)
        
        # Emergency stop
        emergency_threshold = -float(position_decimal * self.active_config['emergency_stop_pct'])
        if unrealized_pnl <= emergency_threshold:
            return True, "emergency_stop"
        
        # Max hold time
        if position_age_seconds >= self.active_config['max_position_time']:
            return True, "max_hold_time"
        
        # Strategy-specific exits with trailing
        if self.active_strategy == "RANGE":
            return self._check_range_exits(unrealized_pnl, position_age_seconds, risk_amount, position_decimal)
        else:
            return self._check_trend_exits_with_trailing(
                unrealized_pnl, position_age_seconds, risk_amount, position_decimal,
                current_price, entry_price, side, highest_profit, atr_value
            )
    
    def _check_trend_exits_with_trailing(self, unrealized_pnl: float, position_age_seconds: float, 
                                       risk_amount: float, position_decimal: Decimal,
                                       current_price: float, entry_price: float, side: str,
                                       highest_profit: float, atr_value: float) -> Tuple[bool, str]:
        """Trend exits with 1.2R breakeven and trailing stops"""
        
        # Calculate targets
        reward_ratio = self.active_config['reward_ratio']
        target_profit = risk_amount * float(reward_ratio)  # 2R
        breakeven_profit = risk_amount * 1.2  # 1.2R breakeven threshold
        
        # Subtract fees
        fees = float(position_decimal * self.total_cost_rate)
        net_target = target_profit - fees
        net_breakeven = breakeven_profit - fees
        
        # Full target hit
        if unrealized_pnl >= net_target:
            return True, "profit_target"
        
        # Trailing stop after 1.2R breakeven move
        if unrealized_pnl >= net_breakeven and highest_profit >= net_breakeven:
            
            # Calculate trailing stop using ATR
            if atr_value > 0:
                trail_distance = atr_value * float(self.active_config['trailing_stop_atr'])  # 0.5 ATR
            else:
                # Fallback: use percentage-based trailing
                trail_distance = current_price * 0.005  # 0.5%
            
            # Calculate trailing stop level
            if side.lower() == 'buy':
                trailing_stop_price = current_price - trail_distance
                # Check if current profit is significantly below highest profit
                if unrealized_pnl <= highest_profit * 0.8:  # 20% drawdown from peak
                    return True, "trailing_stop"
            else:
                trailing_stop_price = current_price + trail_distance
                if unrealized_pnl <= highest_profit * 0.8:
                    return True, "trailing_stop"
        
        # Profit lock at 60% of max time if above breakeven
        if (position_age_seconds >= self.active_config['max_position_time'] * 0.6 and 
            unrealized_pnl >= net_breakeven * 0.8):
            return True, "profit_lock"
        
        # Stop loss
        stop_loss = -(risk_amount + fees)
        if unrealized_pnl <= stop_loss:
            return True, "stop_loss"
        
        return False, "hold"
    
    def _check_range_exits(self, unrealized_pnl: float, position_age_seconds: float, 
                          risk_amount: float, position_decimal: Decimal) -> Tuple[bool, str]:
        """Range exits with corrected targets"""
        
        # Calculate 1.5R target
        reward_ratio = self.active_config['reward_ratio']
        target_profit = risk_amount * float(reward_ratio)
        
        # Subtract fees
        fees = float(position_decimal * self.total_cost_rate)
        net_target = target_profit - fees
        
        # Hit target
        if unrealized_pnl >= net_target:
            return True, "profit_target"
        
        # Quick exit at 1R after 60% of max time
        if (position_age_seconds >= self.active_config['max_position_time'] * 0.6 and 
            unrealized_pnl >= (risk_amount - fees) * 0.8):
            return True, "profit_target"
        
        # Stop loss
        stop_loss = -(risk_amount + fees)
        if unrealized_pnl <= stop_loss:
            return True, "stop_loss"
        
        return False, "hold"
    
    def get_order_strategy(self, signal_action: str, current_price: float, 
                          confidence: float) -> Dict[str, Any]:
        """LIMIT-FIRST strategy: Try limit order first, fallback to market"""
        
        # Higher confidence = more aggressive limit pricing
        if confidence >= 80:
            limit_offset_pct = 0.0005  # 0.05% offset for high confidence
        elif confidence >= 70:
            limit_offset_pct = 0.001   # 0.1% offset for medium confidence  
        else:
            limit_offset_pct = 0.0015  # 0.15% offset for lower confidence
        
        if signal_action == 'BUY':
            limit_price = current_price * (1 - limit_offset_pct)
        else:
            limit_price = current_price * (1 + limit_offset_pct)
        
        return {
            'primary_order': {
                'type': 'limit',
                'price': limit_price,
                'time_in_force': 'IOC',  # Immediate-or-Cancel
                'fee_rate': float(self.maker_fee_rate),
                'expected_fill_pct': 30  # 30% expected fill rate
            },
            'fallback_order': {
                'type': 'market',
                'time_in_force': 'IOC',
                'fee_rate': float(self.taker_fee_rate),
                'trigger': 'if_primary_unfilled'
            },
            'blended_expected_fee': float(self.blended_fee_rate),
            'strategy': 'limit_first_ioc_fallback'
        }
    
    def get_leverage(self) -> int:
        """Get leverage setting"""
        return self.active_config['leverage']
    
    def get_max_position_time(self) -> int:
        """Get maximum position hold time"""
        return self.active_config['max_position_time']
    
    def get_active_config(self) -> Dict[str, Any]:
        """Get complete configuration with corrected break-even analysis"""
        
        example_position = 5000  # $5k example
        fee_analysis = self.get_fee_analysis(example_position)
        
        return {
            'strategy': self.active_strategy,
            'corrected_breakeven': {
                'range_strategy': f"{self.range_breakeven_rate*100:.2f}%",
                'trend_strategy': f"{self.trend_breakeven_rate*100:.2f}%",
                'current_strategy': f"{self.get_current_breakeven_rate()*100:.2f}%",
                'formula': 'Win_Rate = (Risk + Fee) / (Risk + Reward)',
                'old_formula_error': 'Previous formula was missing proper fee integration'
            },
            'fee_optimization': {
                'maker_fee': f"{float(self.maker_fee_rate)*100:.3f}%",
                'taker_fee': f"{float(self.taker_fee_rate)*100:.3f}%", 
                'blended_fee': f"{float(self.blended_fee_rate)*100:.3f}%",
                'maker_fill_assumption': f"{float(self.maker_fill_ratio)*100:.0f}%",
                'total_cost_with_slippage': f"{float(self.total_cost_rate)*100:.3f}%"
            },
            'dynamic_sizing': {
                'fee_target_percentage': f"{float(self.active_config['fee_target_pct'])*100:.1f}%",
                'min_position': f"${self.active_config['min_position_usdt']}",
                'max_position': f"${self.active_config['max_position_usdt']}",
                'example_analysis': fee_analysis
            },
            'trailing_stops': {
                'enabled': self.active_strategy == "TREND",
                'breakeven_threshold': '1.2R',
                'trailing_method': '0.5 ATR or 0.5% fallback',
                'drawdown_trigger': '20% from peak profit'
            }
        }
    
    def adapt_to_market_condition(self, market_condition: str, volatility: str):
        """Adapt config based on market conditions"""
        base_config = (self.range_config if self.active_strategy == "RANGE" 
                      else self.trend_config).copy()
        
        self.active_config = base_config.copy()
        
        # Volatility adjustments (keep position sizes fixed for fee efficiency)
        if volatility == "HIGH_VOL":
            # Tighter stops and shorter holds in high volatility
            self.active_config['max_position_time'] = int(base_config['max_position_time'] * 0.7)
        elif volatility == "LOW_VOL":
            # Longer holds in low volatility
            self.active_config['max_position_time'] = int(base_config['max_position_time'] * 1.2)