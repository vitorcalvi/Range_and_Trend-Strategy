#!/usr/bin/env python3
"""
Comprehensive Trading Conditions Test
Tests all scenarios for the Dual Strategy Trading Bot
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from strategies.range_strategy import RangeStrategy
from strategies.trend_strategy import TrendStrategy
from strategies.strategy_manager import StrategyManager, MarketConditionDetector
from core.risk_manager import RiskManager

class TradingConditionsTest:
    """Comprehensive test suite for all trading conditions"""
    
    def __init__(self):
        self.range_strategy = RangeStrategy()
        self.trend_strategy = TrendStrategy()
        self.strategy_manager = StrategyManager()
        self.risk_manager = RiskManager()
        self.detector = MarketConditionDetector()
        
        self.test_results = []
        self.total_tests = 0
        self.passed_tests = 0
        
    def log_test(self, test_name: str, condition: bool, details: str = ""):
        """Log test result"""
        self.total_tests += 1
        if condition:
            self.passed_tests += 1
            status = "✅ PASS"
        else:
            status = "❌ FAIL"
        
        result = f"{status} | {test_name}"
        if details:
            result += f" | {details}"
        
        self.test_results.append(result)
        print(result)
        
        return condition
    
    def create_market_data(self, scenario: str, periods: int = 100) -> tuple:
        """Create synthetic market data for different scenarios"""
        np.random.seed(42)  # Reproducible results
        
        base_price = 3500.0
        timestamps = [datetime.now() - timedelta(minutes=i) for i in range(periods, 0, -1)]
        
        if scenario == "STRONG_RANGE":
            # Sideways market with low volatility
            price_changes = np.random.normal(0, 0.002, periods)
            prices = [base_price]
            for change in price_changes[1:]:
                new_price = prices[-1] * (1 + change)
                prices.append(max(base_price * 0.98, min(base_price * 1.02, new_price)))
                
        elif scenario == "WEAK_RANGE":
            # Slightly more volatile sideways
            price_changes = np.random.normal(0, 0.004, periods)
            prices = [base_price]
            for change in price_changes[1:]:
                new_price = prices[-1] * (1 + change)
                prices.append(max(base_price * 0.95, min(base_price * 1.05, new_price)))
                
        elif scenario == "TRENDING":
            # Moderate uptrend
            trend = 0.0008
            price_changes = np.random.normal(trend, 0.003, periods)
            prices = [base_price]
            for change in price_changes[1:]:
                prices.append(prices[-1] * (1 + change))
                
        elif scenario == "STRONG_TREND":
            # Strong uptrend
            trend = 0.0015
            price_changes = np.random.normal(trend, 0.004, periods)
            prices = [base_price]
            for change in price_changes[1:]:
                prices.append(prices[-1] * (1 + change))
                
        else:  # DEFAULT
            prices = [base_price + np.random.normal(0, 10) for _ in range(periods)]
        
        # Create OHLCV data
        data = []
        for i, price in enumerate(prices):
            high = price * (1 + abs(np.random.normal(0, 0.001)))
            low = price * (1 - abs(np.random.normal(0, 0.001)))
            volume = np.random.uniform(1000, 5000)
            
            data.append({
                'timestamp': timestamps[i],
                'open': price,
                'high': high,
                'low': low,
                'close': price,
                'volume': volume
            })
        
        df = pd.DataFrame(data).set_index('timestamp').sort_index()
        
        # Create both 1m and 15m versions
        df_1m = df.copy()
        df_15m = df.iloc[::15].copy()  # Sample every 15th row for 15m data
        
        return df_1m, df_15m
    
    def test_fee_calculations(self):
        """Test fee calculation accuracy"""
        print("\n🔍 TESTING FEE CALCULATIONS")
        print("=" * 50)
        
        # Test fee rate
        expected_rate = 0.0011
        actual_rate = self.risk_manager.fee_rate
        self.log_test("Fee Rate", actual_rate == expected_rate, f"{actual_rate} == {expected_rate}")
        
        # Test fee calculations for different position sizes
        test_positions = [1000, 5000, 8000, 10000]
        for pos in test_positions:
            expected_fee = pos * 0.0011
            actual_fee = self.risk_manager.calculate_fees(pos)
            self.log_test(f"Fee Calc ${pos}", 
                         abs(actual_fee - expected_fee) < 0.01,
                         f"${actual_fee:.2f} ≈ ${expected_fee:.2f}")
        
        # Test break-even calculations
        for pos in test_positions:
            break_even = self.risk_manager.calculate_break_even_pnl(pos)
            expected = pos * 0.0011
            self.log_test(f"Break-even ${pos}",
                         abs(break_even - expected) < 0.01,
                         f"${break_even:.2f} ≈ ${expected:.2f}")
    
    def test_position_sizing(self):
        """Test position sizing logic"""
        print("\n📊 TESTING POSITION SIZING")
        print("=" * 50)
        
        # Test range strategy sizing
        self.risk_manager.set_strategy("RANGE")
        range_config = self.risk_manager.config["RANGE"]
        expected_range = 5000
        actual_range = range_config["position_usdt"]
        self.log_test("Range Position Size", 
                     actual_range == expected_range,
                     f"${actual_range} == ${expected_range}")
        
        # Test trend strategy sizing
        self.risk_manager.set_strategy("TREND")
        trend_config = self.risk_manager.config["TREND"]
        expected_trend = 8000
        actual_trend = trend_config["position_usdt"]
        self.log_test("Trend Position Size",
                     actual_trend == expected_trend,
                     f"${actual_trend} == ${expected_trend}")
        
        # Test position size calculation
        balance = 50000
        entry_price = 3500
        stop_price = 3480
        
        self.risk_manager.set_strategy("RANGE")
        range_size = self.risk_manager.calculate_position_size(balance, entry_price, stop_price)
        expected_range_size = 5000 / entry_price
        self.log_test("Range Size Calculation",
                     abs(range_size - expected_range_size) < 0.001,
                     f"{range_size:.6f} ≈ {expected_range_size:.6f}")
        
        self.risk_manager.set_strategy("TREND")
        trend_size = self.risk_manager.calculate_position_size(balance, entry_price, stop_price)
        # For trend, it's risk-based, so just check it's reasonable
        self.log_test("Trend Size Calculation",
                     0.1 < trend_size < 10,
                     f"{trend_size:.6f} in reasonable range")
    
    def test_market_condition_detection(self):
        """Test market condition detection"""
        print("\n🧠 TESTING MARKET CONDITION DETECTION")
        print("=" * 50)
        
        scenarios = ["STRONG_RANGE", "WEAK_RANGE", "TRENDING", "STRONG_TREND"]
        
        for scenario in scenarios:
            data_1m, data_15m = self.create_market_data(scenario, 100)
            
            # Test ADX calculation
            adx = self.detector.calculate_adx(data_15m['high'], data_15m['low'], data_15m['close'])
            self.log_test(f"ADX Calculation {scenario}",
                         0 <= adx <= 100,
                         f"ADX: {adx:.1f}")
            
            # Test market condition detection
            condition_info = self.detector.detect_market_condition(data_1m, data_15m)
            detected_condition = condition_info["condition"]
            confidence = condition_info["confidence"]
            
            self.log_test(f"Condition Detection {scenario}",
                         detected_condition != "INSUFFICIENT_DATA",
                         f"Detected: {detected_condition}, Confidence: {confidence:.1f}")
            
            # Test volatility regime detection
            vol_regime = self.detector.calculate_volatility_regime(data_1m['close'])
            self.log_test(f"Volatility Detection {scenario}",
                         vol_regime in ["LOW_VOL", "NORMAL", "HIGH_VOL"],
                         f"Volatility: {vol_regime}")
    
    def test_range_strategy_signals(self):
        """Test range strategy signal generation"""
        print("\n📊 TESTING RANGE STRATEGY SIGNALS")
        print("=" * 50)
        
        # Test with range market data
        data_1m, data_15m = self.create_market_data("STRONG_RANGE", 50)
        
        # Test RSI calculation
        rsi_series = self.range_strategy.calculate_rsi(data_1m['close'])
        current_rsi = rsi_series.iloc[-1]
        self.log_test("Range RSI Calculation",
                     0 <= current_rsi <= 100,
                     f"RSI: {current_rsi:.1f}")
        
        # Test MFI calculation
        mfi_series = self.range_strategy.calculate_mfi(
            data_1m['high'], data_1m['low'], data_1m['close'], data_1m['volume']
        )
        current_mfi = mfi_series.iloc[-1]
        self.log_test("Range MFI Calculation",
                     0 <= current_mfi <= 100,
                     f"MFI: {current_mfi:.1f}")
        
        # Test signal generation in different conditions
        conditions = ["STRONG_RANGE", "WEAK_RANGE"]
        for condition in conditions:
            # Create oversold scenario
            test_data = data_1m.copy()
            # Manipulate last few rows to create oversold condition
            for i in range(5):
                idx = -(i+1)
                test_data.iloc[idx, test_data.columns.get_loc('close')] *= 0.995
            
            signal = self.range_strategy.generate_signal(test_data, condition)
            self.log_test(f"Range Signal Generation {condition}",
                         signal is None or signal.get('action') in ['BUY', 'SELL'],
                         f"Signal: {signal.get('action') if signal else 'None'}")
    
    def test_trend_strategy_signals(self):
        """Test trend strategy signal generation"""
        print("\n📈 TESTING TREND STRATEGY SIGNALS")
        print("=" * 50)
        
        # Test with trending market data
        data_1m, data_15m = self.create_market_data("TRENDING", 50)
        
        # Test RSI calculation
        rsi_series = self.trend_strategy.calculate_rsi(data_15m['close'])
        current_rsi = rsi_series.iloc[-1]
        self.log_test("Trend RSI Calculation",
                     0 <= current_rsi <= 100,
                     f"RSI: {current_rsi:.1f}")
        
        # Test MA calculation
        ma_series = self.trend_strategy.calculate_ma(data_15m['close'])
        current_ma = ma_series.iloc[-1]
        current_price = data_15m['close'].iloc[-1]
        self.log_test("Trend MA Calculation",
                     current_ma > 0,
                     f"MA: {current_ma:.2f}, Price: {current_price:.2f}")
        
        # Test trend detection
        trend = self.trend_strategy.detect_trend(data_15m['close'], ma_series)
        self.log_test("Trend Detection",
                     trend in ['UPTREND', 'DOWNTREND', 'NEUTRAL'],
                     f"Trend: {trend}")
        
        # Test signal generation
        conditions = ["TRENDING", "STRONG_TREND"]
        for condition in conditions:
            signal = self.trend_strategy.generate_signal(data_15m, condition)
            self.log_test(f"Trend Signal Generation {condition}",
                         signal is None or signal.get('action') in ['BUY', 'SELL'],
                         f"Signal: {signal.get('action') if signal else 'None'}")
    
    def test_strategy_switching(self):
        """Test strategy switching logic"""
        print("\n🔄 TESTING STRATEGY SWITCHING")
        print("=" * 50)
        
        # Test initial strategy selection
        data_1m, data_15m = self.create_market_data("STRONG_RANGE", 100)
        strategy_type, market_info = self.strategy_manager.select_strategy(data_1m, data_15m)
        
        self.log_test("Initial Strategy Selection",
                     strategy_type in ["RANGE", "TREND"],
                     f"Selected: {strategy_type}")
        
        # Test switching logic
        should_switch = self.strategy_manager.should_switch_strategy("TRENDING")
        self.log_test("Strategy Switch Detection",
                     isinstance(should_switch, bool),
                     f"Should switch: {should_switch}")
        
        # Test position sizing multiplier
        multiplier = self.strategy_manager.get_position_sizing_multiplier(strategy_type, market_info)
        self.log_test("Position Sizing Multiplier",
                     0.5 <= multiplier <= 2.0,
                     f"Multiplier: {multiplier:.2f}")
    
    def test_risk_management(self):
        """Test risk management logic"""
        print("\n⚠️ TESTING RISK MANAGEMENT")
        print("=" * 50)
        
        # Test trade validation
        valid_signal = {
            'action': 'BUY',
            'structure_stop': 3480,
            'confidence': 75
        }
        
        is_valid, reason = self.risk_manager.validate_trade(valid_signal, 50000, 3500)
        self.log_test("Valid Trade Validation",
                     is_valid,
                     f"Valid: {is_valid}, Reason: {reason}")
        
        # Test invalid signal
        invalid_signal = {
            'action': 'BUY',
            'structure_stop': 3499,  # Too close stop
            'confidence': 75
        }
        
        is_valid, reason = self.risk_manager.validate_trade(invalid_signal, 50000, 3500)
        self.log_test("Invalid Trade Validation",
                     not is_valid,
                     f"Valid: {is_valid}, Reason: {reason}")
        
        # Test position exit logic
        test_scenarios = [
            # (current_price, entry_price, side, unrealized_pnl, age_seconds, expected_close, description)
            (3520, 3500, 'Buy', 20, 30, False, "Small profit, short time"),
            (3520, 3500, 'Buy', 15, 200, True, "Target profit reached"),
            (3470, 3500, 'Buy', -25, 30, True, "Emergency stop"),
            (3510, 3500, 'Buy', 5, 4000, True, "Max hold time"),
        ]
        
        self.risk_manager.set_strategy("RANGE")
        for scenario in test_scenarios:
            current_price, entry_price, side, pnl, age, expected, desc = scenario
            should_close, reason = self.risk_manager.should_close_position(
                current_price, entry_price, side, pnl, age
            )
            
            self.log_test(f"Exit Logic: {desc}",
                         should_close == expected,
                         f"Should close: {should_close}, Reason: {reason}")
    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        print("\n🔍 TESTING EDGE CASES")
        print("=" * 50)
        
        # Test with insufficient data
        small_data = pd.DataFrame({
            'high': [3500, 3510],
            'low': [3490, 3500],
            'close': [3500, 3505],
            'volume': [1000, 1100]
        })
        
        # Test ADX with insufficient data
        adx = self.detector.calculate_adx(small_data['high'], small_data['low'], small_data['close'])
        self.log_test("ADX Insufficient Data",
                     adx == 25.0,  # Should return default
                     f"ADX: {adx}")
        
        # Test strategies with insufficient data
        range_signal = self.range_strategy.generate_signal(small_data, "STRONG_RANGE")
        self.log_test("Range Strategy Insufficient Data",
                     range_signal is None,
                     "Should return None")
        
        trend_signal = self.trend_strategy.generate_signal(small_data, "TRENDING")
        self.log_test("Trend Strategy Insufficient Data",
                     trend_signal is None,
                     "Should return None")
        
        # Test with extreme values
        extreme_data = pd.DataFrame({
            'high': [np.inf, 3510, np.nan],
            'low': [-np.inf, 3500, 3490],
            'close': [3500, np.nan, 3505],
            'volume': [0, 1100, np.inf]
        })
        
        # Should handle gracefully without crashing
        try:
            self.detector.calculate_adx(extreme_data['high'], extreme_data['low'], extreme_data['close'])
            adx_handled = True
        except:
            adx_handled = False
        
        self.log_test("ADX Extreme Values Handling",
                     adx_handled,
                     "Should handle extreme values gracefully")
    
    def run_comprehensive_test(self):
        """Run all trading condition tests"""
        print("🚀 COMPREHENSIVE TRADING CONDITIONS TEST")
        print("=" * 60)
        print(f"Testing Dual Strategy Bot - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        # Run all test categories
        self.test_fee_calculations()
        self.test_position_sizing()
        self.test_market_condition_detection()
        self.test_range_strategy_signals()
        self.test_trend_strategy_signals()
        self.test_strategy_switching()
        self.test_risk_management()
        self.test_edge_cases()
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        pass_rate = (self.passed_tests / self.total_tests) * 100 if self.total_tests > 0 else 0
        
        print(f"Total Tests: {self.total_tests}")
        print(f"Passed: {self.passed_tests}")
        print(f"Failed: {self.total_tests - self.passed_tests}")
        print(f"Pass Rate: {pass_rate:.1f}%")
        
        if pass_rate >= 95:
            print("\n🎉 EXCELLENT: All critical trading conditions working perfectly!")
        elif pass_rate >= 85:
            print("\n✅ GOOD: Most trading conditions working correctly")
        elif pass_rate >= 70:
            print("\n⚠️ WARNING: Some trading conditions need attention")
        else:
            print("\n❌ CRITICAL: Major issues detected in trading conditions")
        
        print("\n📋 DETAILED RESULTS:")
        print("-" * 60)
        for result in self.test_results:
            print(result)
        
        return pass_rate >= 95

def main():
    """Run the comprehensive test suite"""
    tester = TradingConditionsTest()
    success = tester.run_comprehensive_test()
    
    if success:
        print("\n🎯 RESULT: Bot ready for production trading!")
        return 0
    else:
        print("\n⚠️ RESULT: Review failed tests before deployment")
        return 1

if __name__ == "__main__":
    exit(main())