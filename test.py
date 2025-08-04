#!/usr/bin/env python3
"""
Simple Market Switching Test
Tests the core market switching functionality between Range and Trend strategies.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add the project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_market_data(market_type="range", periods=200, base_price=2000.0):
    """Create test market data for different conditions - FIXED VERSION"""
    np.random.seed(42)  # Reproducible results
    
    timestamps = pd.date_range(start='2024-01-01', periods=periods, freq='1min')
    
    # Use much smaller changes to prevent overflow
    if market_type == "range":
        # Sideways market - low ADX, bounded between 1900-2100
        prices = [base_price]
        for i in range(1, periods):
            # Strong mean reversion to keep in range
            distance_from_center = prices[-1] - base_price
            reversion = -distance_from_center * 0.001  # Pull back to center
            noise = np.random.normal(0, 0.5)  # Small random noise
            
            # Calculate new price with bounds
            new_price = prices[-1] + reversion + noise
            new_price = max(1900.0, min(2100.0, new_price))  # Hard bounds
            prices.append(new_price)
    
    elif market_type == "trend":
        # Trending market - high ADX, gradual uptrend
        prices = [base_price]
        trend_strength = 0.2  # Small consistent trend
        for i in range(1, periods):
            # Consistent upward bias with some noise
            trend_component = trend_strength
            noise = np.random.normal(0, 0.3)
            
            new_price = prices[-1] + trend_component + noise
            new_price = max(1000.0, min(5000.0, new_price))  # Reasonable bounds
            prices.append(new_price)
    
    else:  # volatile
        # High volatility but bounded
        prices = [base_price]
        for i in range(1, periods):
            # Higher volatility but controlled
            change = np.random.normal(0, 2.0)  # Larger noise
            new_price = prices[-1] + change
            new_price = max(1500.0, min(2500.0, new_price))  # Keep bounded
            prices.append(new_price)
    
    # Validate all prices are finite
    prices = np.array(prices)
    if not np.all(np.isfinite(prices)):
        print("❌ Invalid prices generated, using fallback")
        prices = np.linspace(base_price * 0.95, base_price * 1.05, periods)
    
    # Create OHLCV data with small spread
    data = []
    for i, price in enumerate(prices):
        spread = abs(np.random.normal(0, 0.1))  # Very small spread
        high = price + spread
        low = price - spread
        volume = np.random.uniform(1000, 10000)
        
        data.append({
            'open': prices[i-1] if i > 0 else price,
            'high': high,
            'low': low,
            'close': price,
            'volume': volume
        })
    
    df = pd.DataFrame(data, index=timestamps)
    
    # Final validation - ensure no NaN/infinity values
    for col in ['open', 'high', 'low', 'close']:
        if not np.all(np.isfinite(df[col])):
            print(f"❌ Invalid values in {col}, fixing...")
            df[col] = df[col].fillna(base_price)
            df[col] = np.where(np.isinf(df[col]), base_price, df[col])
    
    return df

def test_market_switching():
    """Test the market switching functionality"""
    print("🧪 Testing Market Switching Functionality")
    print("=" * 60)
    
    try:
        # Import our strategies
        from strategies.strategy_manager import StrategyManager
        from core.risk_manager import RiskManager
        
        # Initialize components
        strategy_manager = StrategyManager()
        risk_manager = RiskManager()
        
        print("✅ Components loaded successfully")
        
        # Test 1: Range Market Detection
        print("\n📊 Test 1: Range Market Detection")
        range_data_1m = create_market_data("range", 200)
        range_data_15m = create_market_data("range", 100)
        
        strategy_type, market_info = strategy_manager.select_strategy(range_data_1m, range_data_15m)
        
        print(f"   Market Condition: {market_info['condition']}")
        print(f"   ADX Value: {market_info['adx']:.2f}")
        print(f"   Selected Strategy: {strategy_type}")
        print(f"   Confidence: {market_info['confidence']*100:.0f}%")
        
        # Validate range detection
        expected_range = market_info['condition'] in ['STRONG_RANGE', 'WEAK_RANGE']
        expected_strategy = strategy_type == 'RANGE'
        
        if expected_range and expected_strategy:
            print("   ✅ Range market correctly detected")
        else:
            print("   ❌ Range market detection failed")
            return False
        
        # Test 2: Trend Market Detection
        print("\n📈 Test 2: Trend Market Detection")
        
        # Force cooldown to expire for strategy switching
        strategy_manager.last_switch_time = datetime.now() - pd.Timedelta(seconds=301)
        
        trend_data_1m = create_market_data("trend", 200)
        trend_data_15m = create_market_data("trend", 100)
        
        strategy_type, market_info = strategy_manager.select_strategy(trend_data_1m, trend_data_15m)
        
        print(f"   Market Condition: {market_info['condition']}")
        print(f"   ADX Value: {market_info['adx']:.2f}")
        print(f"   Selected Strategy: {strategy_type}")
        print(f"   Confidence: {market_info['confidence']*100:.0f}%")
        
        # Validate trend detection
        expected_trend = market_info['condition'] in ['TRENDING', 'STRONG_TREND']
        expected_strategy = strategy_type == 'TREND'
        
        if expected_trend and expected_strategy:
            print("   ✅ Trend market correctly detected")
        else:
            print("   ❌ Trend market detection failed")
            return False
        
        # Test 3: Risk Manager Synchronization
        print("\n🛡️  Test 3: Risk Manager Synchronization")
        
        # Test RANGE strategy config
        risk_manager.set_strategy("RANGE")
        range_config = risk_manager.get_active_config()
        
        print(f"   Range Strategy Config: {range_config['strategy']}")
        print(f"   Position Size: ${range_config['config']['fixed_position_usdt']:,}")
        print(f"   Max Hold Time: {range_config['config']['max_position_time']}s")
        
        # Test TREND strategy config
        risk_manager.set_strategy("TREND")
        trend_config = risk_manager.get_active_config()
        
        print(f"   Trend Strategy Config: {trend_config['strategy']}")
        print(f"   Position Size: ${trend_config['config']['fixed_position_usdt']:,}")
        print(f"   Max Hold Time: {trend_config['config']['max_position_time']}s")
        
        # Validate different configurations
        range_time = range_config['config']['max_position_time']
        trend_time = trend_config['config']['max_position_time']
        
        if range_time != trend_time:
            print("   ✅ Risk manager correctly switches configurations")
        else:
            print("   ❌ Risk manager configuration switching failed")
            return False
        
        # Test 4: Position Sizing Multipliers
        print("\n💰 Test 4: Position Sizing")
        
        test_cases = [
            ("STRONG_RANGE", "NORMAL", "RANGE"),
            ("TRENDING", "HIGH_VOL", "TREND"),
        ]
        
        for condition, volatility, expected_strategy in test_cases:
            market_info_test = {
                'condition': condition,
                'volatility': volatility,
                'adx': 30,
                'confidence': 0.8
            }
            
            multiplier = strategy_manager.get_position_sizing_multiplier(expected_strategy, market_info_test)
            print(f"   {condition} + {volatility}: {multiplier:.2f}x multiplier")
            
            # Validate reasonable multiplier
            if not (0.5 <= multiplier <= 2.0):
                print("   ❌ Invalid position sizing multiplier")
                return False
        
        print("   ✅ Position sizing multipliers working correctly")
        
        # Test 5: Strategy Switch Cooldown
        print("\n⏰ Test 5: Switch Cooldown")
        
        # Reset to range market
        strategy_manager.last_switch_time = None
        strategy_type1, _ = strategy_manager.select_strategy(range_data_1m, range_data_15m)
        
        # Immediate attempt to switch (should be blocked)
        strategy_type2, _ = strategy_manager.select_strategy(trend_data_1m, trend_data_15m)
        
        strategy_info = strategy_manager.get_strategy_info()
        cooldown_remaining = strategy_info['switch_cooldown_remaining']
        
        print(f"   Strategy 1: {strategy_type1}")
        print(f"   Strategy 2: {strategy_type2}")
        print(f"   Cooldown Remaining: {cooldown_remaining:.1f}s")
        
        if cooldown_remaining > 0:
            print("   ✅ Switch cooldown working correctly")
        else:
            print("   ❌ Switch cooldown not working")
            return False
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("✅ Market switching functionality is working correctly")
        return True
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("💡 Make sure all streamlined files are in the correct locations")
        return False
    except Exception as e:
        print(f"❌ Test Error: {e}")
        return False

def main():
    """Main test runner"""
    print("🚀 Simple Market Switch Test")
    print("Testing core dual strategy switching functionality...\n")
    
    success = test_market_switching()
    
    if success:
        print("\n🎯 Test Result: SUCCESS")
        print("Your dual strategy system is ready to use!")
    else:
        print("\n💥 Test Result: FAILED")
        print("Please check the streamlined files and try again.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)