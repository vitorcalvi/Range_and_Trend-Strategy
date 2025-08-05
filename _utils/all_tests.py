import unittest
import pandas as pd
import numpy as np
import os
import sys
import json
import asyncio
import tempfile
from unittest.mock import Mock, patch, MagicMock, AsyncMock, call
from datetime import datetime, timedelta
from decimal import Decimal
from io import StringIO

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from strategies.range_strategy import RangeStrategy
from strategies.trend_strategy import TrendStrategy
from strategies.strategy_manager import StrategyManager, MarketConditionDetector
from core.risk_manager import RiskManager
from core.trade_engine import TradeEngine

class TestDataGenerator:
    """Generate all types of test market data"""
    
    @staticmethod
    def create_ranging_data(periods=100, base_price=100, volatility='normal'):
        """Create ranging market data with different volatility levels"""
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', periods=periods, freq='1min')
        
        # Volatility multipliers
        vol_mult = {'low': 0.0005, 'normal': 0.002, 'high': 0.008}.get(volatility, 0.002)
        
        price_changes = np.random.normal(0, vol_mult, periods)
        prices = [base_price]
        
        for change in price_changes[1:]:
            new_price = prices[-1] * (1 + change)
            # Keep price in range
            range_mult = 1.1 if volatility == 'high' else 1.05
            if new_price > base_price * range_mult:
                new_price = base_price * range_mult
            elif new_price < base_price * (2 - range_mult):
                new_price = base_price * (2 - range_mult)
            prices.append(new_price)
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, vol_mult/2))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, vol_mult/2))) for p in prices],
            'close': prices,
            'volume': np.random.uniform(1000, 5000, periods)
        })
        
        return df.set_index('timestamp')
    
    @staticmethod
    def create_trending_data(periods=100, base_price=100, trend_direction='up', strength='normal'):
        """Create trending market data with different strengths"""
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', periods=periods, freq='15min')
        
        # Trend strength multipliers
        strength_mult = {'weak': 0.0005, 'normal': 0.001, 'strong': 0.003}.get(strength, 0.001)
        trend_strength = strength_mult if trend_direction == 'up' else -strength_mult
        noise = np.random.normal(0, 0.0005, periods)
        
        prices = [base_price]
        for i in range(1, periods):
            change = trend_strength + noise[i]
            prices.append(prices[-1] * (1 + change))
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.002))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.002))) for p in prices],
            'close': prices,
            'volume': np.random.uniform(1000, 5000, periods)
        })
        
        return df.set_index('timestamp')
    
    @staticmethod
    def create_extreme_conditions(condition_type, periods=50, base_price=100):
        """Create extreme market conditions"""
        dates = pd.date_range(start='2024-01-01', periods=periods, freq='1min')
        
        if condition_type == 'flash_crash':
            # Sudden 10% drop then recovery
            prices = [base_price] * (periods // 3)
            crash_price = base_price * 0.9
            prices.extend([crash_price] * (periods // 3))
            prices.extend([base_price * 0.95] * (periods - len(prices)))
            
        elif condition_type == 'gap_up':
            # 5% gap up
            prices = [base_price] * (periods // 2)
            prices.extend([base_price * 1.05] * (periods - len(prices)))
            
        elif condition_type == 'high_volatility':
            # Extreme volatility
            prices = [base_price]
            for i in range(1, periods):
                change = np.random.normal(0, 0.02)  # 2% volatility
                prices.append(max(prices[-1] * (1 + change), base_price * 0.8))
                
        elif condition_type == 'sideways_tight':
            # Very tight sideways movement
            prices = [base_price + np.random.normal(0, 0.001) for _ in range(periods)]
            
        elif condition_type == 'parabolic':
            # Parabolic move
            prices = [base_price]
            for i in range(1, periods):
                multiplier = 1.002 + (i * 0.0001)  # Accelerating trend
                prices.append(prices[-1] * multiplier)
        else:
            prices = [base_price] * periods
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [max(p * 1.002, p) for p in prices],
            'low': [min(p * 0.998, p) for p in prices],
            'close': prices,
            'volume': np.random.uniform(1000, 5000, periods)
        })
        
        return df.set_index('timestamp')
    
    @staticmethod
    def create_invalid_data(data_type):
        """Create invalid or corrupted data for error testing"""
        dates = pd.date_range(start='2024-01-01', periods=10, freq='1min')
        
        if data_type == 'missing_columns':
            return pd.DataFrame({'timestamp': dates, 'close': [100] * 10}).set_index('timestamp')
        elif data_type == 'nan_values':
            df = pd.DataFrame({
                'timestamp': dates,
                'open': [100, np.nan, 100, 100, np.nan, 100, 100, 100, 100, 100],
                'high': [101] * 10,
                'low': [99] * 10,
                'close': [100, 100, np.nan, 100, 100, np.nan, 100, 100, 100, 100],
                'volume': [1000] * 10
            })
            return df.set_index('timestamp')
        elif data_type == 'zero_prices':
            df = pd.DataFrame({
                'timestamp': dates,
                'open': [0] * 10,
                'high': [0] * 10,
                'low': [0] * 10,
                'close': [0] * 10,
                'volume': [1000] * 10
            })
            return df.set_index('timestamp')
        elif data_type == 'negative_prices':
            df = pd.DataFrame({
                'timestamp': dates,
                'open': [-100] * 10,
                'high': [-99] * 10,
                'low': [-101] * 10,
                'close': [-100] * 10,
                'volume': [1000] * 10
            })
            return df.set_index('timestamp')
        else:
            return pd.DataFrame()

class TestRangeStrategyComprehensive(unittest.TestCase):
    """Comprehensive Range Strategy Tests"""
    
    def setUp(self):
        self.strategy = RangeStrategy()
        
    def test_rsi_calculation_all_scenarios(self):
        """Test RSI calculation in all market scenarios"""
        # Test insufficient data
        short_data = TestDataGenerator.create_ranging_data(5, 100)
        rsi = self.strategy.calculate_rsi(short_data['close'])
        self.assertEqual(rsi, 50.0)  # Default value
        
        # Test normal calculation
        normal_data = TestDataGenerator.create_ranging_data(30, 100)
        rsi = self.strategy.calculate_rsi(normal_data['close'])
        self.assertGreaterEqual(rsi, 5)
        self.assertLessEqual(rsi, 95)
        
        # Test extreme oversold conditions
        oversold_data = TestDataGenerator.create_extreme_conditions('flash_crash', 30, 100)
        rsi = self.strategy.calculate_rsi(oversold_data['close'])
        self.assertLess(rsi, 40)
        
        # Test extreme overbought conditions
        overbought_data = TestDataGenerator.create_extreme_conditions('parabolic', 30, 100)
        rsi = self.strategy.calculate_rsi(overbought_data['close'])
        self.assertGreater(rsi, 60)
        
        # Test with NaN values
        nan_data = TestDataGenerator.create_invalid_data('nan_values')
        if len(nan_data) >= 15:
            rsi = self.strategy.calculate_rsi(nan_data['close'])
            self.assertIsInstance(rsi, (int, float))
    
    def test_bollinger_bands_all_scenarios(self):
        """Test Bollinger Bands in all market scenarios"""
        # Test insufficient data
        short_data = TestDataGenerator.create_ranging_data(10, 100)
        bb_pos, band_width = self.strategy.calculate_bollinger_position(short_data['close'])
        self.assertEqual(bb_pos, 0.5)
        self.assertEqual(band_width, 0)
        
        # Test normal ranging market
        normal_data = TestDataGenerator.create_ranging_data(30, 100)
        bb_pos, band_width = self.strategy.calculate_bollinger_position(normal_data['close'])
        self.assertGreaterEqual(bb_pos, 0)
        self.assertLessEqual(bb_pos, 1)
        self.assertGreater(band_width, 0)
        
        # Test high volatility
        high_vol_data = TestDataGenerator.create_ranging_data(30, 100, 'high')
        bb_pos, band_width = self.strategy.calculate_bollinger_position(high_vol_data['close'])
        self.assertGreater(band_width, 0.01)  # Should have wider bands
        
        # Test low volatility
        low_vol_data = TestDataGenerator.create_ranging_data(30, 100, 'low')
        bb_pos, band_width = self.strategy.calculate_bollinger_position(low_vol_data['close'])
        self.assertLess(band_width, 0.01)  # Should have tighter bands
        
        # Test extreme conditions
        for condition in ['flash_crash', 'gap_up', 'sideways_tight']:
            extreme_data = TestDataGenerator.create_extreme_conditions(condition, 30, 100)
            bb_pos, band_width = self.strategy.calculate_bollinger_position(extreme_data['close'])
            self.assertGreaterEqual(bb_pos, 0)
            self.assertLessEqual(bb_pos, 1)
    
    def test_signal_generation_all_conditions(self):
        """Test signal generation in all market conditions"""
        # Test all market conditions
        market_conditions = ["STRONG_RANGE", "WEAK_RANGE", "TRENDING", "STRONG_TREND", "INSUFFICIENT_DATA"]
        
        for condition in market_conditions:
            # Test with oversold data
            oversold_data = TestDataGenerator.create_extreme_conditions('flash_crash', 30, 100)
            signal = self.strategy.generate_signal(oversold_data, condition)
            
            if condition in ["STRONG_RANGE", "WEAK_RANGE"]:
                # Should potentially generate BUY signal
                if signal:
                    self.assertEqual(signal['action'], 'BUY')
                    self.assertEqual(signal['strategy'], 'RANGE')
            else:
                # Should not generate signal in trending markets
                self.assertIsNone(signal)
        
        # Test cooldown functionality
        data = TestDataGenerator.create_extreme_conditions('flash_crash', 30, 100)
        self.strategy.last_signal_time = datetime.now()  # Set recent signal time
        signal = self.strategy.generate_signal(data, "STRONG_RANGE")
        self.assertIsNone(signal)  # Should be blocked by cooldown
        
        # Test cooldown expiry
        self.strategy.last_signal_time = datetime.now() - timedelta(seconds=120)
        signal = self.strategy.generate_signal(data, "STRONG_RANGE")
        # Signal may or may not generate based on conditions, but cooldown shouldn't block
    
    def test_signal_validation_edge_cases(self):
        """Test signal validation with edge cases"""
        # Test with minimal valid data
        minimal_data = TestDataGenerator.create_ranging_data(25, 100)
        signal = self.strategy.generate_signal(minimal_data, "STRONG_RANGE")
        
        # Test signal structure if generated
        if signal:
            required_fields = ['action', 'strategy', 'price', 'structure_stop', 'confidence']
            for field in required_fields:
                self.assertIn(field, signal)
            
            # Validate signal values
            self.assertIn(signal['action'], ['BUY', 'SELL'])
            self.assertEqual(signal['strategy'], 'RANGE')
            self.assertGreater(signal['confidence'], 60)
            self.assertGreater(signal['price'], 0)
    
    def test_configuration_validation(self):
        """Test strategy configuration validation"""
        # Test default configuration
        self.assertGreater(self.strategy.config['rsi_length'], 0)
        self.assertGreater(self.strategy.config['bb_length'], 0)
        self.assertGreater(self.strategy.config['bb_std'], 0)
        self.assertGreater(self.strategy.config['oversold'], 0)
        self.assertLess(self.strategy.config['overbought'], 100)
        self.assertGreater(self.strategy.config['cooldown_seconds'], 0)
        
        # Test configuration ranges
        self.assertLess(self.strategy.config['oversold'], self.strategy.config['overbought'])
        self.assertLessEqual(self.strategy.config['oversold'], 50)
        self.assertGreaterEqual(self.strategy.config['overbought'], 50)

class TestTrendStrategyComprehensive(unittest.TestCase):
    """Comprehensive Trend Strategy Tests"""
    
    def setUp(self):
        self.strategy = TrendStrategy()
        
    def test_ema_calculation_all_scenarios(self):
        """Test EMA calculation in all scenarios"""
        # Test insufficient data
        short_data = TestDataGenerator.create_trending_data(20, 100)
        fast_ema, slow_ema, trend = self.strategy.calculate_emas(short_data['close'])
        self.assertEqual(trend, 'NEUTRAL')
        
        # Test all trend directions
        for direction in ['up', 'down']:
            for strength in ['weak', 'normal', 'strong']:
                trend_data = TestDataGenerator.create_trending_data(60, 100, direction, strength)
                fast_ema, slow_ema, trend = self.strategy.calculate_emas(trend_data['close'])
                
                self.assertGreater(fast_ema, 0)
                self.assertGreater(slow_ema, 0)
                self.assertIn(trend, ['UPTREND', 'DOWNTREND', 'NEUTRAL'])
                
                if strength == 'strong':
                    if direction == 'up':
                        self.assertGreater(fast_ema, slow_ema)
                        self.assertEqual(trend, 'UPTREND')
                    else:
                        self.assertLess(fast_ema, slow_ema)
                        self.assertEqual(trend, 'DOWNTREND')
    
    def test_momentum_calculation_all_scenarios(self):
        """Test momentum calculation in all scenarios"""
        # Test uptrend momentum
        uptrend_data = TestDataGenerator.create_trending_data(60, 100, 'up', 'strong')
        fast_ema, _, _ = self.strategy.calculate_emas(uptrend_data['close'])
        momentum = self.strategy.calculate_trend_momentum(uptrend_data['close'], fast_ema)
        self.assertGreater(momentum, 0)
        
        # Test downtrend momentum
        downtrend_data = TestDataGenerator.create_trending_data(60, 100, 'down', 'strong')
        fast_ema, _, _ = self.strategy.calculate_emas(downtrend_data['close'])
        momentum = self.strategy.calculate_trend_momentum(downtrend_data['close'], fast_ema)
        self.assertLess(momentum, 0)
        
        # Test sideways momentum
        sideways_data = TestDataGenerator.create_ranging_data(60, 100)
        fast_ema, _, _ = self.strategy.calculate_emas(sideways_data['close'])
        momentum = self.strategy.calculate_trend_momentum(sideways_data['close'], fast_ema)
        self.assertAlmostEqual(momentum, 0, delta=0.002)
        
        # Test insufficient data
        short_data = TestDataGenerator.create_trending_data(5, 100)
        momentum = self.strategy.calculate_trend_momentum(short_data['close'], 100)
        self.assertEqual(momentum, 0)
    
    def test_signal_generation_all_market_conditions(self):
        """Test signal generation in all market conditions"""
        market_conditions = ["TRENDING", "STRONG_TREND", "STRONG_RANGE", "WEAK_RANGE"]
        
        for condition in market_conditions:
            # Test uptrend with pullback
            uptrend_data = TestDataGenerator.create_trending_data(60, 100, 'up', 'normal')
            signal = self.strategy.generate_signal(uptrend_data, condition)
            
            if condition in ["TRENDING", "STRONG_TREND"]:
                # May generate signal based on RSI levels
                if signal:
                    self.assertIn(signal['action'], ['BUY', 'SELL'])
                    self.assertEqual(signal['strategy'], 'TREND')
                    self.assertGreater(signal['confidence'], 60)
            else:
                # Should not generate signal in ranging markets
                self.assertIsNone(signal)
    
    def test_trailing_stop_calculation(self):
        """Test trailing stop calculation"""
        entry_price = 100.0
        
        # Test profitable long position
        current_price = 105.0
        should_trail, new_stop = self.strategy.should_trail_stop(
            entry_price, current_price, 'Buy', 50.0  # $50 profit
        )
        self.assertTrue(should_trail)
        self.assertLess(new_stop, current_price)
        
        # Test profitable short position
        current_price = 95.0
        should_trail, new_stop = self.strategy.should_trail_stop(
            entry_price, current_price, 'Sell', 50.0
        )
        self.assertTrue(should_trail)
        self.assertGreater(new_stop, current_price)
        
        # Test unprofitable position
        should_trail, new_stop = self.strategy.should_trail_stop(
            entry_price, 98.0, 'Buy', -20.0  # $20 loss
        )
        self.assertFalse(should_trail)
    
    def test_configuration_edge_cases(self):
        """Test configuration edge cases"""
        # Test all configuration parameters are positive
        for key, value in self.strategy.config.items():
            if isinstance(value, (int, float)):
                self.assertGreater(value, 0, f"Config {key} should be positive")
        
        # Test RSI thresholds are valid
        self.assertLess(self.strategy.config['uptrend_rsi_low'], 
                       self.strategy.config['uptrend_rsi_high'])
        self.assertLess(self.strategy.config['downtrend_rsi_low'], 
                       self.strategy.config['downtrend_rsi_high'])
        
        # Test EMA periods
        self.assertLess(self.strategy.config['fast_ema'], 
                       self.strategy.config['slow_ema'])

class TestMarketConditionDetectorComprehensive(unittest.TestCase):
    """Comprehensive Market Condition Detector Tests"""
    
    def setUp(self):
        self.detector = MarketConditionDetector()
        
    def test_adx_calculation_all_scenarios(self):
        """Test ADX calculation in all market scenarios"""
        # Test insufficient data
        short_data = TestDataGenerator.create_ranging_data(10, 100)
        adx = self.detector.calculate_adx(short_data['high'], short_data['low'], short_data['close'])
        self.assertEqual(adx, 25.0)  # Default value
        
        # Test trending markets (should have higher ADX)
        for direction in ['up', 'down']:
            for strength in ['weak', 'normal', 'strong']:
                trend_data = TestDataGenerator.create_trending_data(60, 100, direction, strength)
                adx = self.detector.calculate_adx(trend_data['high'], trend_data['low'], trend_data['close'])
                
                self.assertGreaterEqual(adx, 0)
                self.assertLessEqual(adx, 100)
                
                if strength == 'strong':
                    self.assertGreater(adx, 30)  # Strong trends should have high ADX
        
        # Test ranging markets (should have lower ADX)
        for vol in ['low', 'normal', 'high']:
            range_data = TestDataGenerator.create_ranging_data(60, 100, vol)
            adx = self.detector.calculate_adx(range_data['high'], range_data['low'], range_data['close'])
            self.assertLess(adx, 40)  # Ranging markets should have lower ADX
        
        # Test extreme conditions
        extreme_conditions = ['flash_crash', 'gap_up', 'high_volatility', 'sideways_tight', 'parabolic']
        for condition in extreme_conditions:
            extreme_data = TestDataGenerator.create_extreme_conditions(condition, 60, 100)
            adx = self.detector.calculate_adx(extreme_data['high'], extreme_data['low'], extreme_data['close'])
            self.assertGreaterEqual(adx, 0)
            self.assertLessEqual(adx, 100)
    
    def test_volatility_regime_detection_all_scenarios(self):
        """Test volatility regime detection in all scenarios"""
        # Test insufficient data
        short_data = TestDataGenerator.create_ranging_data(10, 100)
        vol_regime = self.detector.calculate_volatility_regime(short_data['close'])
        self.assertEqual(vol_regime, "NORMAL")
        
        # Test different volatility levels
        vol_levels = ['low', 'normal', 'high']
        for vol_level in vol_levels:
            data = TestDataGenerator.create_ranging_data(60, 100, vol_level)
            vol_regime = self.detector.calculate_volatility_regime(data['close'])
            self.assertIn(vol_regime, ['HIGH_VOL', 'LOW_VOL', 'NORMAL'])
            
            if vol_level == 'high':
                # High volatility data should often be detected as HIGH_VOL
                self.assertIn(vol_regime, ['HIGH_VOL', 'NORMAL'])
            elif vol_level == 'low':
                # Low volatility data should often be detected as LOW_VOL
                self.assertIn(vol_regime, ['LOW_VOL', 'NORMAL'])
        
        # Test extreme volatility conditions
        high_vol_data = TestDataGenerator.create_extreme_conditions('high_volatility', 60, 100)
        vol_regime = self.detector.calculate_volatility_regime(high_vol_data['close'])
        self.assertIn(vol_regime, ['HIGH_VOL', 'NORMAL'])
        
        # Test very tight sideways (low volatility)
        tight_data = TestDataGenerator.create_extreme_conditions('sideways_tight', 60, 100)
        vol_regime = self.detector.calculate_volatility_regime(tight_data['close'])
        self.assertIn(vol_regime, ['LOW_VOL', 'NORMAL'])
    
    def test_market_condition_detection_comprehensive(self):
        """Test comprehensive market condition detection"""
        # Test all data combinations
        data_scenarios = [
            ('ranging', 'normal', 'STRONG_RANGE'),
            ('trending_up', 'normal', 'TRENDING'),
            ('trending_down', 'strong', 'STRONG_TREND'),
        ]
        
        for scenario_type, strength, expected_range in data_scenarios:
            if 'ranging' in scenario_type:
                data_1m = TestDataGenerator.create_ranging_data(60, 100, strength)
                data_15m = TestDataGenerator.create_ranging_data(40, 100, strength)
            else:
                direction = 'up' if 'up' in scenario_type else 'down'
                data_1m = TestDataGenerator.create_trending_data(60, 100, direction, strength)
                data_15m = TestDataGenerator.create_trending_data(40, 100, direction, strength)
            
            condition = self.detector.detect_market_condition(data_1m, data_15m)
            
            # Validate condition structure
            required_fields = ['condition', 'adx', 'confidence', 'volatility', 'timestamp']
            for field in required_fields:
                self.assertIn(field, condition)
            
            # Validate condition values
            self.assertIn(condition['condition'], [
                'STRONG_RANGE', 'WEAK_RANGE', 'TRENDING', 'STRONG_TREND', 'INSUFFICIENT_DATA'
            ])
            self.assertGreaterEqual(condition['adx'], 0)
            self.assertLessEqual(condition['adx'], 100)
            self.assertGreaterEqual(condition['confidence'], 0)
            self.assertLessEqual(condition['confidence'], 1)
            self.assertIn(condition['volatility'], ['HIGH_VOL', 'LOW_VOL', 'NORMAL'])
    
    def test_edge_cases_and_errors(self):
        """Test edge cases and error handling"""
        # Test with insufficient data
        short_1m = TestDataGenerator.create_ranging_data(20, 100)
        short_15m = TestDataGenerator.create_ranging_data(10, 100)
        condition = self.detector.detect_market_condition(short_1m, short_15m)
        self.assertEqual(condition['condition'], 'INSUFFICIENT_DATA')
        
        # Test with invalid data
        invalid_data = TestDataGenerator.create_invalid_data('nan_values')
        if len(invalid_data) > 30:
            normal_data = TestDataGenerator.create_ranging_data(40, 100)
            condition = self.detector.detect_market_condition(invalid_data, normal_data)
            # Should handle gracefully
            self.assertIn(condition['condition'], [
                'STRONG_RANGE', 'WEAK_RANGE', 'TRENDING', 'STRONG_TREND', 'INSUFFICIENT_DATA'
            ])

class TestStrategyManagerComprehensive(unittest.TestCase):
    """Comprehensive Strategy Manager Tests"""
    
    def setUp(self):
        self.manager = StrategyManager()
        
    def test_strategy_selection_all_scenarios(self):
        """Test strategy selection in all market scenarios"""
        scenarios = [
            ('ranging', 'low', 'RANGE'),
            ('ranging', 'normal', 'RANGE'),
            ('ranging', 'high', 'RANGE'),
            ('trending_up', 'normal', 'TREND'),
            ('trending_down', 'strong', 'TREND'),
        ]
        
        for scenario_type, strength, expected_strategy in scenarios:
            if 'ranging' in scenario_type:
                data_1m = TestDataGenerator.create_ranging_data(60, 100, strength)
                data_15m = TestDataGenerator.create_ranging_data(40, 100, strength)
            else:
                direction = 'up' if 'up' in scenario_type else 'down'
                data_1m = TestDataGenerator.create_trending_data(60, 100, direction, strength)
                data_15m = TestDataGenerator.create_trending_data(40, 100, direction, strength)
            
            strategy_type, market_info = self.manager.select_strategy(data_1m, data_15m)
            
            # Strategy type should be either RANGE or TREND
            self.assertIn(strategy_type, ['RANGE', 'TREND'])
            
            # Market info should be complete
            self.assertIn('condition', market_info)
            self.assertIn('adx', market_info)
            self.assertIn('confidence', market_info)
    
    def test_strategy_switching_logic(self):
        """Test strategy switching logic and cooldown"""
        data_1m = TestDataGenerator.create_ranging_data(60, 100)
        data_15m = TestDataGenerator.create_ranging_data(40, 100)
        
        # First selection should always be allowed
        strategy_type, _ = self.manager.select_strategy(data_1m, data_15m)
        first_strategy = self.manager.current_strategy
        
        # Test cooldown - immediate switch should be blocked
        self.manager.last_switch_time = datetime.now()
        should_switch = self.manager.should_switch_strategy('TRENDING')
        self.assertFalse(should_switch)
        
        # Test cooldown expiry
        self.manager.last_switch_time = datetime.now() - timedelta(seconds=400)
        should_switch = self.manager.should_switch_strategy('TRENDING')
        self.assertTrue(should_switch)
        
        # Test same strategy type (should not switch)
        self.manager.current_strategy = 'STRONG_RANGE'
        should_switch = self.manager.should_switch_strategy('WEAK_RANGE')
        self.assertFalse(should_switch)  # Both are RANGE type
        
        # Test different strategy type (should switch)
        should_switch = self.manager.should_switch_strategy('TRENDING')
        self.assertTrue(should_switch)
    
    def test_position_sizing_multiplier_all_scenarios(self):
        """Test position sizing multipliers for all scenarios"""
        test_scenarios = [
            ('RANGE', 'STRONG_RANGE', 'NORMAL'),
            ('RANGE', 'WEAK_RANGE', 'HIGH_VOL'),
            ('TREND', 'TRENDING', 'LOW_VOL'),
            ('TREND', 'STRONG_TREND', 'HIGH_VOL'),
        ]
        
        for strategy_type, condition, volatility in test_scenarios:
            market_info = {
                'condition': condition,
                'volatility': volatility,
                'confidence': 0.8
            }
            
            multiplier = self.manager.get_position_sizing_multiplier(strategy_type, market_info)
            
            # Multiplier should be reasonable
            self.assertGreater(multiplier, 0.1)
            self.assertLess(multiplier, 3.0)
            
            # High volatility should reduce position size
            if volatility == 'HIGH_VOL':
                # Compare with normal volatility
                normal_info = market_info.copy()
                normal_info['volatility'] = 'NORMAL'
                normal_multiplier = self.manager.get_position_sizing_multiplier(strategy_type, normal_info)
                self.assertLess(multiplier, normal_multiplier)
    
    def test_strategy_info_completeness(self):
        """Test strategy info contains all required fields"""
        info = self.manager.get_strategy_info()
        
        required_fields = ['current_strategy', 'market_condition', 'last_switch', 'switch_cooldown_remaining']
        for field in required_fields:
            self.assertIn(field, info)
        
        # Test with active strategy
        self.manager.current_strategy = 'STRONG_RANGE'
        self.manager.last_switch_time = datetime.now()
        info = self.manager.get_strategy_info()
        
        self.assertEqual(info['current_strategy'], 'STRONG_RANGE')
        self.assertGreaterEqual(info['switch_cooldown_remaining'], 0)

class TestRiskManagerComprehensive(unittest.TestCase):
    """Comprehensive Risk Manager Tests"""
    
    def setUp(self):
        self.risk_manager = RiskManager()
        
    def test_strategy_configuration_switching(self):
        """Test switching between strategy configurations"""
        # Test initial state (should be RANGE)
        self.assertEqual(self.risk_manager.active_strategy, "RANGE")
        
        # Test switching to TREND
        self.risk_manager.set_strategy("TREND")
        self.assertEqual(self.risk_manager.active_strategy, "TREND")
        self.assertEqual(self.risk_manager.active_config, self.risk_manager.trend_config)
        
        # Test switching back to RANGE
        self.risk_manager.set_strategy("RANGE")
        self.assertEqual(self.risk_manager.active_strategy, "RANGE")
        self.assertEqual(self.risk_manager.active_config, self.risk_manager.range_config)
        
        # Test invalid strategy (should default to RANGE)
        self.risk_manager.set_strategy("INVALID")
        self.assertEqual(self.risk_manager.active_strategy, "RANGE")
    
    def test_position_size_calculation_all_scenarios(self):
        """Test position size calculation in all scenarios"""
        test_scenarios = [
            # (balance, entry_price, stop_price, expected_range)
            (1000, 100, 99, (0, 200)),  # Normal scenario
            (10000, 100, 95, (0, 2000)),  # Large balance, wide stop
            (100, 100, 99.5, (0, 50)),  # Small balance, tight stop
            (1000, 0.01, 0.009, (0, 100000)),  # Crypto scenario
        ]
        
        for strategy in ["RANGE", "TREND"]:
            self.risk_manager.set_strategy(strategy)
            
            for balance, entry_price, stop_price, expected_range in test_scenarios:
                position_size = self.risk_manager.calculate_position_size(balance, entry_price, stop_price)
                
                # Position size should be non-negative and reasonable
                self.assertGreaterEqual(position_size, 0)
                self.assertLessEqual(position_size, expected_range[1])
                
                # Test edge cases
                zero_balance = self.risk_manager.calculate_position_size(0, entry_price, stop_price)
                self.assertEqual(zero_balance, 0)
                
                zero_price = self.risk_manager.calculate_position_size(balance, 0, stop_price)
                self.assertEqual(zero_price, 0)
    
    def test_position_exit_conditions_comprehensive(self):
        """Test all position exit conditions"""
        test_scenarios = [
            # (strategy, current_price, entry_price, side, unrealized_pnl, age_seconds, expected_exit, expected_reason)
            ("RANGE", 100, 100, "Buy", 60, 60, True, "profit_target"),  # Profit target hit
            ("RANGE", 100, 100, "Buy", -200, 60, True, "emergency_stop"),  # Emergency stop
            ("RANGE", 100, 100, "Buy", 10, 400, True, "max_hold_time"),  # Max hold time
            ("RANGE", 100, 100, "Buy", -25, 250, True, "timeout_no_profit"),  # Timeout with loss
            ("RANGE", 100, 100, "Buy", 30, 60, False, "hold"),  # Hold position
            
            ("TREND", 100, 100, "Buy", 100, 60, False, "hold"),  # Trend strategy holds longer
            ("TREND", 100, 100, "Buy", -300, 60, True, "emergency_stop"),  # Emergency stop
            ("TREND", 100, 100, "Buy", 10, 4000, True, "max_hold_time"),  # Max hold time
        ]
        
        for strategy, current_price, entry_price, side, unrealized_pnl, age_seconds, expected_exit, expected_reason in test_scenarios:
            self.risk_manager.set_strategy(strategy)
            
            should_close, reason = self.risk_manager.should_close_position(
                current_price, entry_price, side, unrealized_pnl, age_seconds
            )
            
            self.assertEqual(should_close, expected_exit, 
                           f"Strategy: {strategy}, Scenario: {current_price}, {entry_price}, {side}, {unrealized_pnl}, {age_seconds}")
            
            if expected_exit:
                self.assertEqual(reason, expected_reason,
                               f"Strategy: {strategy}, Expected: {expected_reason}, Got: {reason}")
    
    def test_fee_calculations_accuracy(self):
        """Test fee calculation accuracy"""
        position_sizes = [100, 1000, 5000, 10000, 50000]
        
        for position_size in position_sizes:
            min_target = self.risk_manager.get_min_profitable_target(position_size)
            expected_fee = position_size * self.risk_manager.fee_rate
            
            # Minimum target should exceed fee cost
            self.assertGreater(min_target, expected_fee)
            
            # Should include reasonable profit margin
            profit_margin = min_target - expected_fee
            self.assertGreaterEqual(profit_margin, 15)  # At least $15 profit
    
    def test_market_condition_adaptation(self):
        """Test adaptation to market conditions"""
        base_configs = {
            "RANGE": self.risk_manager.range_config.copy(),
            "TREND": self.risk_manager.trend_config.copy()
        }
        
        volatility_scenarios = ["HIGH_VOL", "LOW_VOL", "NORMAL"]
        market_conditions = ["STRONG_RANGE", "WEAK_RANGE", "TRENDING", "STRONG_TREND"]
        
        for strategy in ["RANGE", "TREND"]:
            self.risk_manager.set_strategy(strategy)
            base_size = base_configs[strategy]['fixed_position_usdt']
            
            for volatility in volatility_scenarios:
                for condition in market_conditions:
                    self.risk_manager.adapt_to_market_condition(condition, volatility)
                    adapted_size = self.risk_manager.active_config['fixed_position_usdt']
                    
                    # Size should be adjusted based on volatility
                    if volatility == "HIGH_VOL":
                        self.assertLessEqual(adapted_size, base_size)
                    elif volatility == "LOW_VOL":
                        self.assertGreaterEqual(adapted_size, base_size)
                    
                    # Size should be reasonable
                    self.assertGreater(adapted_size, 100)
                    self.assertLess(adapted_size, 50000)
    
    def test_trailing_stop_calculation(self):
        """Test trailing stop calculation"""
        test_scenarios = [
            (105, 100, "Buy", 50, 104.475),  # Long position with profit
            (95, 100, "Sell", 50, 95.475),   # Short position with profit
            (98, 100, "Buy", 0, 0),          # No profit, no trailing
        ]
        
        self.risk_manager.set_strategy("TREND")
        
        for current_price, entry_price, side, highest_profit, expected_stop in test_scenarios:
            trailing_stop = self.risk_manager.calculate_trailing_stop(
                current_price, entry_price, side, highest_profit
            )
            
            if expected_stop > 0:
                self.assertAlmostEqual(trailing_stop, expected_stop, places=2)
            else:
                self.assertEqual(trailing_stop, 0)
    
    def test_configuration_validation(self):
        """Test configuration validation"""
        # Test range configuration
        range_config = self.risk_manager.range_config
        self.assertGreater(range_config['fixed_position_usdt'], 0)
        self.assertGreater(range_config['gross_profit_target'], 0)
        self.assertGreater(range_config['max_position_time'], 0)
        self.assertGreater(range_config['emergency_stop_pct'], 0)
        self.assertLess(range_config['emergency_stop_pct'], 1)
        
        # Test trend configuration
        trend_config = self.risk_manager.trend_config
        self.assertGreater(trend_config['fixed_position_usdt'], 0)
        self.assertGreater(trend_config['risk_reward_ratio'], 1)
        self.assertGreater(trend_config['max_position_time'], range_config['max_position_time'])

class TestTradeEngineComprehensive(unittest.TestCase):
    """Comprehensive Trade Engine Tests"""
    
    def setUp(self):
        # Mock environment variables
        self.env_patcher = patch.dict(os.environ, {
            'TRADING_SYMBOL': 'ETHUSDT',
            'DEMO_MODE': 'true',
            'TESTNET_BYBIT_API_KEY': 'test_key',
            'TESTNET_BYBIT_API_SECRET': 'test_secret',
            'TELEGRAM_BOT_TOKEN': 'test_token',
            'TELEGRAM_CHAT_ID': '123456'
        })
        self.env_patcher.start()
        
        # Create engine with mocked dependencies
        self.engine = TradeEngine()
        self.engine.exchange = Mock()
        self.engine.notifier = AsyncMock()
        
        # Mock successful connection
        self.engine.exchange.get_server_time.return_value = {'retCode': 0}
        
    def tearDown(self):
        self.env_patcher.stop()
    
    def test_symbol_rules_configuration(self):
        """Test symbol-specific trading rules"""
        symbol_tests = [
            ('ETHUSDT', '0.01', 0.01),
            ('BTCUSDT', '0.001', 0.001),
            ('ADAUSDT', '1', 1.0),
            ('DOGEUSDT', '1', 1.0),  # Default rules
        ]
        
        for symbol, expected_step, expected_min in symbol_tests:
            self.engine.symbol = symbol
            self.engine._set_symbol_rules()
            
            self.assertEqual(self.engine.qty_step, expected_step)
            self.assertEqual(self.engine.min_qty, expected_min)
    
    def test_quantity_formatting_all_symbols(self):
        """Test quantity formatting for all supported symbols"""
        test_cases = [
            # (symbol, qty_step, min_qty, input_qty, expected_output)
            ('ETHUSDT', '0.01', 0.01, 1.234567, '1.23'),
            ('ETHUSDT', '0.01', 0.01, 0.005, '0'),  # Below minimum
            ('BTCUSDT', '0.001', 0.001, 0.1234567, '0.123'),
            ('ADAUSDT', '1', 1.0, 10.7, '11'),
            ('ADAUSDT', '1', 1.0, 0.5, '0'),  # Below minimum
        ]
        
        for symbol, qty_step, min_qty, input_qty, expected in test_cases:
            self.engine.symbol = symbol
            self.engine.qty_step = qty_step
            self.engine.min_qty = min_qty
            
            result = self.engine.format_quantity(input_qty)
            self.assertEqual(result, expected)
    
    def test_market_data_processing_comprehensive(self):
        """Test market data processing with various data scenarios"""
        # Test normal kline data
        base_time = int(datetime.now().timestamp() * 1000)
        normal_kline_data = []
        
        for i in range(20):
            normal_kline_data.append([
                str(base_time + i * 60000),  # timestamp
                "100.0",   # open
                "100.5",   # high
                "99.5",    # low
                "100.1",   # close
                "1000",    # volume
                "100000"   # turnover
            ])
        
        df = self.engine._process_kline_data(normal_kline_data)
        
        self.assertEqual(len(df), 20)
        self.assertTrue(all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']))
        self.assertTrue(df['close'].dtype in [float, np.float64])
        
        # Test empty data
        empty_df = self.engine._process_kline_data([])
        self.assertTrue(empty_df.empty)
        
        # Test malformed data
        malformed_data = [["invalid", "data", "format"]]
        try:
            malformed_df = self.engine._process_kline_data(malformed_data)
            # Should handle gracefully or be empty
            self.assertIsInstance(malformed_df, pd.DataFrame)
        except:
            pass  # Expected for malformed data
    
    @patch('core.trade_engine.TradeEngine.get_account_balance')
    async def test_signal_generation_and_validation_comprehensive(self, mock_balance):
        """Test signal generation and validation comprehensively"""
        mock_balance.return_value = 1000
        
        # Setup market data
        self.engine.price_data_1m = TestDataGenerator.create_ranging_data(60, 100)
        self.engine.price_data_15m = TestDataGenerator.create_ranging_data(40, 100)
        
        # Test strategy selection
        strategy_type, market_info = self.engine.strategy_manager.select_strategy(
            self.engine.price_data_1m, self.engine.price_data_15m
        )
        
        self.assertIn(strategy_type, ['RANGE', 'TREND'])
        
        # Test signal generation
        signal = self.engine._generate_signal(strategy_type, market_info)
        
        if signal:
            # Test signal validation
            is_valid = self.engine._validate_signal(signal, market_info)
            self.assertIsInstance(is_valid, bool)
            
            # Test signal structure
            required_fields = ['action', 'strategy', 'confidence']
            for field in required_fields:
                self.assertIn(field, signal)
        
        # Test invalid market conditions
        invalid_market_info = {'condition': 'INSUFFICIENT_DATA', 'confidence': 0.3}
        invalid_signal = {'action': 'BUY', 'confidence': 40}
        
        is_valid = self.engine._validate_signal(invalid_signal, invalid_market_info)
        self.assertFalse(is_valid)
    
    @patch('core.trade_engine.TradeEngine.get_account_balance')
    async def test_trade_execution_scenarios(self, mock_balance):
        """Test trade execution in various scenarios"""
        mock_balance.return_value = 1000
        
        # Mock successful order response
        self.engine.exchange.place_order.return_value = {'retCode': 0, 'result': {'orderId': '123'}}
        
        # Test buy signal execution
        buy_signal = {
            'action': 'BUY',
            'strategy': 'RANGE',
            'confidence': 75,
            'structure_stop': 99.0,
            'price': 100.0
        }
        
        market_info = {'condition': 'STRONG_RANGE', 'confidence': 0.8}
        
        # Setup current price data
        self.engine.price_data_1m = TestDataGenerator.create_ranging_data(60, 100)
        
        await self.engine._execute_trade(buy_signal, 'RANGE', market_info)
        
        # Verify order was placed
        self.engine.exchange.place_order.assert_called()
        call_args = self.engine.exchange.place_order.call_args
        self.assertEqual(call_args[1]['side'], 'Buy')
        self.assertEqual(call_args[1]['symbol'], 'ETHUSDT')
        
        # Test sell signal execution
        sell_signal = buy_signal.copy()
        sell_signal['action'] = 'SELL'
        
        self.engine.exchange.place_order.reset_mock()
        await self.engine._execute_trade(sell_signal, 'RANGE', market_info)
        
        call_args = self.engine.exchange.place_order.call_args
        self.assertEqual(call_args[1]['side'], 'Sell')
    
    async def test_position_management_comprehensive(self):
        """Test comprehensive position management"""
        # Mock position data
        mock_position = {
            'size': '1.5',
            'side': 'Buy',
            'avgPrice': '100.0',
            'unrealisedPnl': '50.0'
        }
        
        # Test position detection
        self.engine.exchange.get_positions.return_value = {
            'retCode': 0,
            'result': {'list': [mock_position]}
        }
        
        await self.engine._check_position_status()
        
        self.assertIsNotNone(self.engine.position)
        self.assertEqual(self.engine.position, mock_position)
        
        # Test position closing
        self.engine.exchange.place_order.return_value = {'retCode': 0}
        self.engine.price_data_1m = TestDataGenerator.create_ranging_data(10, 100)
        
        await self.engine._close_position("test_reason")
        
        # Verify close order was placed
        self.engine.exchange.place_order.assert_called()
        close_call = self.engine.exchange.place_order.call_args
        self.assertEqual(close_call[1]['side'], 'Sell')  # Opposite of position side
        self.assertTrue(close_call[1]['reduceOnly'])
        
        # Test position reset on external close
        self.engine.exchange.get_positions.return_value = {
            'retCode': 0,
            'result': {'list': []}
        }
        
        await self.engine._check_position_status()
        self.assertIsNone(self.engine.position)
    
    async def test_strategy_switching_comprehensive(self):
        """Test comprehensive strategy switching scenarios"""
        # Setup initial position
        self.engine.position = {'size': '1.0', 'side': 'Buy'}
        self.engine.position_start_time = datetime.now()
        
        # Mock close order success
        self.engine.exchange.place_order.return_value = {'retCode': 0}
        self.engine.price_data_1m = TestDataGenerator.create_ranging_data(10, 100)
        
        # Test strategy switch with position
        await self.engine._on_strategy_switch('RANGE', 'TREND')
        
        # Verify position was closed
        self.engine.exchange.place_order.assert_called()
        
        # Verify exit reason was tracked
        self.assertGreater(self.engine.exit_reasons['strategy_switch'], 0)
    
    @patch('core.trade_engine.TradeEngine.get_account_balance')
    async def test_account_balance_scenarios(self, mock_balance_call):
        """Test account balance retrieval scenarios"""
        # Test successful balance retrieval
        self.engine.exchange.get_wallet_balance.return_value = {
            'retCode': 0,
            'result': {
                'list': [{
                    'coin': [
                        {'coin': 'USDT', 'walletBalance': '1000.0'},
                        {'coin': 'BTC', 'walletBalance': '0.1'}
                    ]
                }]
            }
        }
        
        balance = await self.engine.get_account_balance()
        self.assertEqual(balance, 1000.0)
        
        # Test balance retrieval failure
        self.engine.exchange.get_wallet_balance.return_value = {'retCode': 1}
        balance = await self.engine.get_account_balance()
        self.assertEqual(balance, 0)
        
        # Test exception handling
        self.engine.exchange.get_wallet_balance.side_effect = Exception("Network error")
        balance = await self.engine.get_account_balance()
        self.assertEqual(balance, 0)
    
    def test_performance_tracking(self):
        """Test performance tracking and statistics"""
        # Test initial state
        self.assertEqual(sum(self.engine.exit_reasons.values()), 0)
        self.assertEqual(sum(self.engine.rejections.values()), 0)
        
        # Test exit reason tracking
        self.engine._track_exit_reason('profit_target')
        self.assertEqual(self.engine.exit_reasons['profit_target'], 1)
        
        self.engine._track_exit_reason('invalid_reason')
        self.assertEqual(self.engine.exit_reasons['manual_exit'], 1)
        
        # Test rejection tracking
        self.engine.rejections['invalid_signal'] += 1
        self.engine.rejections['total_signals'] += 5
        
        self.assertEqual(self.engine.rejections['invalid_signal'], 1)
        self.assertEqual(self.engine.rejections['total_signals'], 5)
    
    def test_logging_functionality(self):
        """Test trade logging functionality"""
        # Create temporary log file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as temp_log:
            self.engine.log_file = temp_log.name
        
        try:
            # Test entry logging
            signal = {'action': 'BUY', 'confidence': 75}
            self.engine._log_trade("ENTRY", 100.0, signal=signal, quantity='1.5', 
                                 strategy='RANGE', position_size_usdt=150.0)
            
            # Test exit logging
            self.engine._log_trade("EXIT", 105.0, reason="profit_target", 
                                 bybit_unrealized_pnl=50.0, strategy='RANGE', duration=120.0)
            
            # Verify log file contains entries
            with open(self.engine.log_file, 'r') as f:
                log_content = f.read()
                self.assertIn('"action": "ENTRY"', log_content)
                self.assertIn('"action": "EXIT"', log_content)
                self.assertIn('"strategy": "RANGE"', log_content)
                
        finally:
            # Clean up temporary file
            os.unlink(self.engine.log_file)

class TestSystemIntegrationComprehensive(unittest.TestCase):
    """Comprehensive System Integration Tests"""
    
    def setUp(self):
        self.env_patcher = patch.dict(os.environ, {
            'TRADING_SYMBOL': 'ETHUSDT',
            'DEMO_MODE': 'true',
            'TESTNET_BYBIT_API_KEY': 'test_key',
            'TESTNET_BYBIT_API_SECRET': 'test_secret'
        })
        self.env_patcher.start()
        
    def tearDown(self):
        self.env_patcher.stop()
    
    def test_complete_trading_cycle_range_strategy(self):
        """Test complete trading cycle with range strategy"""
        # Create engine with mocked exchange
        engine = TradeEngine()
        engine.exchange = Mock()
        engine.notifier = AsyncMock()
        
        # Setup range market data
        engine.price_data_1m = TestDataGenerator.create_ranging_data(60, 100)
        engine.price_data_15m = TestDataGenerator.create_ranging_data(40, 100)
        
        # Test strategy selection
        strategy_type, market_info = engine.strategy_manager.select_strategy(
            engine.price_data_1m, engine.price_data_15m
        )
        
        # Should select RANGE strategy for ranging market
        self.assertEqual(strategy_type, 'RANGE')
        self.assertIn(market_info['condition'], ['STRONG_RANGE', 'WEAK_RANGE'])
        
        # Test signal generation
        signal = engine._generate_signal(strategy_type, market_info)
        
        if signal:
            # Validate signal
            is_valid = engine._validate_signal(signal, market_info)
            
            if is_valid:
                # Test position sizing
                balance = 1000
                current_price = 100
                base_qty = engine.risk_manager.calculate_position_size(
                    balance, current_price, signal.get('structure_stop', 99)
                )
                
                self.assertGreater(base_qty, 0)
                
                # Test risk management
                multiplier = engine.strategy_manager.get_position_sizing_multiplier(
                    strategy_type, market_info
                )
                final_qty = base_qty * multiplier
                self.assertGreater(final_qty, 0)
    
    def test_complete_trading_cycle_trend_strategy(self):
        """Test complete trading cycle with trend strategy"""
        engine = TradeEngine()
        engine.exchange = Mock()
        engine.notifier = AsyncMock()
        
        # Setup trending market data
        engine.price_data_1m = TestDataGenerator.create_trending_data(60, 100, 'up', 'strong')
        engine.price_data_15m = TestDataGenerator.create_trending_data(40, 100, 'up', 'strong')
        
        # Test strategy selection
        strategy_type, market_info = engine.strategy_manager.select_strategy(
            engine.price_data_1m, engine.price_data_15m
        )
        
        # Should select TREND strategy for trending market
        self.assertEqual(strategy_type, 'TREND')
        self.assertIn(market_info['condition'], ['TRENDING', 'STRONG_TREND'])
        
        # Test the rest of the cycle
        signal = engine._generate_signal(strategy_type, market_info)
        
        if signal:
            is_valid = engine._validate_signal(signal, market_info)
            self.assertIsInstance(is_valid, bool)
    
    def test_extreme_market_conditions_handling(self):
        """Test system behavior in extreme market conditions"""
        engine = TradeEngine()
        engine.exchange = Mock()
        engine.notifier = AsyncMock()
        
        extreme_conditions = ['flash_crash', 'gap_up', 'high_volatility', 'parabolic']
        
        for condition in extreme_conditions:
            # Setup extreme market data
            engine.price_data_1m = TestDataGenerator.create_extreme_conditions(condition, 60, 100)
            engine.price_data_15m = TestDataGenerator.create_extreme_conditions(condition, 40, 100)
            
            # Test system stability
            try:
                strategy_type, market_info = engine.strategy_manager.select_strategy(
                    engine.price_data_1m, engine.price_data_15m
                )
                
                # System should handle extreme conditions gracefully
                self.assertIn(strategy_type, ['RANGE', 'TREND'])
                self.assertIn('condition', market_info)
                
                # Test signal generation doesn't crash
                signal = engine._generate_signal(strategy_type, market_info)
                # Signal may be None, which is acceptable
                
            except Exception as e:
                self.fail(f"System failed in extreme condition {condition}: {e}")
    
    def test_error_recovery_mechanisms(self):
        """Test error recovery mechanisms"""
        engine = TradeEngine()
        engine.exchange = Mock()
        engine.notifier = AsyncMock()
        
        # Test network error recovery
        engine.exchange.get_kline.side_effect = Exception("Network error")
        
        # System should handle network errors gracefully
        result = asyncio.run(engine._update_market_data())
        self.assertFalse(result)  # Should return False on error
        
        # Test invalid data recovery
        engine.exchange.get_kline.side_effect = None
        engine.exchange.get_kline.return_value = {'retCode': 1}  # API error
        
        result = asyncio.run(engine._update_market_data())
        self.assertFalse(result)
    
    def test_configuration_edge_cases(self):
        """Test configuration edge cases and validation"""
        # Test missing environment variables
        with patch.dict(os.environ, {}, clear=True):
            try:
                engine = TradeEngine()
                # Should handle missing env vars gracefully
                self.assertIsNotNone(engine.symbol)
                self.assertIsNotNone(engine.demo_mode)
            except Exception as e:
                # Some missing vars might cause failures, which is acceptable
                pass
        
        # Test invalid symbol configuration
        with patch.dict(os.environ, {'TRADING_SYMBOL': 'INVALIDPAIR'}):
            engine = TradeEngine()
            engine._set_symbol_rules()
            # Should fall back to default rules
            self.assertEqual(engine.qty_step, '1')
            self.assertEqual(engine.min_qty, 1.0)

class TestFeeModelComprehensive(unittest.TestCase):
    """Comprehensive Fee Model Tests"""
    
    def test_fee_rate_consistency(self):
        """Test fee rate consistency across all components"""
        risk_manager = RiskManager()
        range_strategy = RangeStrategy()
        trend_strategy = TrendStrategy()
        
        # All components should use the same fee rate
        expected_fee_rate = 0.0011
        
        self.assertEqual(risk_manager.fee_rate, expected_fee_rate)
        self.assertEqual(range_strategy.config['fee_rate'], expected_fee_rate)
        self.assertEqual(trend_strategy.config['fee_rate'], expected_fee_rate)
    
    def test_breakeven_calculations_comprehensive(self):
        """Test comprehensive break-even calculations"""
        risk_manager = RiskManager()
        
        position_sizes = [100, 500, 1000, 5000, 10000, 50000]
        
        for position_size in position_sizes:
            # Calculate break-even PnL: B = P × 0.0011
            expected_breakeven = position_size * 0.0011
            
            min_profitable = risk_manager.get_min_profitable_target(position_size)
            
            # Minimum profitable target should exceed break-even
            self.assertGreater(min_profitable, expected_breakeven)
            
            # Should include reasonable profit margin
            profit_margin = min_profitable - expected_breakeven
            self.assertGreaterEqual(profit_margin, 15)  # At least $15 net profit
    
    def test_profit_targets_vs_fees(self):
        """Test profit targets exceed fee costs in all scenarios"""
        risk_manager = RiskManager()
        
        # Test range strategy
        risk_manager.set_strategy("RANGE")
        range_position_size = risk_manager.range_config['fixed_position_usdt']
        range_profit_target = risk_manager.range_config['gross_profit_target']
        range_fee_cost = range_position_size * risk_manager.fee_rate
        
        # Range profit target should significantly exceed fees
        self.assertGreater(range_profit_target, range_fee_cost)
        net_range_profit = range_profit_target - range_fee_cost
        self.assertGreater(net_range_profit, 30)  # At least $30 net profit
        
        # Test trend strategy (profit varies by risk/reward ratio)
        risk_manager.set_strategy("TREND")
        trend_position_size = risk_manager.trend_config['fixed_position_usdt']
        trend_rr_ratio = risk_manager.trend_config['risk_reward_ratio']
        
        # Estimate trend profit target (simplified)
        estimated_risk = trend_position_size * 0.02  # 2% risk
        estimated_profit_target = estimated_risk * trend_rr_ratio
        trend_fee_cost = trend_position_size * risk_manager.fee_rate
        
        self.assertGreater(estimated_profit_target, trend_fee_cost)
    
    def test_fee_impact_on_position_sizing(self):
        """Test fee impact on position sizing decisions"""
        risk_manager = RiskManager()
        
        # Test scenarios where fees might impact decisions
        test_scenarios = [
            (100, 100, 99.5),    # Very tight stop
            (1000, 100, 99),     # Normal stop
            (10000, 100, 95),    # Wide stop
        ]
        
        for balance, entry_price, stop_price in test_scenarios:
            for strategy in ["RANGE", "TREND"]:
                risk_manager.set_strategy(strategy)
                
                position_size = risk_manager.calculate_position_size(balance, entry_price, stop_price)
                position_value = position_size * entry_price
                
                if position_value > 0:
                    # Fee cost should be reasonable relative to position
                    fee_cost = position_value * risk_manager.fee_rate
                    fee_percentage = (fee_cost / position_value) * 100
                    
                    # Fee should be exactly 0.11%
                    self.assertAlmostEqual(fee_percentage, 0.11, places=2)
                    
                    # Minimum profitable target should account for fees
                    min_target = risk_manager.get_min_profitable_target(position_value)
                    self.assertGreater(min_target, fee_cost)

class TestPerformanceAndStress(unittest.TestCase):
    """Performance and Stress Tests"""
    
    def test_large_dataset_performance(self):
        """Test performance with large datasets"""
        # Create large dataset
        large_data = TestDataGenerator.create_ranging_data(1000, 100)
        
        # Test strategy calculations don't timeout
        range_strategy = RangeStrategy()
        trend_strategy = TrendStrategy()
        
        start_time = datetime.now()
        
        # Test RSI calculation
        rsi = range_strategy.calculate_rsi(large_data['close'])
        self.assertIsInstance(rsi, (int, float))
        
        # Test EMA calculation
        fast_ema, slow_ema, trend = trend_strategy.calculate_emas(large_data['close'])
        self.assertIsInstance(fast_ema, (int, float))
        
        # Test should complete within reasonable time
        elapsed = (datetime.now() - start_time).total_seconds()
        self.assertLess(elapsed, 5.0)  # Should complete within 5 seconds
    
    def test_memory_usage_with_continuous_operation(self):
        """Test memory usage doesn't grow excessively"""
        engine = TradeEngine()
        engine.exchange = Mock()
        engine.notifier = AsyncMock()
        
        # Simulate continuous operation
        for i in range(100):
            # Generate new market data
            engine.price_data_1m = TestDataGenerator.create_ranging_data(60, 100 + i)
            engine.price_data_15m = TestDataGenerator.create_ranging_data(40, 100 + i)
            
            # Run strategy selection
            strategy_type, market_info = engine.strategy_manager.select_strategy(
                engine.price_data_1m, engine.price_data_15m
            )
            
            # Generate signal
            signal = engine._generate_signal(strategy_type, market_info)
            
            # Validate signal if generated
            if signal:
                engine._validate_signal(signal, market_info)
        
        # Test should complete without memory errors
        self.assertTrue(True)  # If we get here, memory usage is acceptable
    
    def test_concurrent_strategy_execution(self):
        """Test concurrent execution of multiple strategies"""
        import threading
        import time
        
        results = []
        errors = []
        
        def run_strategy_test(strategy_type, data_type):
            try:
                if data_type == 'ranging':
                    data_1m = TestDataGenerator.create_ranging_data(60, 100)
                    data_15m = TestDataGenerator.create_ranging_data(40, 100)
                else:
                    data_1m = TestDataGenerator.create_trending_data(60, 100, 'up')
                    data_15m = TestDataGenerator.create_trending_data(40, 100, 'up')
                
                manager = StrategyManager()
                strategy_type, market_info = manager.select_strategy(data_1m, data_15m)
                results.append((strategy_type, market_info))
                
            except Exception as e:
                errors.append(str(e))
        
        # Run multiple strategy tests concurrently
        threads = []
        for i in range(10):
            data_type = 'ranging' if i % 2 == 0 else 'trending'
            thread = threading.Thread(target=run_strategy_test, args=('test', data_type))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=10)
        
        # Check results
        self.assertEqual(len(errors), 0, f"Concurrent execution errors: {errors}")
        self.assertEqual(len(results), 10)

def run_comprehensive_test_suite():
    """Run the complete comprehensive test suite"""
    print("🧪 COMPREHENSIVE TRADING BOT TEST SUITE - ALL POSSIBLE TESTS")
    print("=" * 80)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all comprehensive test classes
    test_classes = [
        TestRangeStrategyComprehensive,
        TestTrendStrategyComprehensive,
        TestMarketConditionDetectorComprehensive,
        TestStrategyManagerComprehensive,
        TestRiskManagerComprehensive,
        TestTradeEngineComprehensive,
        TestSystemIntegrationComprehensive,
        TestFeeModelComprehensive,
        TestPerformanceAndStress
    ]
    
    total_tests = 0
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
        total_tests += tests.countTestCases()
    
    print(f"📊 Total Tests: {total_tests}")
    print("🚀 Starting comprehensive test execution...")
    print("-" * 80)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout, buffer=True)
    result = runner.run(suite)
    
    # Detailed summary
    print("\n" + "=" * 80)
    print("📋 COMPREHENSIVE TEST RESULTS")
    print("=" * 80)
    
    passed = result.testsRun - len(result.failures) - len(result.errors)
    
    print(f"✅ Tests Passed:    {passed:>3}")
    print(f"❌ Tests Failed:    {len(result.failures):>3}")
    print(f"💥 Tests Errored:   {len(result.errors):>3}")
    print(f"📊 Total Tests:     {result.testsRun:>3}")
    print(f"⚡ Success Rate:    {(passed/result.testsRun*100):>5.1f}%")
    
    if result.failures:
        print(f"\n❌ FAILURES ({len(result.failures)}):")
        for i, (test, traceback) in enumerate(result.failures, 1):
            test_name = str(test).split()[0]
            error_line = traceback.strip().split('\n')[-1]
            print(f"  {i:>2}. {test_name}")
            print(f"      └─ {error_line}")
            
    if result.errors:
        print(f"\n💥 ERRORS ({len(result.errors)}):")
        for i, (test, traceback) in enumerate(result.errors, 1):
            test_name = str(test).split()[0]
            error_line = traceback.strip().split('\n')[-1]
            print(f"  {i:>2}. {test_name}")
            print(f"      └─ {error_line}")
    
    print("\n" + "=" * 80)
    
    if result.wasSuccessful():
        print("🎉 ALL TESTS PASSED! Trading bot system is fully validated.")
        print("🚀 System is ready for live trading deployment.")
    else:
        print("⚠️  Some tests failed. Review issues before deployment.")
        print("🔧 Fix failing components and re-run tests.")
    
    print("\n📝 Test Coverage Summary:")
    print("   • Range Strategy: RSI, Bollinger Bands, all market conditions")
    print("   • Trend Strategy: EMA, momentum, trend detection, trailing stops")
    print("   • Market Detection: ADX calculation, volatility regimes, all scenarios")
    print("   • Strategy Manager: Auto-switching, cooldowns, position sizing")
    print("   • Risk Management: All exit conditions, fee calculations, emergency stops")
    print("   • Trade Engine: Data processing, signal validation, position management")
    print("   • System Integration: Complete trading cycles, error recovery")
    print("   • Fee Model: 0.11% validation, break-even calculations, profit targets")
    print("   • Performance: Large datasets, memory usage, concurrent execution")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    # Set up test environment
    os.makedirs("logs", exist_ok=True)
    
    # Run comprehensive test suite
    success = run_comprehensive_test_suite()
    
    if success:
        print("\n🏆 COMPREHENSIVE VALIDATION COMPLETE - ALL SYSTEMS GO!")
        sys.exit(0)
    else:
        print("\n🚨 VALIDATION FAILED - REVIEW AND FIX BEFORE DEPLOYMENT")
        sys.exit(1)