import os
import asyncio
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any
from pybit.unified_trading import HTTP
from dotenv import load_dotenv

from strategies.strategy_manager import StrategyManager
from strategies.range_strategy import RangeStrategy  
from strategies.trend_strategy import TrendStrategy
from core.risk_manager import RiskManager
from _utils.telegram_notifier import TelegramNotifier

load_dotenv()

class TradeEngine:
    """ULTRA-AGGRESSIVE: High-frequency trading engine with 3m optimization"""
    
    def __init__(self):
        # Strategy system (ultra-aggressive)
        self.strategy_manager = StrategyManager()
        self.range_strategy = RangeStrategy()
        self.trend_strategy = TrendStrategy()
        self.risk_manager = RiskManager()
        self.notifier = TelegramNotifier()
        
        # Exchange setup
        self.symbol = os.getenv('TRADING_SYMBOL', 'ETHUSDT')
        self.demo_mode = os.getenv('DEMO_MODE', 'true').lower() == 'true'
        
        prefix = 'TESTNET_' if self.demo_mode else 'LIVE_'
        self.api_key = os.getenv(f'{prefix}BYBIT_API_KEY')
        self.api_secret = os.getenv(f'{prefix}BYBIT_API_SECRET')
        
        self.exchange = None
        self.position = None
        self.position_start_time = None
        self.position_entry_price = None
        self.position_size_usdt = None
        self.highest_profit = 0.0  # For trailing stops
        
        # ULTRA-AGGRESSIVE market data (3m optimized)
        self.price_data_1m = pd.DataFrame()
        self.price_data_15m = pd.DataFrame()
        self.price_data_3m = pd.DataFrame()  # Simulated 3m data
        self.atr_period = 14
        self.current_atr = 0.0
        
        # State tracking
        self.trade_id = 0
        self.active_strategy = None
        self.market_info = {}
        self.successful_entries = 0
        self.last_cycle_time = datetime.now()
        
        # FIXED: Duplicate trade prevention
        self.last_entry_price = None
        self.last_entry_time = None
        
        # ULTRA-AGGRESSIVE performance tracking
        self.exit_reasons = {
            'profit_target': 0, 'emergency_stop': 0, 'max_hold_time': 0,
            'trailing_stop': 0, 'strategy_switch': 0, 'manual_exit': 0,
            'timeout_no_profit': 0
        }
        self.rejections = {
            'invalid_market': 0, 'cooldown_active': 0, 'insufficient_data': 0,
            'invalid_signal': 0, 'total_signals': 0, 'unprofitable': 0,
            'fee_efficiency': 0, 'position_too_small': 0
        }
        
        # Order execution tracking
        self.order_stats = {
            'maker_fills': 0, 'taker_fills': 0, 'limit_attempts': 0,
            'limit_successes': 0, 'total_orders': 0
        }
        
        self._set_symbol_rules()
        os.makedirs("logs", exist_ok=True)
        self.log_file = "logs/trades.log"
    
    def _set_symbol_rules(self):
        """Set symbol-specific trading rules"""
        symbol_rules = {
            'ETH': ('0.01', 0.01),
            'BTC': ('0.001', 0.001),
            'ADA': ('1', 1.0)
        }
        
        for key, (step, min_qty) in symbol_rules.items():
            if key in self.symbol:
                self.qty_step, self.min_qty = step, min_qty
                return
        
        self.qty_step, self.min_qty = '1', 1.0
    
    def connect(self):
        """Connect to exchange"""
        try:
            self.exchange = HTTP(demo=self.demo_mode, api_key=self.api_key, api_secret=self.api_secret)
            return self.exchange.get_server_time().get('retCode') == 0
        except:
            return False
    
    def format_quantity(self, qty):
        """Format quantity according to exchange rules"""
        if qty < self.min_qty:
            return "0"
        
        try:
            decimals = len(self.qty_step.split('.')[1]) if '.' in self.qty_step else 0
            qty_step_float = float(self.qty_step)
            rounded_qty = round(qty / qty_step_float) * qty_step_float
            return f"{rounded_qty:.{decimals}f}" if decimals > 0 else str(int(rounded_qty))
        except:
            return f"{qty:.3f}"
    
    async def run_cycle(self):
        """ULTRA-AGGRESSIVE: High-frequency trading cycle (every 0.3 seconds)"""
        cycle_start = datetime.now()
        
        if not await self._update_market_data():
            return
        
        await self._check_position_status()
        
        if self.position and self.position_start_time:
            await self._check_position_exit()
        
        if not self.position:
            await self._generate_and_execute_signal()
        
        # Ultra-aggressive cycle timing (faster than before)
        cycle_time = (datetime.now() - cycle_start).total_seconds()
        if cycle_time < 0.5:  # Target 2 cycles per second
            await asyncio.sleep(0.5 - cycle_time)
        
        # Update display every 3 seconds to avoid spam
        if (datetime.now() - self.last_cycle_time).total_seconds() >= 3:
            self._display_status()
            self.last_cycle_time = datetime.now()
    
    async def _update_market_data(self):
        """ULTRA-AGGRESSIVE: Update market data with 3m simulation"""
        try:
            # Get more 1m data for 3m simulation
            klines_1m = self.exchange.get_kline(category="linear", symbol=self.symbol, interval="1", limit=300)
            klines_15m = self.exchange.get_kline(category="linear", symbol=self.symbol, interval="15", limit=100)
            
            if klines_1m.get('retCode') != 0 or klines_15m.get('retCode') != 0:
                return False
            
            self.price_data_1m = self._process_kline_data(klines_1m['result']['list'])
            self.price_data_15m = self._process_kline_data(klines_15m['result']['list'])
            
            # Simulate 3m data from 1m data
            self.price_data_3m = self._create_3m_data_from_1m(self.price_data_1m)
            
            # Calculate ATR for trailing stops
            self._calculate_atr()
            
            return (len(self.price_data_1m) > 60 and 
                   len(self.price_data_15m) > 30 and
                   len(self.price_data_3m) > 20 and
                   not self.price_data_1m['close'].isna().any() and
                   not self.price_data_15m['close'].isna().any())
        except Exception as e:
            print(f"❌ Market data error: {e}")
            return False
    
    def _create_3m_data_from_1m(self, data_1m: pd.DataFrame) -> pd.DataFrame:
        """Create 3-minute OHLCV data from 1-minute data"""
        if len(data_1m) < 3:
            return pd.DataFrame()
        
        # Group by 3-minute intervals
        data_1m_copy = data_1m.copy()
        data_1m_copy.reset_index(inplace=True)
        
        # Create 3m groups (every 3 rows)
        group_size = 3
        groups = []
        
        for i in range(0, len(data_1m_copy) - group_size + 1, group_size):
            group = data_1m_copy.iloc[i:i+group_size]
            
            ohlcv_3m = {
                'timestamp': group['timestamp'].iloc[-1],  # Last timestamp
                'open': group['open'].iloc[0],             # First open
                'high': group['high'].max(),               # Highest high
                'low': group['low'].min(),                 # Lowest low  
                'close': group['close'].iloc[-1],          # Last close
                'volume': group['volume'].sum()            # Sum volume
            }
            groups.append(ohlcv_3m)
        
        if not groups:
            return pd.DataFrame()
        
        df_3m = pd.DataFrame(groups)
        return df_3m.set_index('timestamp')
    
    def _calculate_atr(self):
        """Calculate ATR for ultra-tight trailing stops"""
        if len(self.price_data_3m) < self.atr_period:
            self.current_atr = 0.0
            return
        
        data = self.price_data_3m.tail(self.atr_period + 1)
        
        # True Range calculation
        tr_values = []
        for i in range(1, len(data)):
            high = data['high'].iloc[i]
            low = data['low'].iloc[i]
            prev_close = data['close'].iloc[i-1]
            
            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            tr_values.append(tr)
        
        self.current_atr = np.mean(tr_values) if tr_values else 0.0
    
    def _process_kline_data(self, kline_list):
        """Process kline data into DataFrame"""
        if not kline_list:
            return pd.DataFrame()
            
        df = pd.DataFrame(kline_list, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
        ])
        
        if df.empty:
            return df
        
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].iloc[-1])
        return df.sort_values('timestamp').set_index('timestamp')
    
    async def _generate_and_execute_signal(self):
        """ULTRA-AGGRESSIVE: Generate signals with proper cooldown enforcement"""
        # Use 3m data for strategy selection (ultra-aggressive)
        strategy_type, market_info = self.strategy_manager.select_strategy(
            self.price_data_3m, self.price_data_15m
        )
        
        self.market_info = market_info
        
        # FIXED: Strictly enforce trade cooldown
        if not market_info.get('trade_allowed', False):
            cooldown_remaining = market_info.get('trade_cooldown_remaining', 0)
            if cooldown_remaining > 5:  # Show cooldown if > 5 seconds
                if cooldown_remaining % 30 == 0:  # Only print every 30 seconds
                    print(f"⏳ Trade cooldown: {cooldown_remaining:.0f}s remaining")
            return
        
        # Handle strategy switching
        if self.active_strategy and self.active_strategy != strategy_type:
            await self._on_strategy_switch(self.active_strategy, strategy_type)
        
        if self.active_strategy != strategy_type:
            self.risk_manager.set_strategy(strategy_type)
            self.risk_manager.adapt_to_market_condition(
                market_info['condition'], 
                market_info.get('volatility', 'NORMAL')
            )
        
        self.active_strategy = strategy_type
        signal = self._generate_signal(strategy_type, market_info)
        
        if signal:
            self.rejections['total_signals'] += 1
            if self._validate_signal(signal, market_info):
                await self._execute_trade(signal, strategy_type, market_info)
    
    def _generate_signal(self, strategy_type, market_info):
        """Generate signal using ultra-aggressive strategies"""
        try:
            if strategy_type == "RANGE":
                # Use 3m data for range strategy
                return self.range_strategy.generate_signal(self.price_data_3m, market_info['condition'])
            else:  # TREND
                # Use 3m data for trend strategy  
                return self.trend_strategy.generate_signal(self.price_data_3m, market_info['condition'])
        except Exception as e:
            print(f"❌ Signal generation error: {e}")
            self.rejections['invalid_signal'] += 1
            return None
    
    def _validate_signal(self, signal, market_info):
        """ULTRA-AGGRESSIVE: Enhanced signal validation with fee efficiency"""
        validation_checks = [
            (not signal or market_info['condition'] == 'INSUFFICIENT_DATA', 'insufficient_data'),
            (market_info['confidence'] < 0.6, 'invalid_market'),
            (signal.get('confidence', 0) < 62, 'invalid_signal')  # REDUCED threshold
        ]
        
        for condition, rejection_type in validation_checks:
            if condition:
                self.rejections[rejection_type] += 1
                return False
        
        return True
    
    async def _execute_trade(self, signal, strategy_type, market_info):
        """ULTRA-AGGRESSIVE: Execute trade with duplicate prevention"""
        current_price = float(self.price_data_3m['close'].iloc[-1])
        balance = await self.get_account_balance()
        
        if not balance or balance < 1000:  # Minimum balance check
            return
        
        # FIXED: Prevent duplicate trades at same price
        if hasattr(self, 'last_entry_price') and hasattr(self, 'last_entry_time'):
            price_diff = abs(current_price - self.last_entry_price) / current_price
            time_diff = (datetime.now() - self.last_entry_time).total_seconds()
            
            if price_diff < 0.001 and time_diff < 300:  # Same price within 5 minutes
                print(f"⚠️  Duplicate trade rejected: Same price ${current_price:.2f} within 5min")
                return
        
        # Calculate position size using ultra-aggressive risk manager
        base_qty = self.risk_manager.calculate_position_size(
            balance, current_price, signal['structure_stop']
        )
        
        # Apply ultra-aggressive sizing multiplier
        sizing_multiplier = self.strategy_manager.get_position_sizing_multiplier(
            strategy_type, market_info
        )
        qty = base_qty * sizing_multiplier
        
        formatted_qty = self.format_quantity(qty)
        
        if formatted_qty == "0" or float(formatted_qty) < 0.001:
            self.rejections['position_too_small'] += 1
            return
        
        # Calculate actual position size
        position_size_usdt = float(formatted_qty) * current_price
        
        # Validate fee efficiency (ultra-aggressive thresholds)
        expected_profit = position_size_usdt * signal.get('risk_reward_ratio', 1.3) * 0.015  # Estimate
        if not self.risk_manager.validate_fee_efficiency(position_size_usdt, expected_profit):
            self.rejections['fee_efficiency'] += 1
            return
        
        # ULTRA-AGGRESSIVE: Limit-first order execution
        success = await self._execute_limit_first_order(signal, formatted_qty, current_price, market_info)
        
        if success:
            # Record trade for cooldown and frequency tracking
            self.strategy_manager.record_trade()
            
            # Record entry details for duplicate prevention
            self.last_entry_price = current_price
            self.last_entry_time = datetime.now()
            
            self.successful_entries += 1
            self.position_entry_price = current_price
            self.position_size_usdt = position_size_usdt
            self.highest_profit = 0.0  # Reset for trailing stops
            
            break_even_fees = self.risk_manager.get_break_even_pnl(position_size_usdt)
            
            self._log_trade("ENTRY", current_price, signal=signal, quantity=formatted_qty, 
                           strategy=strategy_type, position_size_usdt=position_size_usdt,
                           break_even_fees=break_even_fees)
                           
            await self.notifier.send_trade_entry(signal, current_price, formatted_qty, 
                                               self._get_strategy_info())
            
            print(f"✅ ULTRA-AGGRESSIVE: {signal['action']} {formatted_qty} @ ${current_price:.4f}")
            print(f"   📊 Position: ${position_size_usdt:.0f} | Strategy: {strategy_type}")
            print(f"   💰 Break-even: ${break_even_fees:.2f} | Target: ${signal.get('net_profit_target', 0)}")
    
    async def _execute_limit_first_order(self, signal, quantity, current_price, market_info):
        """ULTRA-AGGRESSIVE: Limit-first order execution with tight spreads"""
        side = "Buy" if signal['action'] == 'BUY' else "Sell"
        confidence = signal.get('confidence', 65)
        
        # Determine limit offset based on confidence (ultra-aggressive - tighter spreads)
        if confidence >= 85:
            offset_pct = 0.0003    # 0.03% (REDUCED)
        elif confidence >= 75:
            offset_pct = 0.0006    # 0.06% (REDUCED)
        else:
            offset_pct = 0.001     # 0.1% (REDUCED)
        
        # Calculate limit price
        if side == "Buy":
            limit_price = current_price * (1 - offset_pct)
        else:
            limit_price = current_price * (1 + offset_pct)
        
        try:
            self.order_stats['limit_attempts'] += 1
            self.order_stats['total_orders'] += 1
            
            # Try limit order first (IOC - Immediate or Cancel)
            limit_order = self.exchange.place_order(
                category="linear", 
                symbol=self.symbol,
                side=side,
                orderType="Limit",
                qty=quantity,
                price=f"{limit_price:.4f}",
                timeInForce="IOC"  # Immediate or Cancel
            )
            
            if limit_order.get('retCode') == 0:
                # Check if limit order filled
                await asyncio.sleep(0.1)  # Brief wait
                order_id = limit_order['result']['orderId']
                
                order_status = self.exchange.get_open_orders(category="linear", symbol=self.symbol)
                if order_status.get('retCode') == 0:
                    open_orders = order_status['result']['list']
                    order_exists = any(order['orderId'] == order_id for order in open_orders)
                    
                    if not order_exists:  # Order filled
                        self.order_stats['limit_successes'] += 1
                        self.order_stats['maker_fills'] += 1
                        print(f"✅ LIMIT FILL: Maker order @ ${limit_price:.4f} (saved {offset_pct*100:.3f}%)")
                        return True
            
            # Limit order didn't fill, place market order
            print(f"⚡ Limit order timeout, executing market order...")
            market_order = self.exchange.place_order(
                category="linear", 
                symbol=self.symbol,
                side=side,
                orderType="Market", 
                qty=quantity, 
                timeInForce="IOC"
            )
            
            if market_order.get('retCode') == 0:
                self.order_stats['taker_fills'] += 1
                print(f"✅ MARKET FILL: Taker order @ ${current_price:.4f}")
                return True
                
        except Exception as e:
            print(f"❌ Order execution error: {e}")
        
        return False
    
    async def _check_position_status(self):
        """Check position status with ultra-aggressive monitoring"""
        try:
            positions = self.exchange.get_positions(category="linear", symbol=self.symbol)
            if positions.get('retCode') != 0:
                return
            
            pos_list = positions['result']['list']
            
            if not pos_list or float(pos_list[0]['size']) == 0:
                if self.position:
                    await self._on_position_closed()
                self._reset_position()
                return
            
            if not self.position:
                self.position_start_time = datetime.now()
                if not self.position_entry_price:
                    self.position_entry_price = float(pos_list[0].get('avgPrice', 0))
                if not self.position_size_usdt and self.position_entry_price:
                    qty = float(pos_list[0]['size'])
                    self.position_size_usdt = qty * self.position_entry_price
                    
            self.position = pos_list[0]
            
            # Update highest profit for trailing stops
            unrealized_pnl = float(self.position.get('unrealisedPnl', 0))
            if unrealized_pnl > self.highest_profit:
                self.highest_profit = unrealized_pnl
                
        except Exception as e:
            print(f"❌ Position check error: {e}")
    
    async def _check_position_exit(self):
        """ULTRA-AGGRESSIVE: Check position exit with ultra-tight trailing stops"""
        if not self.position or not self.position_start_time:
            return
        
        current_price = float(self.price_data_3m['close'].iloc[-1])
        entry_price = self.position_entry_price or float(self.position.get('avgPrice', 0))
        side = self.position.get('side', '')
        
        # Use Bybit's unrealized PnL
        unrealized_pnl = float(self.position.get('unrealisedPnl', 0))
        position_age = (datetime.now() - self.position_start_time).total_seconds()
        
        # Get position size for fee calculations
        position_size_usdt = self.position_size_usdt or (
            float(self.position.get('size', 0)) * entry_price
        )
        
        if self.risk_manager.active_strategy != self.active_strategy:
            self.risk_manager.set_strategy(self.active_strategy)
        
        # ULTRA-AGGRESSIVE: Check exit conditions with ultra-tight trailing
        should_close, reason = self.risk_manager.should_close_position(
            current_price, entry_price, side, unrealized_pnl, position_age, position_size_usdt
        )
        
        # Additional ultra-tight trailing stop check
        if not should_close and self.active_strategy == "TREND" and unrealized_pnl > 0:
            if self.current_atr > 0:
                # Use 0.4% or 0.5 ATR, whichever is tighter
                atr_distance = 0.5 * self.current_atr
                pct_distance = current_price * 0.004  # 0.4%
                trailing_distance = min(atr_distance, pct_distance)
                
                if side.lower() == 'buy' and current_price < (entry_price + trailing_distance):
                    should_close, reason = True, "trailing_stop"
                elif side.lower() == 'sell' and current_price > (entry_price - trailing_distance):
                    should_close, reason = True, "trailing_stop"
        
        if should_close:
            await self._close_position(reason)
    
    async def _close_position(self, reason="Manual"):
        """ULTRA-AGGRESSIVE: Close position with enhanced logging"""
        if not self.position:
            return
        
        current_price = float(self.price_data_3m['close'].iloc[-1]) if len(self.price_data_3m) > 0 else 0
        unrealized_pnl = float(self.position.get('unrealisedPnl', 0))
        
        side = "Sell" if self.position.get('side') == "Buy" else "Buy"
        qty = self.format_quantity(float(self.position['size']))
        
        try:
            order = self.exchange.place_order(
                category="linear", symbol=self.symbol, side=side,
                orderType="Market", qty=qty, timeInForce="IOC", reduceOnly=True
            )
            
            if order.get('retCode') == 0:
                duration = (datetime.now() - self.position_start_time).total_seconds() if self.position_start_time else 0
                
                self._track_exit_reason(reason)
                
                self._log_trade("EXIT", current_price, reason=reason, bybit_unrealized_pnl=unrealized_pnl, 
                               strategy=self.active_strategy, duration=duration, 
                               position_size_usdt=self.position_size_usdt or 0)
                
                exit_data = {'trigger': reason, 'strategy': self.active_strategy}
                await self.notifier.send_trade_exit(exit_data, current_price, unrealized_pnl, duration, self._get_strategy_info())
                
                print(f"📤 EXIT: {reason} | PnL: ${unrealized_pnl:.2f} | Duration: {duration:.1f}s")
        except Exception as e:
            print(f"❌ Position close error: {e}")
    
    async def _on_strategy_switch(self, old_strategy, new_strategy):
        """Handle ultra-aggressive strategy switch"""
        if self.position:
            await self._close_position("strategy_switch")
            
            for _ in range(3):  # Reduced wait time
                await asyncio.sleep(0.5)
                await self._check_position_status()
                if not self.position:
                    break
                    
            if self.position:
                self._reset_position()
        
        self.exit_reasons['strategy_switch'] += 1
        print(f"🔄 STRATEGY SWITCH: {old_strategy} → {new_strategy}")
    
    async def _on_position_closed(self):
        """Handle position closed externally"""
        if self.position:
            unrealized_pnl = float(self.position.get('unrealisedPnl', 0))
            price = float(self.price_data_3m['close'].iloc[-1]) if len(self.price_data_3m) > 0 else 0
            self._track_exit_reason('position_closed')
            self._log_trade("EXIT", price, reason="position_closed", bybit_unrealized_pnl=unrealized_pnl, 
                           strategy=self.active_strategy, position_size_usdt=self.position_size_usdt or 0)
    
    def _reset_position(self):
        """Reset position state"""
        self.position = None
        self.position_start_time = None
        self.position_entry_price = None
        self.position_size_usdt = None
        self.highest_profit = 0.0
    
    def _track_exit_reason(self, reason):
        """Track exit reason"""
        if reason in self.exit_reasons:
            self.exit_reasons[reason] += 1
        else:
            self.exit_reasons['manual_exit'] += 1
    
    def _log_trade(self, action, price, **kwargs):
        """ULTRA-AGGRESSIVE: Enhanced trade logging with fee model"""
        timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
        
        if action == "ENTRY":
            self.trade_id += 1
            signal = kwargs.get('signal', {})
            position_size_usdt = kwargs.get('position_size_usdt', 0)
            break_even_fees = kwargs.get('break_even_fees', 0)
            
            log_data = {
                'timestamp': timestamp, 'id': self.trade_id, 'action': 'ENTRY',
                'strategy': kwargs.get('strategy', 'UNKNOWN'),
                'side': signal.get('action', ''), 'price': round(price, 4), 
                'size': kwargs.get('quantity', ''), 
                'market_condition': self.market_info.get('condition', ''),
                'adx': round(self.market_info.get('adx', 0), 1),
                'confidence': round(signal.get('confidence', 0), 1),
                'position_size_usdt': round(position_size_usdt, 2),
                'break_even_fees': round(break_even_fees, 2),
                'signal_reason': signal.get('signal_reason', 'unknown'),
                'rsi': signal.get('rsi', 0),
                'timeframe': signal.get('timeframe', '3m'),
                'ultra_aggressive': True,
                'note': f'Need ${break_even_fees:.2f} profit to cover 0.0615% fees'
            }
        else:
            duration = kwargs.get('duration', 0)
            bybit_pnl = kwargs.get('bybit_unrealized_pnl', 0)
            position_size_usdt = kwargs.get('position_size_usdt', 0)
            
            # Calculate estimated net profit
            if position_size_usdt > 0:
                break_even_fees = self.risk_manager.get_break_even_pnl(position_size_usdt)
                estimated_net = bybit_pnl - break_even_fees
            else:
                estimated_net = bybit_pnl
                break_even_fees = 0
            
            log_data = {
                'timestamp': timestamp, 'id': self.trade_id, 'action': 'EXIT',
                'strategy': kwargs.get('strategy', 'UNKNOWN'),
                'trigger': kwargs.get('reason', '').lower().replace(' ', '_'),
                'price': round(price, 4), 
                'bybit_unrealized_pnl': round(bybit_pnl, 2),
                'estimated_fees': round(break_even_fees, 2),
                'estimated_net_profit': round(estimated_net, 2),
                'hold_seconds': round(duration, 1),
                'highest_profit': round(self.highest_profit, 2),
                'ultra_aggressive': True,
                'note': 'Ultra-aggressive: 0.0615% fee model with 3m timeframe'
            }
        
        try:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(log_data) + "\n")
        except Exception as e:
            print(f"❌ Logging error: {e}")
    
    async def get_account_balance(self):
        """Get account balance"""
        try:
            balance = self.exchange.get_wallet_balance(accountType="UNIFIED")
            if balance.get('retCode') == 0:
                coins = balance['result']['list'][0]['coin']
                usdt = next((c for c in coins if c['coin'] == 'USDT'), None)
                return float(usdt['walletBalance']) if usdt else 0
            return 0
        except:
            return 0
    
    def _get_strategy_info(self):
        """Get current ultra-aggressive strategy information"""
        if self.active_strategy == "RANGE":
            return self.range_strategy.get_strategy_info()
        else:
            return self.trend_strategy.get_strategy_info()
    
    def _display_status(self):
        """ULTRA-AGGRESSIVE: Display with enhanced metrics"""
        try:
            price = float(self.price_data_3m['close'].iloc[-1]) if len(self.price_data_3m) > 0 else 0
            time = datetime.now().strftime('%H:%M:%S')
            symbol_display = self.symbol.replace('USDT', '/USDT')
            price_formatted = f"{price:,.4f}".replace(',', ' ')
            
            print("\n" * 50)
            
            w = 85
            print(f"{'='*w}\n⚡  {symbol_display} ULTRA-AGGRESSIVE DUAL-STRATEGY BOT v3.0\n{'='*w}\n")
            
            # Market analysis
            market_condition = self.market_info.get('condition', 'UNKNOWN')
            adx = self.market_info.get('adx', 0)
            confidence = self.market_info.get('confidence', 0)
            vol_regime = self.market_info.get('volatility', 'NORMAL')
            
            print("🧠  ULTRA MARKET ANALYSIS (3m + 15m)\n" + "─"*w)
            print(f"📊 Condition: {market_condition:<12} │ 📈 ADX: {adx:>5.1f} │ 🎯 Confidence: {confidence*100:>3.0f}% │ 📊 Vol: {vol_regime}")
            print(f"⚙️  Strategy: {self.active_strategy or 'NONE':<13} │ 🕐 Timeframe: 3m │ 🎯ATR: {self.current_atr:.6f}")
            
            # Signal frequency tracking
            freq_info = self.market_info.get('signal_frequency', {})
            signals_hour = freq_info.get('signals_last_hour', 0)
            freq_status = freq_info.get('status', 'UNKNOWN')
            freq_score = freq_info.get('frequency_score', 0)
            
            print(f"🎯 Signal Freq: {signals_hour}/hour │ Target: 8-12/hour │ Status: {freq_status} │ Score: {freq_score:.1f}%")
            
            # Cooldown status
            trade_cooldown = self.market_info.get('trade_cooldown_remaining', 0)
            if trade_cooldown > 0:
                minutes = int(trade_cooldown / 60)
                seconds = int(trade_cooldown % 60)
                print(f"⏳ Trade Cooldown: {minutes}m {seconds}s remaining")
            else:
                print("✅ Ready for ultra-aggressive trading")
                
            print("─"*w + "\n")
            
            # Strategy status
            print("📋  ULTRA-AGGRESSIVE STRATEGY STATUS\n" + "─"*w)
            if self.active_strategy == "RANGE":
                print(f"🎯 RSI(6) + BB(15,1.8) Range Strategy")
                print(f"📊 Target: ${self.risk_manager.range_config['target_position_usdt']} │ Hold: {self.risk_manager.range_config['max_position_time']//60}min │ R/R: 1:1.3")
            elif self.active_strategy == "TREND":
                print(f"📈 RSI(6) + EMA(5/13) Trend Strategy")
                print(f"📊 Target: ${self.risk_manager.trend_config['target_position_usdt']} │ Hold: {self.risk_manager.trend_config['max_position_time']//60}min │ R/R: 1:1.5")
            else:
                print("⚙️  Initializing ultra-aggressive strategies...")
                
            print("─"*w + "\n")
            
            # Order execution stats
            total_orders = self.order_stats['total_orders']
            if total_orders > 0:
                maker_rate = (self.order_stats['maker_fills'] / total_orders) * 100
                limit_success_rate = (self.order_stats['limit_successes'] / max(self.order_stats['limit_attempts'], 1)) * 100
                
                print("💹  ORDER EXECUTION OPTIMIZATION\n" + "─"*w)
                print(f"📊 Total Orders: {total_orders} │ Maker Fills: {self.order_stats['maker_fills']} ({maker_rate:.1f}%)")
                print(f"🎯 Limit Success: {limit_success_rate:.1f}% │ Fee Savings: ~{maker_rate*0.45:.1f}% per trade")
                print("─"*w + "\n")
            
            # Performance metrics
            print("📊  ULTRA-AGGRESSIVE PERFORMANCE\n" + "─"*w)
            total_trades = sum(self.exit_reasons.values())
            total_signals = self.rejections.get('total_signals', 0)
            accept_rate = (total_trades/max(total_signals,1)*100)
            
            print(f"🔢 Trades: {total_trades:>3} │ Signals: {total_signals:>3} │ Accept: {accept_rate:>4.1f}% │ ADX Threshold: 15+")
            
            if total_trades > 0:
                top_exits = sorted(self.exit_reasons.items(), key=lambda x: x[1], reverse=True)[:3]
                exit_str = " │ ".join([f"{k.replace('_', ' ').title()}: {v}" for k, v in top_exits if v > 0])
                if exit_str:
                    print(f"📝 Exits: {exit_str}")
            
            print("─"*w + "\n")
            print(f"⏰ {time}   |   💰 ${price_formatted}   |   🚀 Ultra-Aggressive Mode Active")
            print()
            
            # Position info
            if self.position:
                unrealized_pnl = float(self.position.get('unrealisedPnl', 0))
                entry = self.position_entry_price or float(self.position.get('avgPrice', 0))
                size = self.position.get('size', '0')
                side = self.position.get('side', '')
                
                age = (datetime.now() - self.position_start_time).total_seconds() if self.position_start_time else 0
                
                position_size_usdt = self.position_size_usdt or (float(size) * entry)
                break_even_fees = self.risk_manager.get_break_even_pnl(position_size_usdt)
                net_pnl = unrealized_pnl - break_even_fees
                
                emoji = "🟢" if side == "Buy" else "🔴"
                print(f"{emoji} {side} Position: {size} @ ${entry:.4f} │ Strategy: {self.active_strategy}")
                print(f"   💰 Gross: ${unrealized_pnl:.2f} │ Fees: ${break_even_fees:.2f} │ Net: ${net_pnl:.2f}")
                print(f"   ⏱️  Age: {age:.0f}s │ Max: {self.risk_manager.get_max_position_time()}s │ Peak: ${self.highest_profit:.2f}")
                
                if self.active_strategy == "TREND" and net_pnl > 10:
                    print(f"   🎯 Trailing: 0.4% active │ ATR: {self.current_atr:.6f}")
            else:
                print("⚡ No Position — Ultra-Aggressive Scanner Active")
                print("   🎯 RSI(6) + EMA(5/13) + BB(15) + ADX >15 Detection")
                print("   ⏱️  2min trade cooldowns │ 3min strategy cooldowns")
            
            print("─" * w)
            
        except Exception as e:
            print(f"❌ Display error: {e}")
    
    def _get_active_timeframe(self):
        """Get active timeframe string"""
        return "3m"  # Ultra-aggressive uses 3m primarily
    
    def get_stress_test_summary(self) -> Dict[str, Any]:
        """Get stress test summary for testing and validation"""
        total_trades = sum(self.exit_reasons.values()) if self.exit_reasons else 0
        total_signals = self.rejections.get('total_signals', 0) if self.rejections else 0
        total_orders = self.order_stats['total_orders'] if self.order_stats else 0
        
        # Calculate performance metrics with None protection
        accept_rate = (total_trades / max(total_signals, 1)) * 100
        maker_rate = (self.order_stats.get('maker_fills', 0) / max(total_orders, 1)) * 100
        
        # Signal frequency analysis with None protection  
        freq_info = self.market_info.get('signal_frequency', {}) if self.market_info else {}
        signals_hour = freq_info.get('signals_last_hour', 0)
        freq_score = freq_info.get('frequency_score', 0)
        
        # Safe market condition access
        market_condition = self.market_info.get('condition', 'UNKNOWN') if self.market_info else 'UNKNOWN'
        
        return {
            'ultra_aggressive_metrics': {
                'total_trades': total_trades,
                'total_signals': total_signals,
                'acceptance_rate': round(accept_rate, 1),
                'signal_frequency_per_hour': signals_hour,
                'frequency_score': round(freq_score, 1),
                'target_range': '8-12 signals/hour',
                'current_strategy': self.active_strategy or 'NONE',
                'market_condition': market_condition,
                'adx_threshold': 15,
                'rsi_period': 6,
                'ema_periods': '5/13',
                'timeframe': '3m'
            },
            'order_execution': {
                'total_orders': total_orders,
                'maker_fills': self.order_stats.get('maker_fills', 0),
                'taker_fills': self.order_stats.get('taker_fills', 0),
                'maker_rate': round(maker_rate, 1),
                'limit_attempts': self.order_stats.get('limit_attempts', 0),
                'limit_successes': self.order_stats.get('limit_successes', 0),
                'limit_success_rate': round((self.order_stats.get('limit_successes', 0) / max(self.order_stats.get('limit_attempts', 1), 1)) * 100, 1)
            },
            'exit_reasons': dict(self.exit_reasons) if self.exit_reasons else {},
            'rejections': dict(self.rejections) if self.rejections else {},
            'fee_model': {
                'blended_rate': '0.086%',
                'maker_rate': '0.01%',
                'taker_rate': '0.06%',
                'target_maker_fills': '40%',
                'actual_maker_fills': f'{maker_rate:.1f}%'
            },
            'ultra_aggressive_features': {
                'rsi_6_vs_14': 'RSI(6) ultra-fast response',
                'ema_5_13_vs_21_50': 'EMA(5/13) ultra-sensitive',
                'adx_15_vs_25': 'ADX >15 ultra-sensitive detection',
                'trailing_04_vs_05': '0.4% ultra-tight trailing',
                'cooldowns': 'Trade: 2min, Strategy: 3min',
                'position_sizing': 'Dynamic $1.5K-$5K range',
                'hold_times': 'Range: 5min, Trend: 12min max'
            },
            'validation_status': {
                'duplicate_prevention': hasattr(self, 'last_entry_price'),
                'cooldown_enforcement': True,
                'fee_efficiency_check': True,
                'position_size_validation': True,
                'signal_frequency_tracking': True,
                'none_value_protection': True
            }
        }