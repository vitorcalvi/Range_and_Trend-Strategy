import os
import asyncio
import pandas as pd
import numpy as np
import json
from datetime import datetime
from decimal import Decimal
from typing import Dict
from pybit.unified_trading import HTTP
from dotenv import load_dotenv

from strategies.strategy_manager import StrategyManager
from strategies.range_strategy import RangeStrategy  
from strategies.trend_strategy import TrendStrategy
from core.risk_manager import RiskManager
from _utils.telegram_notifier import TelegramNotifier

load_dotenv()

class TradeEngine:
    def __init__(self):
        # Strategy system
        self.strategy_manager = StrategyManager()
        self.range_strategy = RangeStrategy()
        self.trend_strategy = TrendStrategy()
        self.risk_manager = RiskManager()
        self.notifier = TelegramNotifier()
        
        # Exchange setup
        self.symbol = os.getenv('TRADING_SYMBOL', 'ADAUSDT')
        self.demo_mode = os.getenv('DEMO_MODE', 'true').lower() == 'true'
        
        prefix = 'TESTNET_' if self.demo_mode else 'LIVE_'
        self.api_key = os.getenv(f'{prefix}BYBIT_API_KEY')
        self.api_secret = os.getenv(f'{prefix}BYBIT_API_SECRET')
        
        self.exchange = None
        self.position = None
        self.position_start_time = None
        self.position_entry_price = None
        self.position_size_usdt = None
        self.highest_profit = 0  # Track highest profit for trailing stops
        
        # Market data
        self.price_data_1m = pd.DataFrame()
        self.price_data_15m = pd.DataFrame()
        self.current_atr = 0  # For ATR-based trailing stops
        
        # State tracking
        self.trade_id = 0
        self.active_strategy = None
        self.market_info = {}
        self.successful_entries = 0
        
        # Performance tracking
        self.exit_reasons = {
            'profit_target': 0, 'emergency_stop': 0, 'max_hold_time': 0,
            'trailing_stop': 0, 'strategy_switch': 0, 'manual_exit': 0,
            'timeout_no_profit': 0, 'stop_loss': 0, 'profit_lock': 0
        }
        self.rejections = {
            'invalid_market': 0, 'cooldown_active': 0, 'insufficient_data': 0,
            'invalid_signal': 0, 'total_signals': 0, 'unprofitable': 0,
            'fee_inefficient': 0
        }
        
        # Order execution tracking
        self.order_stats = {
            'limit_fills': 0, 'market_fills': 0, 'partial_fills': 0,
            'total_maker_savings': 0.0
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
    
    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range for trailing stops"""
        if len(data) < period + 1:
            return 0
        
        high = data['high']
        low = data['low'] 
        close = data['close']
        
        # Calculate True Range
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(period).mean().iloc[-1]
        
        return float(atr) if not pd.isna(atr) else 0
    
    async def run_cycle(self):
        """Run one trading cycle"""
        if not await self._update_market_data():
            return
        
        # Update ATR for trailing stops
        if len(self.price_data_15m) > 15:
            self.current_atr = self.calculate_atr(self.price_data_15m)
        
        await self._check_position_status()
        
        if self.position and self.position_start_time:
            await self._check_position_exit()
        
        if not self.position:
            await self._generate_and_execute_signal()
        
        self._display_status()
    
    async def _update_market_data(self):
        """Update both 1m and 15m market data"""
        try:
            klines_1m = self.exchange.get_kline(category="linear", symbol=self.symbol, interval="1", limit=200)
            klines_15m = self.exchange.get_kline(category="linear", symbol=self.symbol, interval="15", limit=100)
            
            if klines_1m.get('retCode') != 0 or klines_15m.get('retCode') != 0:
                return False
            
            self.price_data_1m = self._process_kline_data(klines_1m['result']['list'])
            self.price_data_15m = self._process_kline_data(klines_15m['result']['list'])
            
            return (len(self.price_data_1m) > 50 and 
                   len(self.price_data_15m) > 30 and
                   not self.price_data_1m['close'].isna().any() and
                   not self.price_data_15m['close'].isna().any())
        except:
            return False
    
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
        """Generate signals with dynamic position sizing"""
        strategy_type, market_info = self.strategy_manager.select_strategy(
            self.price_data_1m, self.price_data_15m
        )
        
        self.market_info = market_info
        
        # Check trade cooldown
        if not market_info.get('trade_allowed', False):
            cooldown_remaining = market_info.get('trade_cooldown_remaining', 0)
            if cooldown_remaining > 60:
                minutes = int(cooldown_remaining / 60)
                print(f"⏳ Trade cooldown: {minutes}m remaining")
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
                await self._execute_trade_with_limit_first(signal, strategy_type, market_info)
    
    def _generate_signal(self, strategy_type, market_info):
        """Generate signal using the appropriate strategy"""
        try:
            if strategy_type == "RANGE":
                return self.range_strategy.generate_signal(self.price_data_1m, market_info['condition'])
            else:  # TREND
                return self.trend_strategy.generate_signal(self.price_data_15m, market_info['condition'])
        except:
            self.rejections['invalid_signal'] += 1
            return None
    
    def _validate_signal(self, signal, market_info):
        """Enhanced signal validation with fee efficiency check"""
        validation_checks = [
            (not signal or market_info['condition'] == 'INSUFFICIENT_DATA', 'insufficient_data'),
            (market_info['confidence'] < 0.6, 'invalid_market'),
            (signal.get('confidence', 0) < 70, 'invalid_signal')  # Higher threshold
        ]
        
        for condition, rejection_type in validation_checks:
            if condition:
                self.rejections[rejection_type] += 1
                return False
        
        return True
    
    async def _execute_trade_with_limit_first(self, signal, strategy_type, market_info):
        """LIMIT-FIRST execution: Try limit order first, fallback to market"""
        current_price = float(self.price_data_1m['close'].iloc[-1])
        balance = await self.get_account_balance()
        
        if not balance or not self._validate_signal(signal, market_info):
            return
        
        # Dynamic position sizing
        expected_hold_minutes = 10 if strategy_type == "RANGE" else 30
        base_qty = self.risk_manager.calculate_dynamic_position_size(
            balance, current_price, signal['structure_stop'], expected_hold_minutes
        )
        
        # Apply sizing multiplier
        sizing_multiplier = self.strategy_manager.get_position_sizing_multiplier(
            strategy_type, market_info
        )
        qty = base_qty * sizing_multiplier
        
        formatted_qty = self.format_quantity(qty)
        
        if formatted_qty == "0" or float(formatted_qty) < 0.001:
            print(f"❌ Position too small: {formatted_qty}")
            return
        
        position_size_usdt = float(formatted_qty) * current_price
        
        # Fee efficiency validation
        fee_analysis = self.risk_manager.get_fee_analysis(position_size_usdt)
        fee_pct = float(fee_analysis['efficiency_metrics']['fee_percentage'].rstrip('%'))
        
        if fee_pct > 5.0:  # Reject if fees > 5% of expected gross
            self.rejections['fee_inefficient'] += 1
            print(f"❌ Fee inefficient: {fee_pct:.1f}% of expected gross")
            return
        
        # Get order strategy (limit-first with IOC fallback)
        order_strategy = self.risk_manager.get_order_strategy(
            signal['action'], current_price, signal.get('confidence', 75)
        )
        
        side = "Buy" if signal['action'] == 'BUY' else "Sell"
        
        # Execute limit-first strategy
        execution_result = await self._execute_limit_first_order(
            side, formatted_qty, order_strategy, current_price
        )
        
        if execution_result['success']:
            # Record trade execution
            self.strategy_manager.record_trade()
            self.successful_entries += 1
            self.position_entry_price = execution_result['fill_price']
            self.position_size_usdt = position_size_usdt
            self.highest_profit = 0  # Reset for new position
            
            # Update order statistics
            if execution_result['order_type'] == 'limit':
                self.order_stats['limit_fills'] += 1
                savings = position_size_usdt * (self.risk_manager.taker_fee_rate - self.risk_manager.maker_fee_rate)
                self.order_stats['total_maker_savings'] += float(savings)
            else:
                self.order_stats['market_fills'] += 1
            
            self._log_trade("ENTRY", execution_result['fill_price'], signal=signal, 
                           quantity=formatted_qty, strategy=strategy_type, 
                           position_size_usdt=position_size_usdt,
                           order_type=execution_result['order_type'],
                           fee_analysis=fee_analysis)
                           
            await self.notifier.send_trade_entry(signal, execution_result['fill_price'], 
                                               formatted_qty, self._get_strategy_info())
            
            print(f"✅ {execution_result['order_type'].upper()} order filled: {signal['action']} {formatted_qty} @ ${execution_result['fill_price']:.2f}")
            print(f"   Position: ${position_size_usdt:.2f} | Fee efficiency: {fee_pct:.1f}%")
    
    async def _execute_limit_first_order(self, side: str, quantity: str, 
                                       order_strategy: Dict, current_price: float) -> Dict:
        """Execute limit-first order strategy with IOC fallback"""
        
        # Step 1: Try limit order first
        limit_price = order_strategy['primary_order']['price']
        
        try:
            print(f"🎯 Trying LIMIT order: {side} {quantity} @ ${limit_price:.2f}")
            
            limit_order = self.exchange.place_order(
                category="linear",
                symbol=self.symbol,
                side=side,
                orderType="Limit",
                qty=quantity,
                price=f"{limit_price:.2f}",
                timeInForce="IOC"  # Immediate-or-Cancel
            )
            
            if limit_order.get('retCode') == 0:
                # Wait briefly to check if filled
                await asyncio.sleep(0.5)
                
                order_id = limit_order['result']['orderId']
                order_status = self.exchange.get_open_orders(
                    category="linear", symbol=self.symbol, orderId=order_id
                )
                
                if order_status.get('retCode') == 0:
                    orders = order_status['result']['list']
                    
                    if not orders:  # Order filled completely
                        # Get fill details
                        fills = self.exchange.get_executions(
                            category="linear", symbol=self.symbol, orderId=order_id
                        )
                        
                        if fills.get('retCode') == 0 and fills['result']['list']:
                            fill_price = float(fills['result']['list'][0]['execPrice'])
                            return {
                                'success': True,
                                'order_type': 'limit',
                                'fill_price': fill_price,
                                'method': 'limit_ioc_full_fill'
                            }
                    
                    else:  # Order still open or partially filled
                        # Cancel remaining quantity
                        self.exchange.cancel_order(
                            category="linear", symbol=self.symbol, orderId=order_id
                        )
                        
                        # Check for partial fills
                        fills = self.exchange.get_executions(
                            category="linear", symbol=self.symbol, orderId=order_id
                        )
                        
                        if fills.get('retCode') == 0 and fills['result']['list']:
                            # Had partial fill, record it
                            self.order_stats['partial_fills'] += 1
                            print(f"📊 Partial limit fill detected, proceeding to market order")
        
        except Exception as e:
            print(f"⚠️ Limit order failed: {e}")
        
        # Step 2: Fallback to market order
        print(f"🏃 Fallback to MARKET order: {side} {quantity}")
        
        try:
            market_order = self.exchange.place_order(
                category="linear",
                symbol=self.symbol,
                side=side,
                orderType="Market",
                qty=quantity,
                timeInForce="IOC"
            )
            
            if market_order.get('retCode') == 0:
                return {
                    'success': True,
                    'order_type': 'market',
                    'fill_price': current_price,  # Approximate
                    'method': 'market_fallback'
                }
                
        except Exception as e:
            print(f"❌ Market order failed: {e}")
        
        return {'success': False, 'order_type': 'failed', 'fill_price': 0}
    
    async def _check_position_status(self):
        """Check position status and update highest profit"""
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
            current_pnl = float(self.position.get('unrealisedPnl', 0))
            if current_pnl > self.highest_profit:
                self.highest_profit = current_pnl
                
        except:
            pass
    
    async def _check_position_exit(self):
        """Enhanced position exit checking with trailing stops"""
        if not self.position or not self.position_start_time:
            return
        
        current_price = float(self.price_data_1m['close'].iloc[-1])
        entry_price = self.position_entry_price or float(self.position.get('avgPrice', 0))
        side = self.position.get('side', '')
        
        unrealized_pnl = float(self.position.get('unrealisedPnl', 0))
        position_age = (datetime.now() - self.position_start_time).total_seconds()
        
        position_size_usdt = self.position_size_usdt or (
            float(self.position.get('size', 0)) * entry_price
        )
        
        if self.risk_manager.active_strategy != self.active_strategy:
            self.risk_manager.set_strategy(self.active_strategy)
        
        # Enhanced exit check with trailing stops
        should_close, reason = self.risk_manager.should_close_position(
            current_price, entry_price, side, unrealized_pnl, position_age, 
            position_size_usdt, self.highest_profit, self.current_atr
        )
        
        if should_close:
            await self._close_position(reason)
    
    async def _close_position(self, reason="Manual"):
        """Close position with market order"""
        if not self.position:
            return
        
        current_price = float(self.price_data_1m['close'].iloc[-1]) if len(self.price_data_1m) > 0 else 0
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
                
                self._log_trade("EXIT", current_price, reason=reason, 
                               bybit_unrealized_pnl=unrealized_pnl, 
                               strategy=self.active_strategy, duration=duration, 
                               position_size_usdt=self.position_size_usdt or 0,
                               highest_profit=self.highest_profit)
                
                exit_data = {'trigger': reason, 'strategy': self.active_strategy}
                await self.notifier.send_trade_exit(exit_data, current_price, 
                                                  unrealized_pnl, duration, self._get_strategy_info())
        except Exception as e:
            print(f"❌ Position close error: {e}")
    
    async def _on_strategy_switch(self, old_strategy, new_strategy):
        """Handle strategy switch"""
        if self.position:
            await self._close_position("strategy_switch")
            
            for _ in range(5):
                await asyncio.sleep(1)
                await self._check_position_status()
                if not self.position:
                    break
                    
            if self.position:
                self._reset_position()
        
        self.exit_reasons['strategy_switch'] += 1
    
    async def _on_position_closed(self):
        """Handle position closed externally"""
        if self.position:
            unrealized_pnl = float(self.position.get('unrealisedPnl', 0))
            price = float(self.price_data_1m['close'].iloc[-1]) if len(self.price_data_1m) > 0 else 0
            self._track_exit_reason('position_closed')
            self._log_trade("EXIT", price, reason="position_closed", 
                           bybit_unrealized_pnl=unrealized_pnl, 
                           strategy=self.active_strategy, 
                           position_size_usdt=self.position_size_usdt or 0)
    
    def _reset_position(self):
        """Reset position state"""
        self.position = None
        self.position_start_time = None
        self.position_entry_price = None
        self.position_size_usdt = None
        self.highest_profit = 0
    
    def _track_exit_reason(self, reason):
        """Track exit reason"""
        if reason in self.exit_reasons:
            self.exit_reasons[reason] += 1
        else:
            self.exit_reasons['manual_exit'] += 1
    
    def _log_trade(self, action, price, **kwargs):
        """Enhanced trade logging with order execution details"""
        timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
        
        if action == "ENTRY":
            self.trade_id += 1
            signal = kwargs.get('signal', {})
            position_size_usdt = kwargs.get('position_size_usdt', 0)
            order_type = kwargs.get('order_type', 'market')
            fee_analysis = kwargs.get('fee_analysis', {})
            
            log_data = {
                'timestamp': timestamp, 'id': self.trade_id, 'action': 'ENTRY',
                'strategy': kwargs.get('strategy', 'UNKNOWN'),
                'side': signal.get('action', ''), 'price': round(price, 2), 
                'size': kwargs.get('quantity', ''), 
                'order_type': order_type,
                'market_condition': self.market_info.get('condition', ''),
                'adx': round(self.market_info.get('adx', 0), 1),
                'confidence': round(signal.get('confidence', 0), 1),
                'position_size_usdt': round(position_size_usdt, 2),
                'fee_efficiency': fee_analysis.get('efficiency_metrics', {}),
                'breakeven_rate': self.risk_manager.get_current_breakeven_rate() * 100
            }
        else:
            duration = kwargs.get('duration', 0)
            bybit_pnl = kwargs.get('bybit_unrealized_pnl', 0)
            position_size_usdt = kwargs.get('position_size_usdt', 0)
            highest_profit = kwargs.get('highest_profit', 0)
            
            # Calculate fees based on actual order execution
            if position_size_usdt > 0:
                fee_cost = position_size_usdt * float(self.risk_manager.total_cost_rate)
                estimated_net = bybit_pnl - fee_cost
            else:
                estimated_net = bybit_pnl
                fee_cost = 0
            
            log_data = {
                'timestamp': timestamp, 'id': self.trade_id, 'action': 'EXIT',
                'strategy': kwargs.get('strategy', 'UNKNOWN'),
                'trigger': kwargs.get('reason', '').lower().replace(' ', '_'),
                'price': round(price, 2), 
                'bybit_unrealized_pnl': round(bybit_pnl, 2),
                'estimated_fees': round(fee_cost, 2),
                'estimated_net_profit': round(estimated_net, 2),
                'highest_profit': round(highest_profit, 2),
                'hold_seconds': round(duration, 1),
                'breakeven_rate_required': f"{self.risk_manager.get_current_breakeven_rate()*100:.1f}%"
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
        """Get current strategy information"""
        return (self.range_strategy.get_strategy_info() if self.active_strategy == "RANGE" 
                else self.trend_strategy.get_strategy_info())
    
    def _display_status(self):
        """Enhanced display with fee optimization metrics"""
        try:
            price = float(self.price_data_1m['close'].iloc[-1])
            time = self.price_data_1m.index[-1].strftime('%H:%M:%S')
            symbol_display = self.symbol.replace('USDT', '/USDT')
            price_formatted = f"{price:,.2f}".replace(',', ' ')
            
            print("\n" * 50)
            
            w = 85
            print(f"{'='*w}\n⚡  {symbol_display} OPTIMIZED DUAL-STRATEGY BOT (Fee & Slippage Optimized)\n{'='*w}\n")
            
            market_condition = self.market_info.get('condition', 'UNKNOWN')
            adx = self.market_info.get('adx', 0)
            confidence = self.market_info.get('confidence', 0)
            
            print("🧠  MARKET ANALYSIS & STRATEGY SELECTION\n" + "─"*w)
            print(f"📊 Market Condition: {market_condition:<12} │ 📈 ADX: {adx:>5.1f} │ 🎯 Confidence: {confidence*100:>3.0f}%")
            print(f"⚙️  Active Strategy: {self.active_strategy or 'NONE':<13} │ 🕐 Timeframe: {self._get_active_timeframe()}")
            
            # Show corrected break-even rates
            current_be = self.risk_manager.get_current_breakeven_rate() * 100
            print(f"📈 Break-even Win Rate: {current_be:.1f}% (CORRECTED FORMULA)")
            
            # Show cooldown status
            trade_cooldown = self.market_info.get('trade_cooldown_remaining', 0)
            if trade_cooldown > 0:
                minutes = int(trade_cooldown / 60)
                seconds = int(trade_cooldown % 60)
                print(f"⏳ Trade Cooldown: {minutes}m {seconds}s remaining")
            else:
                print("✅ Ready for new trades")
                
            print("─"*w + "\n")
            
            # Fee optimization metrics
            print("💰  FEE OPTIMIZATION METRICS\n" + "─"*w)
            total_orders = self.order_stats['limit_fills'] + self.order_stats['market_fills']
            if total_orders > 0:
                limit_fill_rate = (self.order_stats['limit_fills'] / total_orders) * 100
                print(f"📈 Limit Fill Rate: {limit_fill_rate:.1f}% │ Maker Savings: ${self.order_stats['total_maker_savings']:.2f}")
                print(f"🎯 Limit Orders: {self.order_stats['limit_fills']} │ Market Orders: {self.order_stats['market_fills']} │ Partial: {self.order_stats['partial_fills']}")
            else:
                print("📊 No orders executed yet")
            
            # Show blended fee rate
            blended_fee = float(self.risk_manager.blended_fee_rate) * 100
            print(f"💸 Blended Fee Rate: {blended_fee:.3f}% (30% maker, 70% taker)")
            print("─"*w + "\n")
            
            print("📊  PERFORMANCE METRICS\n" + "─"*w)
            total_trades = sum(self.exit_reasons.values())
            total_signals = self.rejections.get('total_signals', 0)
            fee_rejected = self.rejections.get('fee_inefficient', 0)
            
            print(f"🔢 Total Trades: {total_trades:>3} │ 📈 Signals: {total_signals:>3} │ ✅ Accept Rate: {(total_trades/max(total_signals,1)*100):>4.1f}%")
            print(f"💸 Fee Rejected: {fee_rejected:>3} │ 📝 Check logs/trades.log for detailed analysis")
            
            print("─"*w + "\n")
            print(f"⏰ {time}   |   💰 ${price_formatted}   |   📊 ATR: {self.current_atr:.4f}")
            print()
            
            # Position info with trailing stop details
            if self.position:
                unrealized_pnl = float(self.position.get('unrealisedPnl', 0))
                entry = self.position_entry_price or float(self.position.get('avgPrice', 0))
                size = self.position.get('size', '0')
                side = self.position.get('side', '')
                
                age = (datetime.now() - self.position_start_time).total_seconds() if self.position_start_time else 0
                
                position_size_usdt = self.position_size_usdt or (float(size) * entry)
                fee_cost = position_size_usdt * float(self.risk_manager.total_cost_rate)
                net_pnl = unrealized_pnl - fee_cost
                
                emoji = "🟢" if side == "Buy" else "🔴"
                print(f"{emoji} {side} Position: {size} @ ${entry:.2f} │ Strategy: {self.active_strategy}")
                print(f"   Gross PnL: ${unrealized_pnl:.2f} │ Fees: ${fee_cost:.2f} │ Net: ${net_pnl:.2f}")
                print(f"   Peak PnL: ${self.highest_profit:.2f} │ Age: {age:.1f}s │ Max: {self.risk_manager.get_max_position_time()}s")
                
                if self.active_strategy == "TREND" and self.highest_profit > 0:
                    breakeven_threshold = position_size_usdt * 0.018 * 1.2  # 1.2R for trend
                    if unrealized_pnl >= breakeven_threshold:
                        print(f"   🎯 Trailing Stop Active (1.2R+ move) │ ATR: {self.current_atr:.4f}")
                
            else:
                print("⚡  No Position — Optimized Scanner Active (Limit-First Orders)")
            
            print("─" * 70)
            
        except Exception as e:
            print(f"❌ Display error: {e}")
    
    def _get_active_timeframe(self):
        """Get active timeframe string"""
        timeframes = {"RANGE": "1m", "TREND": "15m"}
        return timeframes.get(self.active_strategy, "Auto")