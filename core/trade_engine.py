import os
import asyncio
import pandas as pd
import numpy as np
import json
from datetime import datetime
from pybit.unified_trading import HTTP
from dotenv import load_dotenv

from strategies.strategy_manager import StrategyManager
from strategies.range_strategy import RangeStrategy  
from strategies.trend_strategy import TrendStrategy
from core.risk_manager import RiskManager
from core.telegram_notifier import TelegramNotifier

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
        
        # Market data - dual timeframe
        self.price_data_1m = pd.DataFrame()
        self.price_data_15m = pd.DataFrame()
        
        # Performance tracking
        self.trade_id = 0
        self.active_strategy = None
        self.market_info = {}
        self.successful_entries = 0  # Track successful position entries
        
        # Performance tracking
        self.exit_reasons = {
            'profit_target': 0, 'emergency_stop': 0, 'max_hold_time': 0,
            'trailing_stop': 0, 'strategy_switch': 0, 'manual_exit': 0
        }
        self.rejections = {
            'invalid_market': 0, 'cooldown_active': 0, 'insufficient_data': 0,
            'invalid_signal': 0, 'total_signals': 0
        }
        
        self._set_symbol_rules()
        os.makedirs("logs", exist_ok=True)
        self.log_file = "logs/trades.log"
    
    def _set_symbol_rules(self):
        """Set symbol-specific trading rules"""
        rules = {'ETH': ('0.01', 0.01), 'BTC': ('0.001', 0.001), 'ADA': ('1', 1.0)}
        for key, (step, min_qty) in rules.items():
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
        """Run one trading cycle with dual strategy support"""
        if not await self._update_market_data():
            return
        
        await self._check_position_status()
        
        if self.position and self.position_start_time:
            await self._check_position_exit()
        
        if not self.position:
            await self._generate_and_execute_signal()
        
        self._display_status()
    
    async def _update_market_data(self):
        """Update both 1m and 15m market data with validation"""
        try:
            # Fetch 1-minute data
            klines_1m = self.exchange.get_kline(category="linear", symbol=self.symbol, interval="1", limit=200)
            if klines_1m.get('retCode') != 0:
                print(f"❌ Failed to fetch 1m data: {klines_1m.get('retMsg', 'Unknown error')}")
                return False
            
            self.price_data_1m = self._process_kline_data(klines_1m['result']['list'])
            
            # Fetch 15-minute data
            klines_15m = self.exchange.get_kline(category="linear", symbol=self.symbol, interval="15", limit=100)
            if klines_15m.get('retCode') != 0:
                print(f"❌ Failed to fetch 15m data: {klines_15m.get('retMsg', 'Unknown error')}")
                return False
            
            self.price_data_15m = self._process_kline_data(klines_15m['result']['list'])
            
            # Validate data quality
            data_valid = (len(self.price_data_1m) > 50 and 
                         len(self.price_data_15m) > 30 and
                         not self.price_data_1m['close'].isna().any() and
                         not self.price_data_15m['close'].isna().any())
            
            if not data_valid:
                print(f"🔍 Data validation failed - 1m: {len(self.price_data_1m)}, 15m: {len(self.price_data_15m)}")
                print(f"🔍 1m NaN check: {self.price_data_1m['close'].isna().sum()}, 15m NaN: {self.price_data_15m['close'].isna().sum()}")
            
            return data_valid
        except Exception as e:
            print(f"❌ Market data update error: {str(e)}")
            return False
    
    def _process_kline_data(self, kline_list):
        """Process kline data into DataFrame with validation"""
        try:
            if not kline_list:
                print("❌ Empty kline data received")
                return pd.DataFrame()
                
            df = pd.DataFrame(kline_list, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
            ])
            
            # Validate we have data
            if df.empty:
                print("❌ Empty DataFrame after processing klines")
                return df
            
            # Convert timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
            
            # Convert price and volume columns to numeric
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Check for NaN values after conversion
            nan_counts = df[numeric_cols].isna().sum()
            if nan_counts.any():
                print(f"⚠️ NaN values detected after conversion: {dict(nan_counts[nan_counts > 0])}")
                # Forward fill NaN values
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].iloc[-1])
            
            # Sort by timestamp and set as index
            df = df.sort_values('timestamp').set_index('timestamp')
            
            # Final validation
            if df.empty or len(df) == 0:
                print("❌ No valid data after processing")
                return pd.DataFrame()
            
            print(f"✅ Processed {len(df)} candles, latest: {df.index[-1]}")
            return df
            
        except Exception as e:
            print(f"❌ Error processing kline data: {str(e)}")
            return pd.DataFrame()
    
    async def _generate_and_execute_signal(self):
        """Generate and execute signals using dual strategy system"""
        # Select strategy based on market conditions
        strategy_type, market_info = self.strategy_manager.select_strategy(
            self.price_data_1m, self.price_data_15m
        )
        
        self.market_info = market_info
        
        # Check for strategy switch
        if self.active_strategy and self.active_strategy != strategy_type:
            await self._on_strategy_switch(self.active_strategy, strategy_type)
        
        # Synchronize risk manager with active strategy
        if self.active_strategy != strategy_type:
            self.risk_manager.set_strategy(strategy_type)
            self.risk_manager.adapt_to_market_condition(
                market_info['condition'], 
                market_info.get('volatility', 'NORMAL')
            )
        
        self.active_strategy = strategy_type
        
        # Generate signal using appropriate strategy
        signal = await self._generate_signal(strategy_type, market_info)
        
        if signal:
            self.rejections['total_signals'] += 1
            # Validate signal before attempting execution
            if self._validate_signal(signal, market_info):
                await self._execute_trade(signal, strategy_type, market_info)
            else:
                # Signal was generated but rejected during validation
                pass
    
    async def _generate_signal(self, strategy_type, market_info):
        """Generate signal using the appropriate strategy"""
        try:
            if strategy_type == "RANGE":
                data = self.price_data_1m
                return self.range_strategy.generate_signal(data, market_info['condition'])
            else:  # TREND
                data = self.price_data_15m
                return self.trend_strategy.generate_signal(data, market_info['condition'])
        except Exception as e:
            self.rejections['invalid_signal'] += 1
            return None
    
    async def _execute_trade(self, signal, strategy_type, market_info):
        """Execute trade with enhanced validation"""
        current_price = float(self.price_data_1m['close'].iloc[-1])
        balance = await self.get_account_balance()
        
        if not balance:
            return
        
        # Validate signal
        if not self._validate_signal(signal, market_info):
            return
        
        # Calculate position size with strategy-specific adjustments
        base_qty = self.risk_manager.calculate_position_size(balance, current_price, signal['structure_stop'])
        sizing_multiplier = self.strategy_manager.get_position_sizing_multiplier(strategy_type, market_info)
        qty = base_qty * sizing_multiplier
        
        formatted_qty = self.format_quantity(qty)
        
        if formatted_qty == "0" or float(formatted_qty) < 0.001:
            return
        
        try:
            order = self.exchange.place_order(
                category="linear", symbol=self.symbol,
                side="Buy" if signal['action'] == 'BUY' else "Sell",
                orderType="Market", qty=formatted_qty, timeInForce="IOC"
            )
            
            if order.get('retCode') == 0:
                self.successful_entries += 1  # Track successful entries
                self._log_trade("ENTRY", current_price, signal=signal, quantity=formatted_qty, strategy=strategy_type)
                await self.notifier.send_trade_entry(signal, current_price, formatted_qty, self._get_strategy_info())
        except:
            pass
    
    def _validate_signal(self, signal, market_info):
        """Validate signal quality"""
        if not signal or market_info['condition'] == 'INSUFFICIENT_DATA':
            self.rejections['insufficient_data'] += 1
            return False
        
        if market_info['confidence'] < 0.6:
            self.rejections['invalid_market'] += 1
            return False
        
        confidence = signal.get('confidence', 0)
        if confidence < 60:
            self.rejections['invalid_signal'] += 1
            return False
        
        return True
    
    async def _check_position_status(self):
        """Check position status"""
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
            self.position = pos_list[0]
        except:
            pass
    
    async def _check_position_exit(self):
        """Check if position should be closed with strategy-specific logic"""
        if not self.position or not self.position_start_time:
            return
        
        current_price = float(self.price_data_1m['close'].iloc[-1])
        entry_price = float(self.position.get('avgPrice', 0))
        side = self.position.get('side', '')
        unrealized_pnl = float(self.position.get('unrealisedPnl', 0))
        position_age = (datetime.now() - self.position_start_time).total_seconds()
        
        # Check strategy-specific exit conditions
        should_close, reason = await self._check_strategy_exit(current_price, entry_price, side, unrealized_pnl, position_age)
        
        if should_close:
            await self._close_position(reason)
    
    async def _check_strategy_exit(self, current_price, entry_price, side, unrealized_pnl, position_age):
        """Check exit conditions based on active strategy"""
        # Ensure risk manager is synchronized with active strategy
        if self.risk_manager.active_strategy != self.active_strategy:
            self.risk_manager.set_strategy(self.active_strategy)
        
        # Basic risk management exits
        basic_exit, basic_reason = self.risk_manager.should_close_position(
            current_price, entry_price, side, unrealized_pnl, position_age
        )
        
        if basic_exit:
            return True, basic_reason
        
        # Strategy-specific exits
        if self.active_strategy == "TREND":
            return await self._check_trend_exit(current_price, entry_price, side, unrealized_pnl)
        else:
            return await self._check_range_exit(current_price, entry_price, side, unrealized_pnl, position_age)
    
    async def _check_trend_exit(self, current_price, entry_price, side, unrealized_pnl):
        """Trend strategy specific exits - trailing stops"""
        if unrealized_pnl > 0:
            should_trail, new_stop = self.trend_strategy.should_trail_stop(
                entry_price, current_price, side, unrealized_pnl
            )
            if should_trail:
                # Implement trailing stop logic here
                pass
        
        return False, "hold"
    
    async def _check_range_exit(self, current_price, entry_price, side, unrealized_pnl, position_age):
        """Range strategy specific exits - quick scalping exits"""
        # Quick profit taking for range strategy
        profit_threshold = self.range_strategy.config['target_profit_usdt']
        max_hold = self.range_strategy.config['max_hold_seconds']
        
        if unrealized_pnl >= profit_threshold:
            return True, "profit_target"
        
        if position_age >= max_hold:
            return True, "max_hold_time"
        
        return False, "hold"
    
    async def _close_position(self, reason="Manual"):
        """Close position"""
        if not self.position:
            return
        
        current_price = float(self.price_data_1m['close'].iloc[-1]) if len(self.price_data_1m) > 0 else 0
        pnl = float(self.position.get('unrealisedPnl', 0))
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
                self._log_trade("EXIT", current_price, reason=reason, pnl=pnl, strategy=self.active_strategy)
                
                exit_data = {'trigger': reason, 'strategy': self.active_strategy}
                await self.notifier.send_trade_exit(exit_data, current_price, pnl, duration, self._get_strategy_info())
        except:
            pass
    
    async def _on_strategy_switch(self, old_strategy, new_strategy):
        """Handle strategy switch with proper synchronization"""
        print(f"\n🔄 Strategy Switch: {old_strategy} → {new_strategy}")
        
        try:
            # Close position if strategy switch occurs
            if self.position:
                print("📤 Closing position due to strategy switch...")
                await self._close_position("strategy_switch")
                
                # Wait for position to close before continuing
                max_wait = 5  # 5 seconds max wait
                wait_count = 0
                while self.position and wait_count < max_wait:
                    await asyncio.sleep(1)
                    await self._check_position_status()
                    wait_count += 1
                    
                if self.position:
                    print("⚠️ Position did not close cleanly, forcing reset")
                    self._reset_position()
            
            # Update exit reasons tracking
            self.exit_reasons['strategy_switch'] += 1
            
        except Exception as e:
            print(f"❌ Error during strategy switch: {e}")
            # Force reset to prevent stuck states
            self._reset_position()
    
    async def _on_position_closed(self):
        """Handle position closed externally"""
        if self.position:
            pnl = float(self.position.get('unrealisedPnl', 0))
            price = float(self.price_data_1m['close'].iloc[-1]) if len(self.price_data_1m) > 0 else 0
            self._track_exit_reason('position_closed')
            self._log_trade("EXIT", price, reason="position_closed", pnl=pnl, strategy=self.active_strategy)
    
    def _reset_position(self):
        """Reset position state"""
        self.position = None
        self.position_start_time = None
    
    def _track_exit_reason(self, reason):
        """Track exit reason"""
        if reason in self.exit_reasons:
            self.exit_reasons[reason] += 1
        else:
            self.exit_reasons['manual_exit'] += 1
    
    def _log_trade(self, action, price, **kwargs):
        """Enhanced trade logging"""
        timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
        
        if action == "ENTRY":
            self.trade_id += 1
            signal = kwargs.get('signal', {})
            log_data = {
                'timestamp': timestamp, 'id': self.trade_id, 'action': 'ENTRY',
                'strategy': kwargs.get('strategy', 'UNKNOWN'),
                'side': signal.get('action', ''), 'price': round(price, 2), 
                'size': kwargs.get('quantity', ''), 'market_condition': self.market_info.get('condition', ''),
                'adx': round(self.market_info.get('adx', 0), 1),
                'confidence': round(signal.get('confidence', 0), 1)
            }
        else:
            duration = (datetime.now() - self.position_start_time).total_seconds() if self.position_start_time else 0
            log_data = {
                'timestamp': timestamp, 'id': self.trade_id, 'action': 'EXIT',
                'strategy': kwargs.get('strategy', 'UNKNOWN'),
                'trigger': kwargs.get('reason', '').lower().replace(' ', '_'),
                'price': round(price, 2), 'pnl': round(kwargs.get('pnl', 0), 2),
                'hold_seconds': round(duration, 1)
            }
        
        try:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(log_data) + "\n")
        except:
            pass
    
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
        if self.active_strategy == "RANGE":
            return self.range_strategy.get_strategy_info()
        else:
            return self.trend_strategy.get_strategy_info()
    
    def _display_status(self):
        """Display enhanced status with dual strategy info"""
        try:
            price = float(self.price_data_1m['close'].iloc[-1])
            time = self.price_data_1m.index[-1].strftime('%H:%M:%S')
            symbol_display = self.symbol.replace('USDT', '/USDT')
            price_formatted = f"{price:,.2f}".replace(',', ' ')
            
            print("\n" * 50)
            
            # Header
            w = 77
            print(f"{'='*w}\n⚡  {symbol_display} DUAL-STRATEGY TRADING BOT\n{'='*w}\n")
            
            # Market condition and strategy
            market_condition = self.market_info.get('condition', 'UNKNOWN')
            adx = self.market_info.get('adx', 0)
            confidence = self.market_info.get('confidence', 0)
            
            print("🧠  MARKET ANALYSIS & STRATEGY SELECTION\n" + "─"*w)
            print(f"📊 Market Condition: {market_condition:<12} │ 📈 ADX: {adx:>5.1f} │ 🎯 Confidence: {confidence*100:>3.0f}%")
            print(f"⚙️  Active Strategy: {self.active_strategy or 'NONE':<13} │ 🕐 Timeframe: {self._get_active_timeframe()}")
            print("─"*w + "\n")
            
            # Strategy status
            print("📋  STRATEGY STATUS\n" + "─"*w)
            if self.active_strategy == "RANGE":
                strategy_info = self.range_strategy.get_strategy_info()
                print(f"🎯 {strategy_info['name']}")
                print(f"📊 RSI({strategy_info['config']['rsi_length']}) + MFI({strategy_info['config']['mfi_length']}) │ Target: ${strategy_info['config']['target_profit_usdt']} │ Hold: {strategy_info['config']['max_hold_seconds']}s")
            else:
                strategy_info = self.trend_strategy.get_strategy_info()
                print(f"📈 {strategy_info['name']}")
                print(f"📊 RSI({strategy_info['config']['rsi_length']}) + MA({strategy_info['config']['ma_length']}) │ RR: 1:{strategy_info['config']['target_profit_multiplier']} │ Win Rate: {strategy_info['win_rate']}")
            print("─"*w + "\n")
            
            # Performance metrics
            print("📊  PERFORMANCE METRICS\n" + "─"*w)
            total_trades = sum(self.exit_reasons.values())
            total_signals = self.rejections.get('total_signals', 0)
            
            print(f"🔢 Total Trades: {total_trades:>3} │ 📈 Signals: {total_signals:>3} │ ✅ Accept Rate: {(total_trades/max(total_signals,1)*100):>4.1f}%")
            
            # Adjust for active position (trade count should include open positions)
            active_trades = total_trades + (1 if self.position else 0)
            if total_signals > 0:
                actual_accept_rate = (active_trades / total_signals) * 100
                print(f"📊 Active Trades: {active_trades:>3} │ 🎯 Current Accept Rate: {actual_accept_rate:>4.1f}%")
            
            # Exit reasons
            top_exits = sorted(self.exit_reasons.items(), key=lambda x: x[1], reverse=True)[:3]
            if any(count > 0 for _, count in top_exits):
                exit_str = " │ ".join([f"{reason}: {count}" for reason, count in top_exits if count > 0])
                print(f"🏁 Top Exits: {exit_str}")
            print("─"*w + "\n")
            
            # Current status
            print(f"⏰ {time}   |   💰 ${price_formatted}")
            print()
            
            # Position info
            if self.position:
                pnl = float(self.position.get('unrealisedPnl', 0))
                entry = float(self.position.get('avgPrice', 0))
                size = self.position.get('size', '0')
                side = self.position.get('side', '')
                
                pnl_pct = (pnl / (float(size) * entry)) * 100 if entry > 0 and size != '0' else 0
                age = (datetime.now() - self.position_start_time).total_seconds() if self.position_start_time else 0
                
                emoji = "🟢" if side == "Buy" else "🔴"
                print(f"{emoji} {side} Position: {size} @ ${entry:.2f} │ Strategy: {self.active_strategy}")
                print(f"   PnL: ${pnl:.2f} ({pnl_pct:+.2f}%) | Age: {age:.1f}s")
            else:
                print("⚡  No Position — Multi-Strategy Scanner Active")
            
            print("─" * 60)
            
        except Exception as e:
            print(f"❌ Display error: {e}")
    
    def _get_active_timeframe(self):
        """Get active timeframe string"""
        if self.active_strategy == "RANGE":
            return "1m"
        elif self.active_strategy == "TREND":
            return "15m"
        else:
            return "Auto"