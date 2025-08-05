import os
import asyncio
import pandas as pd
import numpy as np
import json
from datetime import datetime
from decimal import Decimal
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
        
        # Market data - 3-minute timeframes
        self.price_data_3m = pd.DataFrame()
        self.price_data_3m_secondary = pd.DataFrame()  # Secondary 3m for confirmation
        
        # State tracking
        self.trade_id = 0
        self.active_strategy = None
        self.market_info = {}
        self.successful_entries = 0
        
        # Performance tracking
        self.exit_reasons = {
            'profit_target': 0, 'emergency_stop': 0, 'max_hold_time': 0,
            'trailing_stop': 0, 'strategy_switch': 0, 'manual_exit': 0,
            'timeout_no_profit': 0
        }
        self.rejections = {
            'invalid_market': 0, 'cooldown_active': 0, 'insufficient_data': 0,
            'invalid_signal': 0, 'total_signals': 0, 'unprofitable': 0
        }
        
        self._set_symbol_rules()
        os.makedirs("logs", exist_ok=True)
        self.log_file = "logs/trades_3m.log"
    
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
        """Run one 3-minute trading cycle"""
        if not await self._update_market_data_3m():
            return
        
        await self._check_position_status()
        
        if self.position and self.position_start_time:
            await self._check_position_exit()
        
        if not self.position:
            await self._generate_and_execute_signal()
        
        self._display_status_3m()
    
    async def _update_market_data_3m(self):
        """Update 3-minute market data"""
        try:
            # Primary 3m data for main analysis
            klines_3m = self.exchange.get_kline(category="linear", symbol=self.symbol, interval="3", limit=100)
            # Secondary 3m data for confirmation (slightly offset)
            klines_3m_secondary = self.exchange.get_kline(category="linear", symbol=self.symbol, interval="3", limit=60)
            
            if klines_3m.get('retCode') != 0 or klines_3m_secondary.get('retCode') != 0:
                return False
            
            self.price_data_3m = self._process_kline_data(klines_3m['result']['list'])
            self.price_data_3m_secondary = self._process_kline_data(klines_3m_secondary['result']['list'])
            
            return (len(self.price_data_3m) > 30 and 
                   len(self.price_data_3m_secondary) > 20 and
                   not self.price_data_3m['close'].isna().any() and
                   not self.price_data_3m_secondary['close'].isna().any())
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
        """3-Minute Trading: Generate signals with faster cooldowns"""
        strategy_type, market_info = self.strategy_manager.select_strategy(
            self.price_data_3m, self.price_data_3m_secondary
        )
        
        self.market_info = market_info
        
        # 3m trading: Shorter cooldowns (5 minutes between trades)
        if not market_info.get('trade_allowed', False):
            cooldown_remaining = market_info.get('trade_cooldown_remaining', 0)
            if cooldown_remaining > 30:  # Only show if > 30 seconds remaining
                minutes = int(cooldown_remaining / 60)
                seconds = int(cooldown_remaining % 60)
                print(f"⏳ Trade cooldown: {minutes}m {seconds}s remaining")
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
        signal = self._generate_signal_3m(strategy_type, market_info)
        
        if signal:
            self.rejections['total_signals'] += 1
            if self._validate_signal_3m(signal, market_info):
                await self._execute_trade_3m(signal, strategy_type, market_info)
    
    def _generate_signal_3m(self, strategy_type, market_info):
        """Generate signal using 3-minute data"""
        try:
            # Both strategies use 3m data for fast execution
            if strategy_type == "RANGE":
                return self.range_strategy.generate_signal(self.price_data_3m, market_info['condition'])
            else:  # TREND
                return self.trend_strategy.generate_signal(self.price_data_3m, market_info['condition'])
        except:
            self.rejections['invalid_signal'] += 1
            return None
    
    def _validate_signal_3m(self, signal, market_info):
        """Enhanced signal validation for 3m trading"""
        validation_checks = [
            (not signal or market_info['condition'] == 'INSUFFICIENT_DATA', 'insufficient_data'),
            (market_info['confidence'] < 0.65, 'invalid_market'),  # Higher confidence for 3m
            (signal.get('confidence', 0) < 65, 'invalid_signal')   # Higher confidence threshold
        ]
        
        for condition, rejection_type in validation_checks:
            if condition:
                self.rejections[rejection_type] += 1
                return False
        
        return True
    
    async def _execute_trade_3m(self, signal, strategy_type, market_info):
        """Execute 3-minute trade with $3,817 position sizing"""
        current_price = float(self.price_data_3m['close'].iloc[-1])
        balance = await self.get_account_balance()
        
        if not balance or not self._validate_signal_3m(signal, market_info):
            return
        
        # 3m trading: $3,817 position size
        base_qty = self.risk_manager.calculate_position_size(
            balance, current_price, signal['structure_stop']
        )
        
        # More conservative sizing multiplier for 3m
        sizing_multiplier = self.strategy_manager.get_position_sizing_multiplier(
            strategy_type, market_info
        )
        qty = base_qty * sizing_multiplier
        
        formatted_qty = self.format_quantity(qty)
        
        if formatted_qty == "0" or float(formatted_qty) < 0.001:
            print(f"❌ Position too small: {formatted_qty}")
            return
        
        # Calculate actual position size
        position_size_usdt = float(formatted_qty) * current_price
        
        # Validate position size for 3m trading (cap at $4,000)
        if position_size_usdt > 4000:
            print(f"❌ Position too large: ${position_size_usdt:.2f}, capping at $4,000")
            qty = 4000 / current_price
            formatted_qty = self.format_quantity(qty)
            position_size_usdt = float(formatted_qty) * current_price
        
        break_even_fees = self.risk_manager.get_break_even_pnl(position_size_usdt)
        
        try:
            order = self.exchange.place_order(
                category="linear", 
                symbol=self.symbol,
                side="Buy" if signal['action'] == 'BUY' else "Sell",
                orderType="Market", 
                qty=formatted_qty, 
                timeInForce="IOC"
            )
            
            if order.get('retCode') == 0:
                # Record trade time for 3m cooldown (5 minutes)
                self.strategy_manager.record_trade()
                
                self.successful_entries += 1
                self.position_entry_price = current_price
                self.position_size_usdt = position_size_usdt
                
                self._log_trade_3m("ENTRY", current_price, signal=signal, quantity=formatted_qty, 
                                  strategy=strategy_type, position_size_usdt=position_size_usdt,
                                  break_even_fees=break_even_fees)
                               
                await self.notifier.send_trade_entry(signal, current_price, formatted_qty, 
                                                   self._get_strategy_info())
                
                print(f"✅ 3M Trade: {signal['action']} {formatted_qty} @ ${current_price:.2f}")
                print(f"   Position: ${position_size_usdt:.2f} | Break-even: ${break_even_fees:.2f}")
                
        except Exception as e:
            print(f"❌ Trade execution error: {e}")
    
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
                if not self.position_entry_price:
                    self.position_entry_price = float(pos_list[0].get('avgPrice', 0))
                if not self.position_size_usdt and self.position_entry_price:
                    qty = float(pos_list[0]['size'])
                    self.position_size_usdt = qty * self.position_entry_price
                    
            self.position = pos_list[0]
        except:
            pass
    
    async def _check_position_exit(self):
        """3-Minute Trading: Fast exit checking"""
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
        
        # Fast 3m exit decision
        should_close, reason = self.risk_manager.should_close_position(
            current_price, entry_price, side, unrealized_pnl, position_age, position_size_usdt
        )
        
        if should_close:
            await self._close_position(reason)
    
    async def _close_position(self, reason="Manual"):
        """Close 3m position and log"""
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
                
                self._log_trade_3m("EXIT", current_price, reason=reason, bybit_unrealized_pnl=unrealized_pnl, 
                                  strategy=self.active_strategy, duration=duration, 
                                  position_size_usdt=self.position_size_usdt or 0)
                
                exit_data = {'trigger': reason, 'strategy': self.active_strategy}
                await self.notifier.send_trade_exit(exit_data, current_price, unrealized_pnl, duration, self._get_strategy_info())
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
            price = float(self.price_data_3m['close'].iloc[-1]) if len(self.price_data_3m) > 0 else 0
            self._track_exit_reason('position_closed')
            self._log_trade_3m("EXIT", price, reason="position_closed", bybit_unrealized_pnl=unrealized_pnl, 
                              strategy=self.active_strategy, position_size_usdt=self.position_size_usdt or 0)
    
    def _reset_position(self):
        """Reset position state"""
        self.position = None
        self.position_start_time = None
        self.position_entry_price = None
        self.position_size_usdt = None
    
    def _track_exit_reason(self, reason):
        """Track exit reason"""
        if reason in self.exit_reasons:
            self.exit_reasons[reason] += 1
        else:
            self.exit_reasons['manual_exit'] += 1
    
    def _log_trade_3m(self, action, price, **kwargs):
        """3-Minute trading log with proper fees"""
        timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
        
        if action == "ENTRY":
            self.trade_id += 1
            signal = kwargs.get('signal', {})
            position_size_usdt = kwargs.get('position_size_usdt', 0)
            break_even_fees = kwargs.get('break_even_fees', 0)
            
            log_data = {
                'timestamp': timestamp, 'id': self.trade_id, 'action': 'ENTRY',
                'strategy': kwargs.get('strategy', 'UNKNOWN'), 'timeframe': '3m',
                'side': signal.get('action', ''), 'price': round(price, 2), 
                'size': kwargs.get('quantity', ''), 
                'market_condition': self.market_info.get('condition', ''),
                'adx': round(self.market_info.get('adx', 0), 1),
                'confidence': round(signal.get('confidence', 0), 1),
                'position_size_usdt': round(position_size_usdt, 2),
                'break_even_fees': round(break_even_fees, 2),
                'note': f'3m trading: Need ${break_even_fees:.2f} profit to cover 0.11% fees'
            }
        else:
            duration = kwargs.get('duration', 0)
            bybit_pnl = kwargs.get('bybit_unrealized_pnl', 0)
            position_size_usdt = kwargs.get('position_size_usdt', 0)
            
            if position_size_usdt > 0:
                break_even_fees = self.risk_manager.get_break_even_pnl(position_size_usdt)
                estimated_net = bybit_pnl - break_even_fees
            else:
                estimated_net = bybit_pnl
                break_even_fees = 0
            
            log_data = {
                'timestamp': timestamp, 'id': self.trade_id, 'action': 'EXIT',
                'strategy': kwargs.get('strategy', 'UNKNOWN'), 'timeframe': '3m',
                'trigger': kwargs.get('reason', '').lower().replace(' ', '_'),
                'price': round(price, 2), 
                'bybit_unrealized_pnl': round(bybit_pnl, 2),
                'estimated_fees': round(break_even_fees, 2),
                'estimated_net_profit': round(estimated_net, 2),
                'hold_seconds': round(duration, 1),
                'note': '3m trading: Estimated net = unrealized_pnl - fees (0.11%)'
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
    
    def _display_status_3m(self):
        """3-Minute Trading Display"""
        try:
            price = float(self.price_data_3m['close'].iloc[-1])
            time = self.price_data_3m.index[-1].strftime('%H:%M:%S')
            symbol_display = self.symbol.replace('USDT', '/USDT')
            price_formatted = f"{price:,.2f}".replace(',', ' ')
            
            print("\n" * 50)
            
            w = 77
            print(f"{'='*w}\n⚡  {symbol_display} 3-MINUTE TRADING BOT v3.0\n{'='*w}\n")
            
            market_condition = self.market_info.get('condition', 'UNKNOWN')
            adx = self.market_info.get('adx', 0)
            confidence = self.market_info.get('confidence', 0)
            
            print("🧠  3-MINUTE MARKET ANALYSIS\n" + "─"*w)
            print(f"📊 Market Condition: {market_condition:<12} │ 📈 ADX: {adx:>5.1f} │ 🎯 Confidence: {confidence*100:>3.0f}%")
            print(f"⚙️  Active Strategy: {self.active_strategy or 'NONE':<13} │ 🕐 Timeframe: 3m")
            
            # Show cooldown status
            trade_cooldown = self.market_info.get('trade_cooldown_remaining', 0)
            if trade_cooldown > 0:
                minutes = int(trade_cooldown / 60)
                seconds = int(trade_cooldown % 60)
                print(f"⏳ Trade Cooldown: {minutes}m {seconds}s remaining")
            else:
                print("✅ Ready for 3m trades")
                
            print("─"*w + "\n")
            
            print("📋  3-MINUTE STRATEGY STATUS\n" + "─"*w)
            if self.active_strategy == "RANGE":
                print(f"🎯 3m Range Strategy: RSI+BB (Fast Scalping)")
                print(f"📊 Position Size: $3,817 │ Target: $92 (2.4%)")
            else:
                print(f"📈 3m Trend Strategy: EMA+RSI (Quick Momentum)")
                print(f"📊 Position Size: $3,817 │ RR: 1:2.5")
                
            print(f"⏰ Max Hold: 45 minutes │ Emergency Stop: 1.2%")
            print("─"*w + "\n")
            
            print("📊  3-MINUTE PERFORMANCE\n" + "─"*w)
            total_trades = sum(self.exit_reasons.values())
            total_signals = self.rejections.get('total_signals', 0)
            print(f"🔢 Total Trades: {total_trades:>3} │ 📈 Signals: {total_signals:>3} │ ✅ Accept Rate: {(total_trades/max(total_signals,1)*100):>4.1f}%")
            print(f"📝 Log: logs/trades_3m.log")
            
            print("─"*w + "\n")
            print(f"⏰ {time}   |   💰 ${price_formatted}")
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
                print(f"{emoji} 3m {side}: {size} @ ${entry:.2f} │ Strategy: {self.active_strategy}")
                print(f"   Gross PnL: ${unrealized_pnl:.2f} │ Fees: ${break_even_fees:.2f} │ Net: ${net_pnl:.2f}")
                print(f"   Age: {age:.1f}s │ Max: 2700s (45min)")
            else:
                print("⚡  No Position — 3-Minute Scanner Active (5min cooldowns)")
            
            print("─" * 60)
            
        except Exception as e:
            print(f"❌ Display error: {e}")
    
    def _get_active_timeframe(self):
        """Get active timeframe string"""
        return "3m"