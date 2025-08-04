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
        self.position_entry_price = None  # FIXED: Track entry price
        self.position_size_usdt = None    # FIXED: Track position size in USDT
        
        # Market data
        self.price_data_1m = pd.DataFrame()
        self.price_data_15m = pd.DataFrame()
        
        # State tracking
        self.trade_id = 0
        self.active_strategy = None
        self.market_info = {}
        self.successful_entries = 0
        
        # Performance tracking
        self.exit_reasons = {
            'profit_target': 0, 'emergency_stop': 0, 'max_hold_time': 0,
            'trailing_stop': 0, 'strategy_switch': 0, 'manual_exit': 0,
            'timeout_no_profit': 0  # FIXED: Add timeout tracking
        }
        self.rejections = {
            'invalid_market': 0, 'cooldown_active': 0, 'insufficient_data': 0,
            'invalid_signal': 0, 'total_signals': 0, 'unprofitable': 0  # FIXED: Track unprofitable rejections
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
        """Run one trading cycle"""
        if not await self._update_market_data():
            return
        
        await self._check_position_status()
        
        if self.position and self.position_start_time:
            await self._check_position_exit()
        
        if not self.position:
            await self._generate_and_execute_signal()
        
        self._display_status()
    
    async def _update_market_data(self):
        """Update both 1m and 15m market data"""
        try:
            # Fetch data
            klines_1m = self.exchange.get_kline(category="linear", symbol=self.symbol, interval="1", limit=200)
            klines_15m = self.exchange.get_kline(category="linear", symbol=self.symbol, interval="15", limit=100)
            
            if klines_1m.get('retCode') != 0 or klines_15m.get('retCode') != 0:
                return False
            
            self.price_data_1m = self._process_kline_data(klines_1m['result']['list'])
            self.price_data_15m = self._process_kline_data(klines_15m['result']['list'])
            
            # Validate data quality
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
        
        # Convert and validate data
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Forward fill any NaN values
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].iloc[-1])
        
        # Sort and set index
        return df.sort_values('timestamp').set_index('timestamp')
    
    async def _generate_and_execute_signal(self):
        """Generate and execute signals using dual strategy system"""
        # Select strategy based on market conditions
        strategy_type, market_info = self.strategy_manager.select_strategy(
            self.price_data_1m, self.price_data_15m
        )
        
        self.market_info = market_info
        
        # Handle strategy switch
        if self.active_strategy and self.active_strategy != strategy_type:
            await self._on_strategy_switch(self.active_strategy, strategy_type)
        
        # Synchronize risk manager
        if self.active_strategy != strategy_type:
            self.risk_manager.set_strategy(strategy_type)
            self.risk_manager.adapt_to_market_condition(
                market_info['condition'], 
                market_info.get('volatility', 'NORMAL')
            )
        
        self.active_strategy = strategy_type
        
        # Generate and execute signal
        signal = self._generate_signal(strategy_type, market_info)
        
        if signal:
            self.rejections['total_signals'] += 1
            if self._validate_signal(signal, market_info):
                await self._execute_trade(signal, strategy_type, market_info)
    
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
    
    async def _execute_trade(self, signal, strategy_type, market_info):
        """FIXED: Execute trade with proper fee calculations"""
        current_price = float(self.price_data_1m['close'].iloc[-1])
        balance = await self.get_account_balance()
        
        if not balance or not self._validate_signal(signal, market_info):
            return
        
        # Calculate position size
        base_qty = self.risk_manager.calculate_position_size(balance, current_price, signal['structure_stop'])
        sizing_multiplier = self.strategy_manager.get_position_sizing_multiplier(strategy_type, market_info)
        qty = base_qty * sizing_multiplier
        
        formatted_qty = self.format_quantity(qty)
        
        if formatted_qty == "0" or float(formatted_qty) < 0.001:
            return
        
        # FIXED: Calculate position size in USDT for fee tracking
        position_size_usdt = float(formatted_qty) * current_price
        
        # FIXED: Pre-validate profitability
        if not self.risk_manager.validate_trade_profitability(position_size_usdt, 
                                                             position_size_usdt * 0.02):  # Assume 2% potential
            self.rejections['unprofitable'] += 1
            return
        
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
                self.successful_entries += 1
                self.position_entry_price = current_price  # FIXED: Track entry price
                self.position_size_usdt = position_size_usdt  # FIXED: Track position size
                
                self._log_trade("ENTRY", current_price, signal=signal, quantity=formatted_qty, 
                               strategy=strategy_type, position_size_usdt=position_size_usdt)
                await self.notifier.send_trade_entry(signal, current_price, formatted_qty, self._get_strategy_info())
        except:
            pass
    
    def _validate_signal(self, signal, market_info):
        """FIXED: Enhanced signal validation"""
        validation_checks = [
            (not signal or market_info['condition'] == 'INSUFFICIENT_DATA', 'insufficient_data'),
            (market_info['confidence'] < 0.6, 'invalid_market'),
            (signal.get('confidence', 0) < 60, 'invalid_signal')
        ]
        
        for condition, rejection_type in validation_checks:
            if condition:
                self.rejections[rejection_type] += 1
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
                # Try to get entry price from position data if not tracked
                if not self.position_entry_price:
                    self.position_entry_price = float(pos_list[0].get('avgPrice', 0))
                if not self.position_size_usdt and self.position_entry_price:
                    qty = float(pos_list[0]['size'])
                    self.position_size_usdt = qty * self.position_entry_price
                    
            self.position = pos_list[0]
        except:
            pass
    
    async def _check_position_exit(self):
        """FIXED: Check if position should be closed with accurate PnL"""
        if not self.position or not self.position_start_time:
            return
        
        current_price = float(self.price_data_1m['close'].iloc[-1])
        entry_price = self.position_entry_price or float(self.position.get('avgPrice', 0))
        side = self.position.get('side', '')
        unrealized_pnl = float(self.position.get('unrealisedPnl', 0))
        position_age = (datetime.now() - self.position_start_time).total_seconds()
        
        # Ensure risk manager is synchronized
        if self.risk_manager.active_strategy != self.active_strategy:
            self.risk_manager.set_strategy(self.active_strategy)
        
        # FIXED: Pass position size for accurate fee calculation
        should_close, reason = self.risk_manager.should_close_position(
            current_price, entry_price, side, unrealized_pnl, position_age, self.position_size_usdt
        )
        
        if should_close:
            await self._close_position(reason)
    
    async def _close_position(self, reason="Manual"):
        """FIXED: Close position with accurate PnL calculation"""
        if not self.position:
            return
        
        current_price = float(self.price_data_1m['close'].iloc[-1]) if len(self.price_data_1m) > 0 else 0
        gross_pnl = float(self.position.get('unrealisedPnl', 0))
        
        # FIXED: Calculate net PnL after fees
        net_pnl = gross_pnl
        if self.position_size_usdt:
            net_pnl = self.risk_manager.calculate_net_pnl(gross_pnl, self.position_size_usdt)
        
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
                self._log_trade("EXIT", current_price, reason=reason, pnl=net_pnl, 
                               gross_pnl=gross_pnl, strategy=self.active_strategy, duration=duration)
                
                exit_data = {'trigger': reason, 'strategy': self.active_strategy}
                await self.notifier.send_trade_exit(exit_data, current_price, net_pnl, duration, self._get_strategy_info())
        except:
            pass
    
    async def _on_strategy_switch(self, old_strategy, new_strategy):
        """Handle strategy switch"""
        if self.position:
            await self._close_position("strategy_switch")
            
            # Wait for position to close
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
            gross_pnl = float(self.position.get('unrealisedPnl', 0))
            net_pnl = self.risk_manager.calculate_net_pnl(gross_pnl, self.position_size_usdt) if self.position_size_usdt else gross_pnl
            price = float(self.price_data_1m['close'].iloc[-1]) if len(self.price_data_1m) > 0 else 0
            self._track_exit_reason('position_closed')
            self._log_trade("EXIT", price, reason="position_closed", pnl=net_pnl, 
                           gross_pnl=gross_pnl, strategy=self.active_strategy)
    
    def _reset_position(self):
        """FIXED: Reset position state completely"""
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
    
    def _log_trade(self, action, price, **kwargs):
        """FIXED: Enhanced trade logging"""
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
                'confidence': round(signal.get('confidence', 0), 1),
                'position_size_usdt': round(kwargs.get('position_size_usdt', 0), 2),
                'estimated_fees': round(kwargs.get('position_size_usdt', 0) * self.risk_manager.fee_rate, 2)
            }
        else:
            duration = kwargs.get('duration', 0)
            log_data = {
                'timestamp': timestamp, 'id': self.trade_id, 'action': 'EXIT',
                'strategy': kwargs.get('strategy', 'UNKNOWN'),
                'trigger': kwargs.get('reason', '').lower().replace(' ', '_'),
                'price': round(price, 2), 
                'gross_pnl': round(kwargs.get('gross_pnl', 0), 2),
                'net_pnl': round(kwargs.get('pnl', 0), 2),
                'hold_seconds': round(duration, 1),
                'fees_paid': round(kwargs.get('gross_pnl', 0) - kwargs.get('pnl', 0), 2) if kwargs.get('gross_pnl') else 0
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
        return (self.range_strategy.get_strategy_info() if self.active_strategy == "RANGE" 
                else self.trend_strategy.get_strategy_info())
    
    def _display_status(self):
        """FIXED: Display trading status with accurate PnL"""
        try:
            price = float(self.price_data_1m['close'].iloc[-1])
            time = self.price_data_1m.index[-1].strftime('%H:%M:%S')
            symbol_display = self.symbol.replace('USDT', '/USDT')
            price_formatted = f"{price:,.2f}".replace(',', ' ')
            
            print("\n" * 50)
            
            # Header
            w = 77
            print(f"{'='*w}\n⚡  {symbol_display} DUAL-STRATEGY TRADING BOT (FIXED)\n{'='*w}\n")
            
            # Market condition and strategy
            market_condition = self.market_info.get('condition', 'UNKNOWN')
            adx = self.market_info.get('adx', 0)
            confidence = self.market_info.get('confidence', 0)
            
            print("🧠  MARKET ANALYSIS & STRATEGY SELECTION\n" + "─"*w)
            print(f"📊 Market Condition: {market_condition:<12} │ 📈 ADX: {adx:>5.1f} │ 🎯 Confidence: {confidence*100:>3.0f}%")
            print(f"⚙️  Active Strategy: {self.active_strategy or 'NONE':<13} │ 🕐 Timeframe: {self._get_active_timeframe()}")
            print("─"*w + "\n")
            
            # FIXED: Strategy status with accurate fee info
            print("📋  STRATEGY STATUS (FIXED PROFIT TARGETS)\n" + "─"*w)
            if self.active_strategy == "RANGE":
                strategy_info = self.range_strategy.get_strategy_info()
                position_size = self.risk_manager.range_config['fixed_position_usdt']
                fee_cost = position_size * self.risk_manager.fee_rate
                gross_target = self.risk_manager.calculate_fee_adjusted_profit_target(position_size)
                net_target = self.risk_manager.range_config['base_profit_usdt']
                print(f"🎯 {strategy_info['name']}")
                print(f"📊 Target: ${net_target} NET (${gross_target:.2f} gross) │ Fees: ${fee_cost:.2f} │ Hold: {strategy_info['config']['max_hold_seconds']}s")
            else:
                strategy_info = self.trend_strategy.get_strategy_info()
                print(f"📈 {strategy_info['name']}")
                print(f"📊 RSI({strategy_info['config']['rsi_length']}) + MA({strategy_info['config']['ma_length']}) │ RR: 1:{strategy_info['config']['target_profit_multiplier']} │ Win Rate: {strategy_info['win_rate']}")
            print("─"*w + "\n")
            
            # FIXED: Performance metrics with rejection reasons
            print("📊  PERFORMANCE METRICS\n" + "─"*w)
            total_trades = sum(self.exit_reasons.values())
            total_signals = self.rejections.get('total_signals', 0)
            unprofitable_rejections = self.rejections.get('unprofitable', 0)
            
            print(f"🔢 Total Trades: {total_trades:>3} │ 📈 Signals: {total_signals:>3} │ ✅ Accept Rate: {(total_trades/max(total_signals,1)*100):>4.1f}%")
            if unprofitable_rejections > 0:
                print(f"💰 Unprofitable Rejections: {unprofitable_rejections} (Good - avoiding losses!)")
            
            # Current status
            print("─"*w + "\n")
            print(f"⏰ {time}   |   💰 ${price_formatted}")
            print()
            
            # FIXED: Position info with net PnL
            if self.position:
                gross_pnl = float(self.position.get('unrealisedPnl', 0))
                entry = self.position_entry_price or float(self.position.get('avgPrice', 0))
                size = self.position.get('size', '0')
                side = self.position.get('side', '')
                
                # Calculate net PnL
                net_pnl = gross_pnl
                if self.position_size_usdt:
                    net_pnl = self.risk_manager.calculate_net_pnl(gross_pnl, self.position_size_usdt)
                    fee_cost = self.position_size_usdt * self.risk_manager.fee_rate
                else:
                    fee_cost = 0
                
                net_pnl_pct = (net_pnl / (float(size) * entry)) * 100 if entry > 0 and size != '0' else 0
                age = (datetime.now() - self.position_start_time).total_seconds() if self.position_start_time else 0
                
                emoji = "🟢" if side == "Buy" else "🔴"
                print(f"{emoji} {side} Position: {size} @ ${entry:.2f} │ Strategy: {self.active_strategy}")
                print(f"   Gross PnL: ${gross_pnl:.2f} │ Fees: ${fee_cost:.2f} │ NET PnL: ${net_pnl:.2f} ({net_pnl_pct:+.2f}%)")
                print(f"   Age: {age:.1f}s │ Max Hold: {self.risk_manager.get_max_position_time()}s")
            else:
                print("⚡  No Position — Fee-Aware Multi-Strategy Scanner Active")
            
            print("─" * 60)
            
        except Exception as e:
            print(f"❌ Display error: {e}")
    
    def _get_active_timeframe(self):
        """Get active timeframe string"""
        timeframes = {"RANGE": "1m", "TREND": "15m"}
        return timeframes.get(self.active_strategy, "Auto")