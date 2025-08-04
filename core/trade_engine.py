import os
import asyncio
import pandas as pd
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
        # Core components
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
        
        # Market data
        self.price_data_1m = pd.DataFrame()
        self.price_data_15m = pd.DataFrame()
        
        # State tracking
        self.trade_id = 0
        self.active_strategy = None
        self.market_info = {}
        
        # Performance tracking
        self.exit_reasons = {}
        self.rejections = {'total_signals': 0}
        
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
        """Main trading cycle"""
        if not await self._update_market_data():
            return
        
        await self._check_position_status()
        
        if self.position:
            await self._check_position_exit()
        else:
            await self._generate_and_execute_signal()
        
        self._display_status()
    
    async def _update_market_data(self):
        """Update market data for both timeframes"""
        try:
            # Fetch both timeframes
            klines_1m = self.exchange.get_kline(category="linear", symbol=self.symbol, interval="1", limit=200)
            klines_15m = self.exchange.get_kline(category="linear", symbol=self.symbol, interval="15", limit=100)
            
            if klines_1m.get('retCode') != 0 or klines_15m.get('retCode') != 0:
                return False
            
            # Process data
            self.price_data_1m = self._process_kline_data(klines_1m['result']['list'])
            self.price_data_15m = self._process_kline_data(klines_15m['result']['list'])
            
            # Validate data quality
            return (len(self.price_data_1m) > 50 and len(self.price_data_15m) > 30 and
                   not self.price_data_1m['close'].isna().any() and
                   not self.price_data_15m['close'].isna().any())
        except:
            return False
    
    def _process_kline_data(self, kline_list):
        """Process kline data into DataFrame"""
        try:
            if not kline_list:
                return pd.DataFrame()
                
            df = pd.DataFrame(kline_list, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
            ])
            
            if df.empty:
                return df
            
            # Convert types
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Clean and sort (pandas 2.1+ compatible)
            df = df.ffill().sort_values('timestamp').set_index('timestamp')
            return df if not df.empty else pd.DataFrame()
            
        except:
            return pd.DataFrame()
    
    async def _generate_and_execute_signal(self):
        """Generate and execute trading signals"""
        # Select strategy based on market conditions
        strategy_type, market_info = self.strategy_manager.select_strategy(
            self.price_data_1m, self.price_data_15m
        )
        
        self.market_info = market_info
        
        # Handle strategy switch
        if self.active_strategy and self.active_strategy != strategy_type:
            await self._handle_strategy_switch()
        
        # Update risk manager
        if self.active_strategy != strategy_type:
            self.risk_manager.set_strategy(strategy_type)
            self.risk_manager.adapt_to_market_condition(
                market_info['condition'], market_info.get('volatility', 'NORMAL')
            )
        
        self.active_strategy = strategy_type
        
        # Generate and execute signal
        signal = self._generate_signal(strategy_type, market_info)
        
        if signal and self._validate_signal(signal, market_info):
            self.rejections['total_signals'] += 1
            await self._execute_trade(signal, strategy_type)
    
    def _generate_signal(self, strategy_type, market_info):
        """Generate signal using the appropriate strategy"""
        try:
            if strategy_type == "RANGE":
                return self.range_strategy.generate_signal(self.price_data_1m, market_info['condition'])
            else:  # TREND
                return self.trend_strategy.generate_signal(self.price_data_15m, market_info['condition'])
        except:
            return None
    
    def _validate_signal(self, signal, market_info):
        """Streamlined signal validation"""
        return (signal and market_info['condition'] != 'INSUFFICIENT_DATA' and 
                market_info['confidence'] >= 0.6 and signal.get('confidence', 0) >= 60)
    
    async def _execute_trade(self, signal, strategy_type):
        """Execute trade with fee consideration"""
        current_price = float(self.price_data_1m['close'].iloc[-1])
        balance = await self.get_account_balance()
        
        if not balance:
            return
        
        # Calculate position size
        base_qty = self.risk_manager.calculate_position_size(balance, current_price, signal['structure_stop'])
        sizing_multiplier = self.strategy_manager.get_position_sizing_multiplier(strategy_type, self.market_info)
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
                self._log_trade("ENTRY", current_price, signal=signal, quantity=formatted_qty, strategy=strategy_type)
                await self.notifier.send_trade_entry(signal, current_price, formatted_qty, self._get_strategy_info())
        except:
            pass
    
    async def _check_position_status(self):
        """Check position status"""
        try:
            positions = self.exchange.get_positions(category="linear", symbol=self.symbol)
            if positions.get('retCode') != 0:
                return
            
            pos_list = positions['result']['list']
            has_position = pos_list and float(pos_list[0]['size']) > 0
            
            if not has_position:
                if self.position:
                    await self._on_position_closed()
                self.position = None
                self.position_start_time = None
            else:
                if not self.position:
                    self.position_start_time = datetime.now()
                self.position = pos_list[0]
        except:
            pass
    
    async def _check_position_exit(self):
        """Check if position should be closed"""
        if not self.position or not self.position_start_time:
            return
        
        current_price = float(self.price_data_1m['close'].iloc[-1])
        entry_price = float(self.position.get('avgPrice', 0))
        side = self.position.get('side', '')
        unrealized_pnl = float(self.position.get('unrealisedPnl', 0))
        position_age = (datetime.now() - self.position_start_time).total_seconds()
        
        # Sync risk manager
        if self.risk_manager.active_strategy != self.active_strategy:
            self.risk_manager.set_strategy(self.active_strategy)
        
        should_close, reason = self.risk_manager.should_close_position(
            current_price, entry_price, side, unrealized_pnl, position_age
        )
        
        if should_close:
            await self._close_position(reason)
    
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
                
                # Track exit reason
                self.exit_reasons[reason] = self.exit_reasons.get(reason, 0) + 1
                
                self._log_trade("EXIT", current_price, reason=reason, pnl=pnl, strategy=self.active_strategy)
                
                exit_data = {'trigger': reason, 'strategy': self.active_strategy}
                await self.notifier.send_trade_exit(exit_data, current_price, pnl, duration, self._get_strategy_info())
        except:
            pass
    
    async def _handle_strategy_switch(self):
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
                self.position = None
                self.position_start_time = None
    
    async def _on_position_closed(self):
        """Handle position closed externally"""
        if self.position:
            pnl = float(self.position.get('unrealisedPnl', 0))
            price = float(self.price_data_1m['close'].iloc[-1]) if len(self.price_data_1m) > 0 else 0
            self.exit_reasons['position_closed'] = self.exit_reasons.get('position_closed', 0) + 1
            self._log_trade("EXIT", price, reason="position_closed", pnl=pnl, strategy=self.active_strategy)
    
    def _log_trade(self, action, price, **kwargs):
        """Streamlined trade logging"""
        timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
        
        if action == "ENTRY":
            self.trade_id += 1
            signal = kwargs.get('signal', {})
            log_data = {
                'timestamp': timestamp, 'id': self.trade_id, 'action': 'ENTRY',
                'strategy': kwargs.get('strategy', 'UNKNOWN'), 'side': signal.get('action', ''),
                'price': round(price, 2), 'size': kwargs.get('quantity', ''),
                'condition': self.market_info.get('condition', ''), 'adx': round(self.market_info.get('adx', 0), 1),
                'confidence': round(signal.get('confidence', 0), 1)
            }
        else:
            duration = (datetime.now() - self.position_start_time).total_seconds() if self.position_start_time else 0
            log_data = {
                'timestamp': timestamp, 'id': self.trade_id, 'action': 'EXIT',
                'strategy': kwargs.get('strategy', 'UNKNOWN'), 'trigger': kwargs.get('reason', '').replace(' ', '_'),
                'price': round(price, 2), 'pnl': round(kwargs.get('pnl', 0), 2), 'duration': round(duration, 1)
            }
        
        try:
            with open(self.log_file, "a") as f:
                f.write(f"{log_data}\n")
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
        """Ultra-streamlined status display"""
        try:
            price = float(self.price_data_1m['close'].iloc[-1])
            time = self.price_data_1m.index[-1].strftime('%H:%M:%S')
            
            print("\n" * 50 + "=" * 60)
            print(f"⚡ {self.symbol} DUAL-STRATEGY BOT")
            print("=" * 60)
            
            # Market & Strategy
            condition = self.market_info.get('condition', 'UNKNOWN')
            adx = self.market_info.get('adx', 0)
            confidence = self.market_info.get('confidence', 0)
            
            print(f"📊 {condition} | ADX: {adx:.1f} | Conf: {confidence*100:.0f}% | Strategy: {self.active_strategy or 'NONE'}")
            
            # Position or Scanner Status
            if self.position:
                pnl = float(self.position.get('unrealisedPnl', 0))
                entry = float(self.position.get('avgPrice', 0))
                side = self.position.get('side', '')
                age = (datetime.now() - self.position_start_time).total_seconds() if self.position_start_time else 0
                
                emoji = "🟢" if side == "Buy" else "🔴"
                print(f"{emoji} {side}: ${entry:.2f} | PnL: ${pnl:.2f} | Age: {age:.0f}s")
            else:
                print("⚡ Scanner Active - No Position")
            
            print(f"💰 {time} | ${price:.2f}")
            
            # Quick stats
            total_trades = sum(self.exit_reasons.values())
            total_signals = self.rejections.get('total_signals', 0)
            if total_trades > 0 or total_signals > 0:
                rate = (total_trades / max(total_signals, 1)) * 100
                print(f"📈 Trades: {total_trades} | Rate: {rate:.1f}%")
            
            print("-" * 60)
            
        except Exception as e:
            print(f"❌ Display error: {e}")