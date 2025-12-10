import os
import sys
import asyncio
import pandas as pd
import numpy as np
import json
from datetime import datetime
from typing import Dict, Any, Optional
from pybit.unified_trading import HTTP
from dotenv import load_dotenv

from strategies.strategy_manager import StrategyManager
from strategies.range_strategy import RangeStrategy
from strategies.trend_strategy import TrendStrategy
from core.risk_manager import RiskManager
from core.fee_calculator import FeeCalculator

load_dotenv()

class TradeEngine:
    """Simplified trading engine with correct fee calculations"""
    
    def __init__(self):
        # Core components
        self.strategy_manager = StrategyManager()
        self.range_strategy = RangeStrategy()
        self.trend_strategy = TrendStrategy()
        self.risk_manager = RiskManager()
        self.fee_calc = FeeCalculator()
        
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
        
        # Market data
        self.price_data_3m = pd.DataFrame()
        
        # Performance tracking
        self.trade_count = 0
        self.exit_reasons = {}
        
        # Symbol-specific rules
        self._set_symbol_rules()
        
        # Logging
        os.makedirs("logs", exist_ok=True)
        self.log_file = "logs/trades.log"
    
    def _set_symbol_rules(self):
        """Set symbol-specific trading rules"""
        if 'ETH' in self.symbol:
            self.qty_step = '0.01'
            self.min_qty = 0.01
        elif 'BTC' in self.symbol:
            self.qty_step = '0.001'
            self.min_qty = 0.001
        else:
            self.qty_step = '1'
            self.min_qty = 1.0
    
    def connect(self) -> bool:
        """Connect to exchange"""
        try:
            self.exchange = HTTP(
                demo=self.demo_mode,
                api_key=self.api_key,
                api_secret=self.api_secret
            )
            return self.exchange.get_server_time().get('retCode') == 0
        except Exception as e:
            print(f"Connection error: {e}")
            return False
    
    def format_quantity(self, qty: float) -> str:
        """Format quantity according to exchange rules"""
        if qty < self.min_qty:
            return "0"
        
        decimals = len(self.qty_step.split('.')[1]) if '.' in self.qty_step else 0
        qty_step_float = float(self.qty_step)
        rounded_qty = round(qty / qty_step_float) * qty_step_float
        
        return f"{rounded_qty:.{decimals}f}" if decimals > 0 else str(int(rounded_qty))
    
    async def run_cycle(self):
        """Main trading cycle"""
        # Update market data
        if not await self._update_market_data():
            return
        
        # Check position status
        await self._check_position_status()
        
        # Manage existing position or generate new signal
        if self.position:
            await self._check_position_exit()
        else:
            await self._generate_and_execute_signal()
        
        # Display status
        self._display_status()
        
        # Cycle delay
        await asyncio.sleep(1)
    
    async def _update_market_data(self) -> bool:
        """Update 3-minute market data"""
        try:
            # Fetch 3-minute klines
            klines = self.exchange.get_kline(
                category="linear",
                symbol=self.symbol,
                interval="3",
                limit=100
            )
            
            if klines.get('retCode') != 0:
                return False
            
            self.price_data_3m = self._process_kline_data(klines['result']['list'])
            return len(self.price_data_3m) > 30
            
        except Exception as e:
            print(f"Data update error: {e}")
            return False
    
    def _process_kline_data(self, kline_list) -> pd.DataFrame:
        """Process kline data into DataFrame"""
        if not kline_list:
            return pd.DataFrame()
        
        df = pd.DataFrame(kline_list, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
        ])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df.sort_values('timestamp').set_index('timestamp')
    
    async def _generate_and_execute_signal(self):
        """Generate and execute trading signal"""
        # Select strategy
        strategy_type, market_info = self.strategy_manager.select_strategy(self.price_data_3m)
        
        # Check trade cooldown
        if not market_info.get('trade_allowed', False):
            return
        
        # Set risk manager strategy
        self.risk_manager.set_strategy(strategy_type)
        
        # Generate signal
        if strategy_type == "RANGE":
            signal = self.range_strategy.generate_signal(self.price_data_3m, market_info['condition'])
        else:
            signal = self.trend_strategy.generate_signal(self.price_data_3m, market_info['condition'])
        
        if not signal:
            return
        
        # Validate and execute
        if signal['confidence'] >= 65:
            await self._execute_trade(signal, strategy_type, market_info)
    
    async def _execute_trade(self, signal: Dict, strategy_type: str, market_info: Dict):
        """Execute trade with correct position sizing"""
        current_price = float(self.price_data_3m['close'].iloc[-1])
        balance = await self.get_account_balance()
        
        if balance < 1000:  # Minimum balance check
            return
        
        # Calculate position size
        base_qty = self.risk_manager.calculate_position_size(
            balance, current_price, signal['structure_stop']
        )
        
        if base_qty <= 0:
            return
        
        # Apply market condition multiplier
        multiplier = self.strategy_manager.get_position_sizing_multiplier(market_info)
        qty = base_qty * multiplier
        
        formatted_qty = self.format_quantity(qty)
        if formatted_qty == "0":
            return
        
        # Calculate position value
        position_size_usdt = float(formatted_qty) * current_price
        
        # Validate profitability
        expected_profit = position_size_usdt * 0.02 * signal['risk_reward_ratio']
        if not self.fee_calc.validate_profitability(position_size_usdt, expected_profit):
            print(f"Trade rejected: Insufficient profit margin")
            return
        
        # Execute order
        success = await self._place_order(signal['action'], formatted_qty, current_price)
        
        if success:
            self.strategy_manager.record_trade()
            self.position_entry_price = current_price
            self.position_size_usdt = position_size_usdt
            self.position_start_time = datetime.now()
            self.trade_count += 1
            
            # Log trade
            self._log_trade("ENTRY", signal, formatted_qty, position_size_usdt)
            
            # Display execution
            breakeven = self.fee_calc.get_breakeven_pnl(position_size_usdt)
            min_target = self.fee_calc.get_minimum_profit_target(position_size_usdt)
            
            print(f"\n{'='*60}")
            print(f"✅ {signal['action']} {formatted_qty} @ ${current_price:.2f}")
            print(f"   Position: ${position_size_usdt:.0f} | Strategy: {strategy_type}")
            print(f"   Breakeven: ${breakeven:.2f} | Target: ${min_target:.2f}")
            print(f"   Confidence: {signal['confidence']}% | R/R: 1:{signal['risk_reward_ratio']}")
            print(f"{'='*60}")
    
    async def _place_order(self, side: str, quantity: str, price: float) -> bool:
        """Place order with limit-first approach"""
        order_side = "Buy" if side == "BUY" else "Sell"
        
        try:
            # Try limit order first
            limit_offset = 0.001  # 0.1%
            if order_side == "Buy":
                limit_price = price * (1 - limit_offset)
            else:
                limit_price = price * (1 + limit_offset)
            
            # Place limit order
            order = self.exchange.place_order(
                category="linear",
                symbol=self.symbol,
                side=order_side,
                orderType="Limit",
                qty=quantity,
                price=f"{limit_price:.2f}",
                timeInForce="IOC"
            )
            
            if order.get('retCode') == 0:
                return True
            
            # Fallback to market order
            order = self.exchange.place_order(
                category="linear",
                symbol=self.symbol,
                side=order_side,
                orderType="Market",
                qty=quantity,
                timeInForce="IOC"
            )
            
            return order.get('retCode') == 0
            
        except Exception as e:
            print(f"Order error: {e}")
            return False
    
    async def _check_position_status(self):
        """Check current position status"""
        try:
            positions = self.exchange.get_positions(
                category="linear",
                symbol=self.symbol
            )
            
            if positions.get('retCode') != 0:
                return
            
            pos_list = positions['result']['list']
            
            if not pos_list or float(pos_list[0]['size']) == 0:
                if self.position:
                    self._reset_position()
            else:
                self.position = pos_list[0]
                if not self.position_start_time:
                    self.position_start_time = datetime.now()
                
        except Exception as e:
            print(f"Position check error: {e}")
    
    async def _check_position_exit(self):
        """Check if position should be closed"""
        if not self.position or not self.position_start_time:
            return
        
        current_price = float(self.price_data_3m['close'].iloc[-1])
        entry_price = self.position_entry_price or float(self.position['avgPrice'])
        side = self.position['side']
        unrealized_pnl = float(self.position.get('unrealisedPnl', 0))
        position_age = (datetime.now() - self.position_start_time).total_seconds()
        
        # Check exit conditions
        should_close, reason = self.risk_manager.should_close_position(
            current_price, entry_price, side, unrealized_pnl,
            position_age, self.position_size_usdt
        )
        
        if should_close:
            await self._close_position(reason)
    
    async def _close_position(self, reason: str):
        """Close current position"""
        if not self.position:
            return
        
        side = "Sell" if self.position['side'] == "Buy" else "Buy"
        qty = self.format_quantity(float(self.position['size']))
        
        try:
            order = self.exchange.place_order(
                category="linear",
                symbol=self.symbol,
                side=side,
                orderType="Market",
                qty=qty,
                timeInForce="IOC",
                reduceOnly=True
            )
            
            if order.get('retCode') == 0:
                # Log exit
                unrealized_pnl = float(self.position.get('unrealisedPnl', 0))
                duration = (datetime.now() - self.position_start_time).total_seconds()
                
                self._log_exit(reason, unrealized_pnl, duration)
                self._track_exit_reason(reason)
                
                print(f"\n📤 CLOSED: {reason} | PnL: ${unrealized_pnl:.2f} | Duration: {duration:.0f}s")
                
                self._reset_position()
                
        except Exception as e:
            print(f"Close position error: {e}")
    
    def _reset_position(self):
        """Reset position state"""
        self.position = None
        self.position_start_time = None
        self.position_entry_price = None
        self.position_size_usdt = None
    
    def _track_exit_reason(self, reason: str):
        """Track exit reasons"""
        if reason not in self.exit_reasons:
            self.exit_reasons[reason] = 0
        self.exit_reasons[reason] += 1
    
    def _log_trade(self, action: str, signal: Dict, quantity: str, position_size: float):
        """Log trade entry"""
        try:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'id': self.trade_count,
                'action': action,
                'strategy': signal['strategy'],
                'side': signal['action'],
                'price': signal['price'],
                'size': quantity,
                'position_size_usdt': round(position_size, 2),
                'confidence': signal['confidence'],
                'rsi': signal.get('rsi', 0),
                'breakeven_fees': round(self.fee_calc.get_breakeven_pnl(position_size), 2),
                'minimum_target': round(self.fee_calc.get_minimum_profit_target(position_size), 2)
            }
            
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
                
        except Exception as e:
            print(f"Logging error: {e}")
    
    def _log_exit(self, reason: str, pnl: float, duration: float):
        """Log trade exit"""
        try:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'id': self.trade_count,
                'action': 'EXIT',
                'trigger': reason,
                'unrealized_pnl': round(pnl, 2),
                'duration_seconds': round(duration, 1),
                'breakeven_fees': round(self.fee_calc.get_breakeven_pnl(self.position_size_usdt), 2) if self.position_size_usdt else 0,
                'net_profit': round(pnl - self.fee_calc.get_breakeven_pnl(self.position_size_usdt), 2) if self.position_size_usdt else pnl
            }
            
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
                
        except Exception as e:
            print(f"Logging error: {e}")
    
    async def get_account_balance(self) -> float:
        """Get account balance"""
        try:
            balance = self.exchange.get_wallet_balance(accountType="UNIFIED")
            if balance.get('retCode') == 0:
                coins = balance['result']['list'][0]['coin']
                usdt = next((c for c in coins if c['coin'] == 'USDT'), None)
                return float(usdt['walletBalance']) if usdt else 0
        except:
            pass
        return 0
    
    def _display_status(self):
        """Display current status"""
        if not self.price_data_3m.empty:
            price = float(self.price_data_3m['close'].iloc[-1])
            print(f"\r{datetime.now().strftime('%H:%M:%S')} | {self.symbol}: ${price:.2f} | ", end='')
            
            if self.position:
                pnl = float(self.position.get('unrealisedPnl', 0))
                print(f"Position: {self.position['side']} | PnL: ${pnl:.2f}", end='')
            else:
                print(f"No position | Trades: {self.trade_count}", end='')
        
        sys.stdout.flush()