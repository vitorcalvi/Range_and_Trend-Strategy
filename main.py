#!/usr/bin/env python3
"""
Fixed Bybit Trading Bot with Correct Fee Calculations
3-minute timeframe for stress testing
"""

import asyncio
import signal
import sys
from datetime import datetime
from core.trade_engine import TradeEngine
from core.risk_manager import RiskManager
from core.fee_calculator import FeeCalculator

class TradingBot:
    def __init__(self):
        self.engine = TradeEngine()
        self.running = False
        
    async def start(self):
        """Start the trading bot"""
        print("\n" + "="*60)
        print("BYBIT TRADING BOT - FIXED VERSION")
        print("="*60)
        
        # Display corrected fee model
        fee_calc = FeeCalculator()
        print(f"\n📊 CORRECTED FEE MODEL:")
        print(f"   Spot Fees: {float(fee_calc.taker_fee_rate)*100:.1f}% per side")
        print(f"   Slippage: {float(fee_calc.slippage_rate)*100:.1f}% estimated")
        print(f"   Round Trip: {float(fee_calc.round_trip_cost)*100:.1f}% total")
        
        # Display risk parameters
        risk_mgr = RiskManager()
        config = risk_mgr.get_config_summary()
        print(f"\n⚙️  RISK PARAMETERS:")
        print(f"   Range Strategy: {config['breakeven_winrate']}% breakeven | Target: ${config['minimum_target']}")
        print(f"   Required Move: {config['required_move_pct']}% to profit")
        print(f"   Risk/Reward: {config['risk_reward']}")
        
        # Connect to exchange
        if not self.engine.connect():
            print("\n❌ Failed to connect to exchange")
            return
        
        # Get initial balance
        balance = await self.engine.get_account_balance()
        print(f"\n💰 Account Balance: ${balance:.2f}")
        
        if balance < 1000:
            print("⚠️  Warning: Low balance for testing")
        
        print(f"\n🚀 Starting 3-minute stress test mode...")
        print("   Press Ctrl+C to stop\n")
        print("-"*60)
        
        self.running = True
        
        # Main trading loop
        while self.running:
            try:
                await self.engine.run_cycle()
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                await asyncio.sleep(5)
        
        await self.shutdown()
    
    async def shutdown(self):
        """Graceful shutdown"""
        print("\n\n" + "="*60)
        print("SHUTTING DOWN...")
        self.running = False
        
        # Close any open position
        if self.engine.position:
            print("Closing open position...")
            await self.engine._close_position("shutdown")
            await asyncio.sleep(2)
        
        # Display summary
        print(f"\n📊 TRADING SUMMARY:")
        print(f"   Total Trades: {self.engine.trade_count}")
        
        if self.engine.exit_reasons:
            print(f"\n   Exit Reasons:")
            for reason, count in self.engine.exit_reasons.items():
                print(f"     {reason}: {count}")
        
        print("\n✅ Shutdown complete")
        print("="*60)
    
    def stop(self):
        """Stop the bot"""
        self.running = False

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    raise KeyboardInterrupt

def main():
    """Main entry point"""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    bot = TradingBot()
    
    try:
        asyncio.run(bot.start())
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()