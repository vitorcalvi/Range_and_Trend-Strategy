import asyncio
import signal
import sys
import os
from datetime import datetime
from dotenv import load_dotenv
from core.trade_engine import TradeEngine

load_dotenv()

class DualStrategyBot:
    """Streamlined Dual Strategy Trading Bot"""
    
    def __init__(self):
        self.engine = TradeEngine()
        self.running = False
        
    async def start(self):
        """Start the bot"""
        if not self._validate_environment():
            return
            
        if not self.engine.connect():
            print("❌ Failed to connect to exchange")
            return
        
        await self._startup()
        self.running = True
        
        print("🚀 Dual Strategy Bot Active - Scanning markets...")
        
        while self.running:
            try:
                await self.engine.run_cycle()
                await asyncio.sleep(0.5)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Cycle Error: {e}")
                await asyncio.sleep(5)
        
        await self._shutdown()
    
    def _validate_environment(self):
        """Validate environment"""
        required_vars = ['TRADING_SYMBOL', 'DEMO_MODE', 'TESTNET_BYBIT_API_KEY', 'TESTNET_BYBIT_API_SECRET']
        missing = [var for var in required_vars if not os.getenv(var)]
        
        if missing:
            print(f"❌ Missing environment variables: {', '.join(missing)}")
            return False
        return True
    
    async def _startup(self):
        """Display startup info"""
        balance = await self.engine.get_account_balance()
        demo_mode = "TESTNET" if self.engine.demo_mode else "LIVE"
        
        print(f"\n🚀 {self.engine.symbol} DUAL STRATEGY BOT")
        print("=" * 60)
        print(f"🌐 Environment: {demo_mode}")
        print(f"💰 Balance: ${balance:,.2f} USDT")
        print(f"📊 Symbol: {self.engine.symbol}")
        print(f"🧠 Auto-switching: Range ↔ Trend (ADX-based)")
        print(f"💡 Optimized: $5K Range | $8K Trend (min exposure, max profit)")
        print("=" * 60)
        
        await self.engine.notifier.send_bot_status("started", "Optimized Dual Strategy System Active")
    
    async def _shutdown(self):
        """Shutdown gracefully"""
        print("\n🛑 Shutting down...")
        self.running = False
        
        if self.engine.position:
            await self.engine._close_position("Bot shutdown")
        
        self._show_stats()
        await self.engine.notifier.send_bot_status("stopped", "Bot safely shutdown")
        print("✅ Bot stopped")
    
    def _show_stats(self):
        """Show session stats"""
        try:
            print(f"\n📊 SESSION STATS")
            print("-" * 40)
            
            total_trades = sum(self.engine.exit_reasons.values())
            total_signals = self.engine.rejections.get('total_signals', 0)
            
            if total_trades > 0:
                print(f"🔢 Total Trades: {total_trades}")
                for reason, count in sorted(self.engine.exit_reasons.items(), key=lambda x: x[1], reverse=True):
                    if count > 0:
                        print(f"   • {reason.replace('_', ' ').title()}: {count}")
            
            if total_signals > 0:
                acceptance_rate = (total_trades / total_signals) * 100
                print(f"📈 Signals: {total_signals} | Rate: {acceptance_rate:.1f}%")
            
            if total_trades == 0 and total_signals == 0:
                print("📊 No trades this session")
                
        except Exception as e:
            print(f"❌ Stats error: {e}")

def _signal_handler(signum, frame):
    """Handle shutdown signals"""
    raise KeyboardInterrupt

def main():
    """Main entry point"""
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    
    print("⚡ Initializing Dual Strategy Bot...")
    
    try:
        bot = DualStrategyBot()
        asyncio.run(bot.start())
    except KeyboardInterrupt:
        print("\n👋 Bot stopped by user")
    except Exception as e:
        print(f"❌ Critical error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()