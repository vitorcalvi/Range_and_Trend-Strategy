import asyncio
import signal
import sys
import os
from datetime import datetime
from dotenv import load_dotenv
from core.trade_engine import TradeEngine

load_dotenv()

class DualStrategyBot:
    """Dual Strategy Trading Bot - Auto-switches between Range and Trend strategies"""
    
    def __init__(self):
        self.engine = TradeEngine()
        self.running = False
        
    async def start(self):
        """Start the dual strategy trading bot"""
        if not self._validate_environment():
            return
            
        if not self.engine.connect():
            print("❌ Failed to connect to exchange")
            return
        
        await self._startup()
        self.running = True
        
        print("🚀 Dual Strategy Bot is now active - Scanning markets...")
        
        while self.running:
            try:
                await self.engine.run_cycle()
                await asyncio.sleep(0.5)  # 500ms cycle
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Cycle Error: {e}")
                print(f"💡 Attempting to continue... (retry in 5s)")
                
                try:
                    with open("logs/errors.log", "a") as f:
                        f.write(f"{datetime.now()}: {str(e)}\n")
                except:
                    pass
                    
                await asyncio.sleep(5)
        
        await self._shutdown()
    
    def _validate_environment(self):
        """Validate environment configuration"""
        required_vars = [
            'TRADING_SYMBOL', 'DEMO_MODE', 
            'TESTNET_BYBIT_API_KEY', 'TESTNET_BYBIT_API_SECRET'
        ]
        
        missing = [var for var in required_vars if not os.getenv(var)]
        if missing:
            print(f"❌ Missing environment variables: {', '.join(missing)}")
            print("📝 Check your .env file")
            return False
        
        return True
    
    async def _startup(self):
        """Display startup info for dual strategy system"""
        balance = await self.engine.get_account_balance()
        
        print(f"\n🚀 {self.engine.symbol} DUAL STRATEGY TRADING BOT")
        print("=" * 70)
        
        # Environment
        demo_mode = "TESTNET" if self.engine.demo_mode else "LIVE"
        print(f"🌐 Environment: {demo_mode}")
        print(f"💰 Account Balance: ${balance:,.2f} USDT")
        print(f"📊 Symbol: {self.engine.symbol}")
        
        # Dual Strategy Overview
        print(f"\n🧠 DUAL STRATEGY SYSTEM")
        print("-" * 70)
        print("📈 Auto-switching based on market conditions using ADX indicator")
        print("⚡ Real-time market analysis with 1m + 15m timeframe monitoring")
        
        # Strategy configurations
        range_info = self.engine.range_strategy.get_strategy_info()
        trend_info = self.engine.trend_strategy.get_strategy_info()
        
        print(f"\n📊 STRATEGY 1: {range_info['name'].upper()}")
        print("-" * 70)
        print(f"🎯 Trigger: ADX < 25 (Range-bound markets)")
        print(f"⏱️  Timeframe: {range_info['timeframe']} (scalping)")
        print(f"💰 Position Size: ${self.engine.risk_manager.range_config['fixed_position_usdt']:,} USDT")
        print(f"🎯 Profit Target: ${range_info['config']['target_profit_usdt']} USDT")
        
        print(f"\n📈 STRATEGY 2: {trend_info['name'].upper()}")
        print("-" * 70)
        print(f"🎯 Trigger: ADX > 25 (Trending markets)")
        print(f"⏱️  Timeframe: {trend_info['timeframe']} (trend following)")
        print(f"💰 Position Size: ${self.engine.risk_manager.trend_config['fixed_position_usdt']:,} USDT")
        print(f"🎯 Risk-Reward: {trend_info['risk_reward']}")
        
        print("\n" + "=" * 70)
        print("🟢 Dual Strategy Bot initialized successfully")
        
        # Send Telegram notification
        await self.engine.notifier.send_bot_status("started", 
            "Dual Strategy System Active - Range + Trend strategies with ADX switching")
    
    async def _shutdown(self):
        """Shutdown bot gracefully"""
        print("\n🛑 Shutting down Dual Strategy Bot...")
        self.running = False
        
        # Close any open positions
        if self.engine.position:
            print("⚠️ Closing open position...")
            await self.engine._close_position("Bot shutdown")
        
        # Show final statistics
        self._show_session_stats()
        
        # Send shutdown notification
        await self.engine.notifier.send_bot_status("stopped", 
            "Dual Strategy Bot safely shutdown")
        print("✅ Bot stopped successfully")
    
    def _show_session_stats(self):
        """Show session statistics"""
        try:
            print(f"\n📊 SESSION STATISTICS")
            print("-" * 50)
            
            # Strategy manager stats
            strategy_info = self.engine.strategy_manager.get_strategy_info()
            if strategy_info['current_strategy']:
                print(f"🎯 Final Strategy: {strategy_info['current_strategy']}")
                print(f"📊 Market Condition: {strategy_info['market_condition'].get('condition', 'Unknown')}")
                print(f"📈 Final ADX: {strategy_info['market_condition'].get('adx', 0):.1f}")
            
            # Trading statistics
            exit_reasons = self.engine.exit_reasons
            rejections = self.engine.rejections
            
            total_trades = sum(exit_reasons.values())
            total_signals = rejections.get('total_signals', 0)
            
            if total_trades > 0:
                print(f"🔢 Total Trades: {total_trades}")
                sorted_exits = sorted(exit_reasons.items(), key=lambda x: x[1], reverse=True)
                for reason, count in sorted_exits:
                    if count > 0:
                        print(f"   • {reason.replace('_', ' ').title()}: {count}")
            
            if total_signals > 0:
                acceptance_rate = (total_trades / total_signals) * 100
                print(f"📈 Signal Processing:")
                print(f"   • Total Signals: {total_signals}")
                print(f"   • Acceptance Rate: {acceptance_rate:.1f}%")
            
            if total_trades == 0 and total_signals == 0:
                print("📊 No trades executed this session")
                
        except Exception as e:
            print(f"❌ Error generating session stats: {e}")

def _signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    raise KeyboardInterrupt

def main():
    """Main entry point for dual strategy bot"""
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    
    print("⚡ Initializing Dual Strategy Trading Bot...")
    print("🧠 Loading Range + Trend strategies with market detection...")
    
    try:
        bot = DualStrategyBot()
        asyncio.run(bot.start())
    except KeyboardInterrupt:
        print("\n👋 Bot stopped by user")
    except Exception as e:
        print(f"❌ Critical error: {e}")
        print(f"💡 Check your configuration and try again")
        sys.exit(1)

if __name__ == "__main__":
    main()