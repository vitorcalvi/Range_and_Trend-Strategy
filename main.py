import asyncio
import signal
import sys
import os
import json
from datetime import datetime
from dotenv import load_dotenv
from core.trade_engine import TradeEngine

load_dotenv()

class UltraAggressiveStressTestBot:
    """ULTRA-AGGRESSIVE 3-Minute Trading Bot for Stress Testing"""
    
    def __init__(self):
        self.engine = TradeEngine()
        self.running = False
        self.stress_test_duration = 3600  # 1 hour stress test
        self.stress_test_start = None
        
    async def start_stress_test(self):
        """Start ultra-aggressive stress test"""
        if not self._validate_environment():
            return
            
        if not self.engine.connect():
            print("❌ Failed to connect to exchange")
            return
        
        await self._startup_stress_test()
        self.running = True
        self.stress_test_start = datetime.now()
        
        print("🔥 ULTRA-AGGRESSIVE STRESS TEST STARTED")
        print(f"   Duration: {self.stress_test_duration/60:.0f} minutes")
        print(f"   Expected signals: 30-80 per hour")
        print(f"   Position size: $1,500")
        print(f"   Profit targets: $8-$20")
        print("   ⚠️  FOR TESTING ONLY - MONITOR CLOSELY")
        
        # Ultra-fast execution loop
        cycle_count = 0
        while self.running:
            try:
                # Check if stress test time limit reached
                if self._should_stop_stress_test():
                    print(f"⏰ Stress test time limit reached ({self.stress_test_duration/60:.0f} minutes)")
                    break
                
                await self.engine.run_cycle()
                cycle_count += 1
                
                # Ultra-short delay for stress testing
                await asyncio.sleep(0.1)  # Only 100ms between cycles!
                
                # Periodic status updates
                if cycle_count % 50 == 0:
                    await self._display_stress_metrics()
                
            except KeyboardInterrupt:
                print("\n⚠️  Stress test interrupted by user")
                break
            except Exception as e:
                print(f"❌ Stress test error: {e}")
                await asyncio.sleep(1)  # Brief pause on error
        
        await self._shutdown_stress_test()
    
    def _should_stop_stress_test(self) -> bool:
        """Check if stress test should stop"""
        if not self.stress_test_start:
            return False
        
        elapsed = (datetime.now() - self.stress_test_start).total_seconds()
        return elapsed >= self.stress_test_duration
    
    def _validate_environment(self):
        """Validate environment for stress testing"""
        required_vars = ['TRADING_SYMBOL', 'DEMO_MODE', 'TESTNET_BYBIT_API_KEY', 'TESTNET_BYBIT_API_SECRET']
        missing = [var for var in required_vars if not os.getenv(var)]
        
        if missing:
            print(f"❌ Missing: {', '.join(missing)}")
            return False
        
        # Ensure we're in demo mode for stress testing
        demo_mode = os.getenv('DEMO_MODE', 'false').lower()
        if demo_mode != 'true':
            print("❌ STRESS TEST REQUIRES DEMO_MODE=true")
            print("   Set DEMO_MODE=true in .env file for safety")
            return False
        
        return True
    
    async def _startup_stress_test(self):
        """Startup for stress test"""
        balance = await self.engine.get_account_balance()
        
        print(f"\n🔥 ULTRA-AGGRESSIVE STRESS TEST INITIALIZATION")
        print("=" * 70)
        print(f"🌐 Environment: TESTNET (REQUIRED)")
        print(f"💰 Balance: ${balance:,.2f} USDT")
        print(f"📊 Symbol: {self.engine.symbol}")
        print(f"🎯 Position Size: $1,500 (stress test)")
        print(f"⚡ Targets: $8-$20 micro-profits")
        print(f"⏱️  Max Hold: 3-5 minutes")
        print(f"🔄 Cooldowns: 5-10 seconds")
        print(f"📈 Expected Frequency: 30-80 signals/hour")
        print("=" * 70)
        
        # Safety checks
        if balance < 5000:
            print("⚠️  WARNING: Low balance for stress testing")
            print("   Recommended: >$5,000 for comprehensive testing")
        
        print("\n🔥 ULTRA-AGGRESSIVE PARAMETERS ACTIVE:")
        range_info = self.engine.range_strategy.get_strategy_info()
        trend_info = self.engine.trend_strategy.get_strategy_info()
        
        print(f"   Range: RSI({range_info['config']['rsi_length']}) + BB({range_info['config']['bb_length']}) | Target: ${range_info['config']['base_profit_usdt']}")
        print(f"   Trend: EMA({trend_info['config']['fast_ema']}/{trend_info['config']['slow_ema']}) + RSI({trend_info['config']['rsi_length']}) | Hold: {trend_info['config']['max_hold_seconds']}s")
        
        await self.engine.notifier.send_bot_status("started", "ULTRA-AGGRESSIVE STRESS TEST Active")
    
    async def _display_stress_metrics(self):
        """Display stress test metrics"""
        summary = self.engine.get_stress_test_summary()
        
        print(f"\n📊 STRESS TEST METRICS (Runtime: {summary['runtime_minutes']:.1f}m)")
        print(f"   Cycles: {summary['cycles_total']} ({summary['cycles_per_minute']:.1f}/min)")
        print(f"   API Calls: {summary['api_calls_total']} ({summary['api_calls_per_minute']:.1f}/min)")
        print(f"   Signals: {summary['signals_total']} | Success: {summary['success_rate']:.1f}%")
        print(f"   Trades: {summary['trades_executed']} | Switches: {summary['strategy_switches']}")
        
        if summary['data_fetch_failures'] > 0:
            print(f"   ⚠️  Failures: {summary['data_fetch_failures']} data fetch errors")
    
    async def _shutdown_stress_test(self):
        """Shutdown stress test with comprehensive reporting"""
        print("\n🛑 STOPPING ULTRA-AGGRESSIVE STRESS TEST...")
        self.running = False
        
        # Close any open positions
        if self.engine.position:
            await self.engine._close_position_stress_test("Stress test complete")
            await asyncio.sleep(2)  # Allow time for position close
        
        # Generate comprehensive stress test report
        await self._generate_stress_test_report()
        
        await self.engine.notifier.send_bot_status("stopped", "Ultra-Aggressive Stress Test Complete")
        print("✅ Stress test completed successfully")
    
    async def _generate_stress_test_report(self):
        """Generate comprehensive stress test report"""
        summary = self.engine.get_stress_test_summary()
        
        print(f"\n" + "=" * 80)
        print("📋 ULTRA-AGGRESSIVE STRESS TEST FINAL REPORT")
        print("=" * 80)
        
        # Overall performance
        print(f"⏱️  EXECUTION PERFORMANCE:")
        print(f"   Total Runtime: {summary['runtime_minutes']:.1f} minutes")
        print(f"   Cycles Executed: {summary['cycles_total']} ({summary['cycles_per_minute']:.1f}/min)")
        print(f"   API Calls Made: {summary['api_calls_total']} ({summary['api_calls_per_minute']:.1f}/min)")
        
        # Signal performance
        print(f"\n📊 SIGNAL PERFORMANCE:")
        print(f"   Total Signals: {summary['signals_total']}")
        print(f"   Successful: {summary['signals_successful']} ({summary['success_rate']:.1f}%)")
        print(f"   Failed: {summary['signals_failed']}")
        print(f"   Expected Rate: 30-80/hour | Actual: {summary['signals_total']/(summary['runtime_minutes']/60):.1f}/hour")
        
        # Trading performance
        print(f"\n💼 TRADING PERFORMANCE:")
        print(f"   Trades Executed: {summary['trades_executed']}")
        print(f"   Strategy Switches: {summary['strategy_switches']}")
        print(f"   Position Checks: {summary['position_checks']}")
        
        # Exit analysis
        if summary['exit_reasons']:
            print(f"\n🚪 EXIT ANALYSIS:")
            total_exits = sum(summary['exit_reasons'].values())
            for reason, count in sorted(summary['exit_reasons'].items(), key=lambda x: x[1], reverse=True):
                if count > 0:
                    percentage = (count / total_exits) * 100
                    print(f"   {reason.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
        
        # Error analysis
        print(f"\n⚠️  ERROR ANALYSIS:")
        print(f"   Data Fetch Failures: {summary['data_fetch_failures']}")
        error_types = ['api_rate_limited', 'execution_failed', 'data_validation_failed']
        for error_type in error_types:
            count = summary['rejections'].get(error_type, 0)
            if count > 0:
                print(f"   {error_type.replace('_', ' ').title()}: {count}")
        
        # Risk metrics
        risk_metrics = summary.get('risk_metrics', {})
        if risk_metrics:
            print(f"\n🎯 RISK MANAGEMENT:")
            print(f"   Emergency Stops: {risk_metrics.get('emergency_stops', 0)} ({risk_metrics.get('emergency_stop_rate', '0%')})")
            print(f"   Quick Exits: {risk_metrics.get('quick_exits', 0)} ({risk_metrics.get('quick_exit_rate', '0%')})")
            print(f"   Micro Profits: {risk_metrics.get('micro_profits', 0)} ({risk_metrics.get('micro_profit_rate', '0%')})")
        
        # System stress assessment
        print(f"\n🔥 STRESS TEST ASSESSMENT:")
        
        if summary['cycles_per_minute'] > 50:
            print(f"   ✅ High Frequency: {summary['cycles_per_minute']:.1f} cycles/min (Target: >50)")
        else:
            print(f"   ⚠️  Low Frequency: {summary['cycles_per_minute']:.1f} cycles/min (Target: >50)")
        
        if summary['api_calls_per_minute'] < 100:
            print(f"   ✅ API Load: {summary['api_calls_per_minute']:.1f} calls/min (Sustainable)")
        else:
            print(f"   ⚠️  High API Load: {summary['api_calls_per_minute']:.1f} calls/min (Risk of rate limiting)")
        
        if summary['success_rate'] > 70:
            print(f"   ✅ Signal Quality: {summary['success_rate']:.1f}% success rate")
        else:
            print(f"   ⚠️  Signal Issues: {summary['success_rate']:.1f}% success rate (Target: >70%)")
        
        # Save report to file
        report_file = f"logs/stress_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(report_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            print(f"\n📄 Detailed report saved: {report_file}")
        except Exception as e:
            print(f"❌ Failed to save report: {e}")
        
        print("=" * 80)
        
        # Final recommendation
        if (summary['cycles_per_minute'] > 50 and 
            summary['success_rate'] > 70 and 
            summary['data_fetch_failures'] < 10):
            print("🎉 STRESS TEST PASSED - Bot performed well under ultra-aggressive conditions")
        else:
            print("⚠️  STRESS TEST ISSUES - Review performance metrics before live deployment")

def _signal_handler(signum, frame):
    """Handle signals"""
    raise KeyboardInterrupt

def main():
    """Main stress test entry point"""
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    
    print("🔥 INITIALIZING ULTRA-AGGRESSIVE STRESS TEST BOT...")
    print("   ⚠️  WARNING: This is an extreme stress test configuration")
    print("   📊 Expected: 30-80 signals per hour")
    print("   💰 Micro-profits: $8-$20 targets")
    print("   ⏱️  Ultra-fast: 5-10 second cooldowns")
    print("   🎯 FOR TESTING SYSTEM LIMITS ONLY")
    
    try:
        bot = UltraAggressiveStressTestBot()
        asyncio.run(bot.start_stress_test())
    except KeyboardInterrupt:
        print("\n👋 Stress test stopped by user")
    except Exception as e:
        print(f"❌ Critical stress test error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()