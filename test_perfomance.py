#!/usr/bin/env python3
"""
Debug why the test shows unprofitable when math says profitable
"""

from core.fee_calculator import FeeCalculator

def debug_math():
    """Show exact calculations step by step"""
    
    print("\n" + "="*70)
    print("DEBUGGING PROFITABILITY CALCULATIONS")
    print("="*70)
    
    fee_calc = FeeCalculator()
    
    # Test the exact scenario from the test
    print("\n🔍 EXACT TEST SCENARIO: RANGE $2000")
    print("-" * 50)
    
    position_size = 2000
    risk_pct = 0.02  # 2%
    reward_ratio = 2.0  # 2R
    
    print(f"Position: ${position_size}")
    print(f"Risk: {risk_pct*100}% = ${position_size * risk_pct}")
    print(f"Reward: {risk_pct*reward_ratio*100}% = ${position_size * risk_pct * reward_ratio}")
    
    # What the test shows
    min_profit = fee_calc.get_minimum_profit_target(position_size)
    print(f"\nMin Profit Target (from fee_calc): ${min_profit:.2f}")
    print(f"  This is NOT the profit per win!")
    print(f"  This is minimum to cover fees + margin")
    
    # Actual P&L calculation
    print(f"\n📊 ACTUAL P&L AT 50% WIN RATE:")
    
    wins = 50
    losses = 50
    
    # Per trade
    profit_per_win = position_size * risk_pct * reward_ratio  # $80
    loss_per_loss = position_size * risk_pct  # $40
    fee_per_trade = position_size * 0.008  # $16
    
    print(f"\nPer Trade:")
    print(f"  Win: ${profit_per_win:.2f}")
    print(f"  Loss: ${loss_per_loss:.2f}")
    print(f"  Fee: ${fee_per_trade:.2f}")
    
    # Totals
    total_gross_wins = profit_per_win * wins
    total_gross_losses = loss_per_loss * losses
    total_fees = fee_per_trade * 100
    
    print(f"\nFor 100 Trades:")
    print(f"  Gross Wins: {wins} × ${profit_per_win:.0f} = ${total_gross_wins:.2f}")
    print(f"  Gross Losses: {losses} × ${loss_per_loss:.0f} = ${total_gross_losses:.2f}")
    print(f"  Total Fees: 100 × ${fee_per_trade:.2f} = ${total_fees:.2f}")
    
    net_pnl = total_gross_wins - total_gross_losses - total_fees
    
    print(f"\nNET P&L: ${total_gross_wins:.0f} - ${total_gross_losses:.0f} - ${total_fees:.0f} = ${net_pnl:.2f}")
    
    if net_pnl > 0:
        print(f"✅ PROFITABLE!")
    else:
        print(f"❌ UNPROFITABLE (This shouldn't happen!)")
    
    # Show the WRONG calculation (what the test might be doing)
    print(f"\n⚠️  WRONG CALCULATION (if using min_profit):")
    wrong_calc = (50 * min_profit) - (50 * (position_size * risk_pct))
    print(f"  (50 × ${min_profit:.0f}) - (50 × ${position_size * risk_pct:.0f}) = ${wrong_calc:.2f}")
    print(f"  This is WRONG because min_profit is not the profit per win!")
    
    # All position sizes
    print("\n" + "="*70)
    print("ALL POSITION SIZES - CORRECT MATH:")
    print("="*70)
    
    scenarios = [
        ("RANGE", 2000, 0.02, 2.0),
        ("TREND", 2500, 0.02, 2.5),
        ("RANGE", 1500, 0.02, 2.0),
        ("TREND", 3000, 0.02, 2.5),
    ]
    
    for strategy, pos_size, risk, rr in scenarios:
        profit_per_win = pos_size * risk * rr
        loss_per_loss = pos_size * risk
        fee_per_trade = pos_size * 0.008
        
        # At 50% win rate
        net = (50 * profit_per_win) - (50 * loss_per_loss) - (100 * fee_per_trade)
        
        status = "✅" if net > 0 else "❌"
        print(f"{strategy} ${pos_size}: {status} ${net:.2f} net profit")
        print(f"  Formula: (50×${profit_per_win:.0f}) - (50×${loss_per_loss:.0f}) - (100×${fee_per_trade:.1f})")
    
    print("\n✅ ALL positions should be PROFITABLE at 50% win rate!")
    print("If test shows otherwise, the test calculation is WRONG.")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    debug_math()