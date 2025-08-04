#!/usr/bin/env python3
"""
Trading Bot MCP Server - Streamlined trades.log analysis
"""

from fastmcp import FastMCP
import sys
import logging
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
mcp = FastMCP("trading-bot-data")

# Get project root directory (parent of _utils folder where this script is located)
PROJECT_ROOT = Path(__file__).parent.parent
TRADES_LOG_PATH = PROJECT_ROOT / "logs" / "trades.log"

def _read_trades() -> List[Dict[str, Any]]:
    """Read all trades from log file"""
    if not TRADES_LOG_PATH.exists():
        return []
    
    trades = []
    try:
        with TRADES_LOG_PATH.open('r') as f:
            for line in f:
                if line.strip():
                    try:
                        trades.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        logging.error(f"Error reading trades: {e}")
    
    return trades

def _get_complete_trades() -> List[Dict[str, Any]]:
    """Get matched ENTRY/EXIT trade pairs"""
    trades = _read_trades()
    complete_trades, open_trades = [], {}
    
    for trade in trades:
        trade_id = trade.get('id')
        if trade.get('action') == 'ENTRY':
            open_trades[trade_id] = trade
        elif trade.get('action') == 'EXIT' and trade_id in open_trades:
            entry = open_trades.pop(trade_id)
            complete_trades.append({
                'id': trade_id, 'entry': entry, 'exit': trade,
                'strategy': entry.get('strategy'), 'side': entry.get('side'),
                'entry_price': entry.get('price'), 'exit_price': trade.get('price'),
                'position_size_usdt': entry.get('position_size_usdt', 0),
                'break_even_fees': entry.get('break_even_fees', 0),
                'bybit_unrealized_pnl': trade.get('bybit_unrealized_pnl', 0),
                'estimated_net_profit': trade.get('estimated_net_profit', 0),
                'hold_seconds': trade.get('hold_seconds', 0),
                'exit_trigger': trade.get('trigger'),
                'entry_timestamp': entry.get('timestamp'),
                'exit_timestamp': trade.get('timestamp'),
                'market_condition': entry.get('market_condition'),
                'confidence': entry.get('confidence', 0)
            })
    
    return complete_trades

def _get_open_positions() -> Dict[str, Dict[str, Any]]:
    """Get unmatched ENTRY trades (open positions)"""
    trades = _read_trades()
    open_positions = {}
    
    for trade in trades:
        trade_id = trade.get('id')
        if trade.get('action') == 'ENTRY':
            open_positions[trade_id] = trade
        elif trade.get('action') == 'EXIT' and trade_id in open_positions:
            del open_positions[trade_id]
    
    return open_positions

def _format_trade_summary(trade: Dict[str, Any]) -> str:
    """Format single trade line summary"""
    net_profit = trade['estimated_net_profit']
    emoji = "🟢" if net_profit > 0 else "🔴" if net_profit < 0 else "⚪"
    time_str = trade['exit_timestamp'][:19].replace('T', ' ')
    return f"{emoji} #{trade['id']} {trade['strategy']} {trade['side']} | ${net_profit:.2f} net | {trade['hold_seconds']:.1f}s | {time_str} | {trade['exit_trigger']}"

def _format_trade_detail(trade: Dict[str, Any]) -> str:
    """Format detailed trade information"""
    entry_time = trade['entry_timestamp'][:19].replace('T', ' ')
    exit_time = trade['exit_timestamp'][:19].replace('T', ' ')
    net_profit = trade['estimated_net_profit']
    emoji = "🟢" if net_profit > 0 else "🔴" if net_profit < 0 else "⚪"
    
    return f"""{emoji} Trade #{trade['id']} | {trade['strategy']} | {trade['side']}
   📅 {entry_time} → {exit_time}
   💰 ${trade['entry_price']:.2f} → ${trade['exit_price']:.2f}
   📊 Gross: ${trade['bybit_unrealized_pnl']:.2f} | Fees: ${trade['break_even_fees']:.2f} | Net: ${net_profit:.2f}
   ⏱️  {trade['hold_seconds']:.1f}s | Exit: {trade['exit_trigger']}"""

def _calculate_stats(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate comprehensive statistics"""
    if not trades:
        return {}
    
    winners = [t for t in trades if t['estimated_net_profit'] > 0]
    losers = [t for t in trades if t['estimated_net_profit'] < 0]
    
    return {
        'total_trades': len(trades), 'winners': len(winners), 'losers': len(losers),
        'win_rate': (len(winners) / len(trades)) * 100,
        'total_net_pnl': sum(t['estimated_net_profit'] for t in trades),
        'total_gross_pnl': sum(t['bybit_unrealized_pnl'] for t in trades),
        'total_fees': sum(t['break_even_fees'] for t in trades),
        'avg_win': sum(t['estimated_net_profit'] for t in winners) / len(winners) if winners else 0,
        'avg_loss': sum(t['estimated_net_profit'] for t in losers) / len(losers) if losers else 0,
        'avg_hold_time': sum(t['hold_seconds'] for t in trades) / len(trades),
        'range_trades': len([t for t in trades if t['strategy'] == 'RANGE']),
        'trend_trades': len([t for t in trades if t['strategy'] == 'TREND'])
    }

def _filter_trades(trades: List[Dict[str, Any]], **filters) -> List[Dict[str, Any]]:
    """Apply multiple filters to trades"""
    filtered = trades
    
    if filters.get('strategy'):
        filtered = [t for t in filtered if t['strategy'].upper() == filters['strategy'].upper()]
    if filters.get('side'):
        filtered = [t for t in filtered if t['side'].upper() == filters['side'].upper()]
    if filters.get('min_profit') is not None:
        filtered = [t for t in filtered if t['estimated_net_profit'] >= filters['min_profit']]
    if filters.get('max_profit') is not None:
        filtered = [t for t in filtered if t['estimated_net_profit'] <= filters['max_profit']]
    if filters.get('exit_trigger'):
        filtered = [t for t in filtered if filters['exit_trigger'].lower() in t.get('exit_trigger', '').lower()]
    if filters.get('hours_back'):
        cutoff = datetime.now() - timedelta(hours=filters['hours_back'])
        filtered = [t for t in filtered if 
                   datetime.fromisoformat(t['exit_timestamp'].replace('Z', '+00:00')).replace(tzinfo=None) >= cutoff]
    
    return filtered

@mcp.tool()
def test_connection(message: str = "Hello from Trading MCP!") -> str:
    """Test MCP connection"""
    return f"✅ Trading MCP Server Active!\nMessage: {message}\nBot: Dual-Strategy System (Range+Trend)"

@mcp.tool()
def get_log_file_info() -> str:
    """Get log file information"""
    try:
        if not TRADES_LOG_PATH.exists():
            return f"❌ Log file not found: {TRADES_LOG_PATH}\nProject root: {PROJECT_ROOT}"
        
        stat = TRADES_LOG_PATH.stat()
        with TRADES_LOG_PATH.open('r') as f:
            line_count = sum(1 for _ in f)
        
        return f"""📊 Trading Log File Info:
📁 Path: {TRADES_LOG_PATH}
📏 Size: {stat.st_size / (1024 * 1024):.2f} MB
📝 Lines: {line_count}
🕐 Modified: {datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')}
✅ Status: Ready for analysis"""
    except Exception as e:
        return f"❌ Error reading log file: {e}"

@mcp.tool()
def debug_log_file() -> str:
    """Debug log file contents"""
    try:
        if not TRADES_LOG_PATH.exists():
            return f"""❌ Log file not found at: {TRADES_LOG_PATH}
📁 Project root: {PROJECT_ROOT}
📁 Current working directory: {os.getcwd()}
📁 Logs directory exists: {(PROJECT_ROOT / "logs").exists()}"""
        
        trades = _read_trades()
        if not trades:
            return f"❌ No trades found in log file\nFile exists but appears empty or corrupted"
        
        entries = [t for t in trades if t.get('action') == 'ENTRY']
        exits = [t for t in trades if t.get('action') == 'EXIT']
        open_positions = _get_open_positions()
        
        result = f"""🔍 LOG FILE DEBUG INFO
{'='*50}
📁 File: {TRADES_LOG_PATH}
📊 Total records: {len(trades)}
📈 ENTRY records: {len(entries)}
📉 EXIT records: {len(exits)}
🔄 Unmatched entries: {len(open_positions)}

📋 RECENT RECORDS (last 10):"""
        
        for trade in trades[-10:]:
            timestamp = trade.get('timestamp', 'N/A')[:19].replace('T', ' ')
            result += f"\n   {trade.get('action', 'N/A')} #{trade.get('id', 'N/A')} | {trade.get('strategy', 'N/A')} {trade.get('side', 'N/A')} | {timestamp}"
        
        if open_positions:
            result += f"\n\n🔄 DETECTED OPEN POSITIONS:"
            for trade_id, entry in open_positions.items():
                result += f"\n   #{trade_id} | {entry.get('strategy', 'N/A')} {entry.get('side', 'N/A')} | {entry.get('timestamp', 'N/A')[:19].replace('T', ' ')}"
        else:
            result += f"\n\n❌ No unmatched ENTRY records found"
        
        return result
    except Exception as e:
        return f"❌ Debug error: {e}\nProject root: {PROJECT_ROOT}\nWorking directory: {os.getcwd()}"

@mcp.tool()
def get_all_trades() -> str:
    """Get all complete trades"""
    trades = _get_complete_trades()
    if not trades:
        return "📊 No complete trades found in log"
    
    result = f"📊 ALL COMPLETE TRADES ({len(trades)} total)\n{'='*60}\n"
    return result + "\n\n".join(_format_trade_detail(trade) for trade in trades) + "\n"

@mcp.tool()
def get_recent_trades(hours: int = 24) -> str:
    """Get recent trades within specified hours"""
    trades = _filter_trades(_get_complete_trades(), hours_back=hours)
    
    if not trades:
        return f"📊 No trades found in last {hours} hours"
    
    stats = _calculate_stats(trades)
    result = f"""📊 RECENT TRADES - Last {hours}h ({len(trades)} trades)
{'='*50}
💰 Summary: ${stats['total_gross_pnl']:.2f} gross - ${stats['total_fees']:.2f} fees = ${stats['total_net_pnl']:.2f} net

"""
    return result + "\n".join(_format_trade_summary(trade) for trade in trades[-10:])

@mcp.tool()
def get_trade_statistics() -> str:
    """Get comprehensive trading statistics"""
    trades = _get_complete_trades()
    if not trades:
        return "📊 No complete trades for analysis"
    
    stats = _calculate_stats(trades)
    exit_reasons = {}
    for trade in trades:
        reason = trade.get('exit_trigger', 'unknown')
        exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
    
    win_loss_ratio = abs(stats['avg_win']/stats['avg_loss']) if stats['avg_loss'] != 0 else float('inf')
    
    result = f"""📊 TRADING PERFORMANCE STATISTICS
{'='*60}

💰 PROFIT & LOSS (Fee-Aware Analysis)
   Total Trades: {stats['total_trades']}
   Gross PnL: ${stats['total_gross_pnl']:.2f}
   Total Fees: ${stats['total_fees']:.2f} (0.11% per trade)
   Net PnL: ${stats['total_net_pnl']:.2f}
   
🎯 WIN/LOSS ANALYSIS
   Winning Trades: {stats['winners']} ({stats['win_rate']:.1f}%)
   Losing Trades: {stats['losers']} ({100-stats['win_rate']:.1f}%)
   Average Win: ${stats['avg_win']:.2f}
   Average Loss: ${stats['avg_loss']:.2f}
   Win/Loss Ratio: {win_loss_ratio:.2f}x

⚙️  STRATEGY BREAKDOWN
   Range Strategy: {stats['range_trades']} trades
   Trend Strategy: {stats['trend_trades']} trades
   
⏱️  TIMING ANALYSIS
   Average Hold Time: {stats['avg_hold_time']:.1f} seconds ({stats['avg_hold_time']/60:.1f} minutes)
   
🚪 EXIT REASONS"""
    
    for reason, count in sorted(exit_reasons.items(), key=lambda x: x[1], reverse=True):
        result += f"\n   {reason.replace('_', ' ').title()}: {count} ({count/stats['total_trades']*100:.1f}%)"
    
    return result

@mcp.tool()
def get_strategy_comparison() -> str:
    """Compare RANGE vs TREND strategies"""
    trades = _get_complete_trades()
    if not trades:
        return "📊 No trades for strategy comparison"
    
    range_trades = [t for t in trades if t['strategy'] == 'RANGE']
    trend_trades = [t for t in trades if t['strategy'] == 'TREND']
    
    def format_strategy_stats(strategy_trades, name):
        if not strategy_trades:
            return f"   {name} Strategy: No trades"
        stats = _calculate_stats(strategy_trades)
        return f"""   {name} Strategy:
     Trades: {stats['total_trades']} | Win Rate: {stats['win_rate']:.1f}%
     Net PnL: ${stats['total_net_pnl']:.2f} | Avg: ${stats['total_net_pnl']/stats['total_trades']:.2f}
     Avg Hold: {stats['avg_hold_time']:.1f}s ({stats['avg_hold_time']/60:.1f}min)"""
    
    result = f"""⚙️  STRATEGY PERFORMANCE COMPARISON
{'='*50}

{format_strategy_stats(range_trades, "RANGE")}

{format_strategy_stats(trend_trades, "TREND")}

📊 STRATEGY EFFICIENCY:"""
    
    if range_trades and trend_trades:
        range_per_trade = sum(t['estimated_net_profit'] for t in range_trades) / len(range_trades)
        trend_per_trade = sum(t['estimated_net_profit'] for t in trend_trades) / len(trend_trades)
        better_strategy = "RANGE" if range_per_trade > trend_per_trade else "TREND"
        result += f"\n   Best Performer: {better_strategy} strategy"
        result += f"\n   Range: ${range_per_trade:.2f} per trade"
        result += f"\n   Trend: ${trend_per_trade:.2f} per trade"
    
    return result

@mcp.tool()
def get_fee_analysis() -> str:
    """Analyze fee impact on trading performance"""
    trades = _get_complete_trades()
    if not trades:
        return "📊 No trades for fee analysis"
    
    stats = _calculate_stats(trades)
    fee_impact = (stats['total_fees'] / abs(stats['total_gross_pnl'])) * 100 if stats['total_gross_pnl'] != 0 else 0
    profitable_before_fees = len([t for t in trades if t['bybit_unrealized_pnl'] > 0])
    trades_killed_by_fees = profitable_before_fees - stats['winners']
    
    result = f"""💸 FEE IMPACT ANALYSIS (0.11% Bybit Model)
{'='*50}

📊 OVERALL FEE IMPACT:
   Total Fees Paid: ${stats['total_fees']:.2f}
   Fee Impact: {fee_impact:.1f}% of gross PnL
   
🎯 PROFITABILITY IMPACT:
   Profitable Before Fees: {profitable_before_fees}
   Profitable After Fees: {stats['winners']}
   Trades Killed by Fees: {trades_killed_by_fees}
   
💡 FEE EFFICIENCY:
   Average Fees per Trade: ${stats['total_fees']/stats['total_trades']:.2f}
   
📈 RECOMMENDATIONS:"""
    
    if trades_killed_by_fees > 0:
        result += f"\n   ⚠️  {trades_killed_by_fees} trades were profitable but became unprofitable due to fees"
        result += f"\n   💡 Consider higher profit targets or tighter stops"
    else:
        result += f"\n   ✅ Fee model is working well - no profitable trades killed by fees"
    
    worst_trades = sorted(trades, key=lambda t: t['break_even_fees'] - t['estimated_net_profit'], reverse=True)[:3]
    if worst_trades:
        result += f"\n\n🔴 WORST FEE IMPACT TRADES:"
        for i, trade in enumerate(worst_trades, 1):
            impact = trade['break_even_fees'] - trade['estimated_net_profit']
            result += f"\n   {i}. Trade #{trade['id']}: ${trade['bybit_unrealized_pnl']:.2f} gross → ${trade['estimated_net_profit']:.2f} net (${impact:.2f} fee impact)"
    
    return result

@mcp.tool()
def search_trades(strategy: Optional[str] = None, side: Optional[str] = None, min_profit: Optional[float] = None,
                 max_profit: Optional[float] = None, exit_trigger: Optional[str] = None, hours_back: Optional[int] = None) -> str:
    """Search and filter trades by criteria"""
    trades = _get_complete_trades()
    if not trades:
        return "📊 No trades to search"
    
    filtered_trades = _filter_trades(trades, strategy=strategy, side=side, min_profit=min_profit,
                                   max_profit=max_profit, exit_trigger=exit_trigger, hours_back=hours_back)
    
    if not filtered_trades:
        return "📊 No trades match the search criteria"
    
    # Build filter description
    filters = []
    if strategy: filters.append(f"strategy={strategy}")
    if side: filters.append(f"side={side}")
    if min_profit is not None: filters.append(f"profit>=${min_profit}")
    if max_profit is not None: filters.append(f"profit<=${max_profit}")
    if exit_trigger: filters.append(f"exit='{exit_trigger}'")
    if hours_back: filters.append(f"last {hours_back}h")
    
    filter_str = " | ".join(filters) if filters else "no filters"
    
    result = f"""🔍 TRADE SEARCH RESULTS ({filter_str})
Found {len(filtered_trades)} trades:
{'='*50}
"""
    
    result += "\n".join(_format_trade_summary(trade) for trade in filtered_trades[-10:])
    
    if len(filtered_trades) > 10:
        result += f"\n\n... and {len(filtered_trades) - 10} more trades"
    
    return result

@mcp.tool()
def get_live_positions() -> str:
    """Get open positions (ENTRY without EXIT)"""
    open_positions = _get_open_positions()
    
    if not open_positions:
        trades = _read_trades()
        entries = len([t for t in trades if t.get('action') == 'ENTRY'])
        exits = len([t for t in trades if t.get('action') == 'EXIT'])
        
        return f"""📊 No open positions found
        
🔍 DEBUG INFO:
   Total trades in log: {len(trades)}
   ENTRY records: {entries}
   EXIT records: {exits}
   
💡 If your bot shows a position but this shows none:
   1. Check if the bot has logged the ENTRY yet
   2. Verify log file path: {TRADES_LOG_PATH}
   3. Use debug_log_file() for detailed analysis"""
    
    result = f"🔄 OPEN POSITIONS ({len(open_positions)})\n{'='*40}\n"
    
    for trade_id, entry in open_positions.items():
        entry_time = entry['timestamp'][:19].replace('T', ' ')
        try:
            entry_dt = datetime.fromisoformat(entry['timestamp'].replace('Z', '+00:00'))
            age_minutes = (datetime.now().replace(tzinfo=entry_dt.tzinfo) - entry_dt).total_seconds() / 60
        except:
            age_minutes = 0
        
        result += f"""🔄 Position #{trade_id} | {entry['strategy']} | {entry['side']}
   📅 Opened: {entry_time} ({age_minutes:.1f}min ago)
   💰 Entry: ${entry['price']:.2f} | Size: ${entry.get('position_size_usdt', 0):.2f}
   📊 Market: {entry.get('market_condition', 'N/A')} | Confidence: {entry.get('confidence', 0):.1f}%
   💸 Break-even: ${entry.get('break_even_fees', 0):.2f} fees to cover

"""
    
    return result

@mcp.tool()
def calculate_rsi(prices: List[float], period: int = 14) -> float:
    """Calculate RSI for given prices"""
    if len(prices) < period + 1:
        return 50.0
    
    gains, losses = [], []
    for i in range(1, len(prices)):
        change = prices[i] - prices[i-1]
        gains.append(max(change, 0))
        losses.append(max(-change, 0))
    
    if len(gains) < period:
        return 50.0
    
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    
    if avg_loss == 0:
        return 100.0
    
    return round(100 - (100 / (1 + avg_gain / avg_loss)), 2)

@mcp.tool()
def export_trades_summary() -> str:
    """Export comprehensive trading summary"""
    trades = _get_complete_trades()
    if not trades:
        return "📊 No trades to export"
    
    stats = _calculate_stats(trades)
    first_trade = min(trades, key=lambda t: t['entry_timestamp'])
    last_trade = max(trades, key=lambda t: t['exit_timestamp'])
    top_winners = sorted(trades, key=lambda t: t['estimated_net_profit'], reverse=True)[:5]
    top_losers = sorted(trades, key=lambda t: t['estimated_net_profit'])[:5]
    
    report = f"""📊 COMPREHENSIVE TRADING REPORT
{'='*60}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Period: {first_trade['entry_timestamp'][:10]} to {last_trade['exit_timestamp'][:10]}

💰 FINANCIAL SUMMARY
   Total Trades: {stats['total_trades']}
   Win Rate: {stats['win_rate']:.1f}%
   Net PnL: ${stats['total_net_pnl']:.2f}
   Total Fees: ${stats['total_fees']:.2f}
   Net Per Trade: ${stats['total_net_pnl']/stats['total_trades']:.2f}

🎯 TOP PERFORMING TRADES:"""
    
    for i, trade in enumerate(top_winners, 1):
        report += f"\n   {i}. Trade #{trade['id']}: ${trade['estimated_net_profit']:.2f} ({trade['strategy']} {trade['side']})"
    
    report += f"\n\n🔴 WORST PERFORMING TRADES:"
    
    for i, trade in enumerate(top_losers, 1):
        report += f"\n   {i}. Trade #{trade['id']}: ${trade['estimated_net_profit']:.2f} ({trade['strategy']} {trade['side']})"
    
    return report

if __name__ == "__main__":
    mcp.run()