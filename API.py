#!/usr/bin/env python3
"""
Trading Bot HTTP Server for Claude Custom Connector
Simple HTTP API to expose trading log data
"""

import json
import os
import sys
from datetime import datetime
from typing import List, Dict, Optional
from decimal import Decimal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TradingLogConnector:
    """Connector for trading log data"""
    
    def __init__(self, log_file: str = "logs/trades.log"):
        self.log_file = log_file
        self.fee_rate = Decimal('0.0011')
        
    def read_logs(self) -> List[Dict]:
        """Read and parse all log entries"""
        if not os.path.exists(self.log_file):
            return []
        
        logs = []
        try:
            with open(self.log_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            logs.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except Exception:
            return []
        
        return logs
    
    def get_complete_trades(self) -> List[Dict]:
        """Get completed trades"""
        logs = self.read_logs()
        trades = {}
        complete_trades = []
        
        for log in logs:
            trade_id = log.get('id')
            if not trade_id:
                continue
                
            if log.get('action') == 'ENTRY':
                trades[trade_id] = log
            elif log.get('action') == 'EXIT' and trade_id in trades:
                entry = trades[trade_id]
                exit_log = log
                
                entry_price = Decimal(str(entry.get('price', 0)))
                position_size_usdt = Decimal(str(entry.get('position_size_usdt', 0)))
                bybit_pnl = Decimal(str(exit_log.get('bybit_unrealized_pnl', 0)))
                estimated_fees = position_size_usdt * self.fee_rate
                
                complete_trade = {
                    'trade_id': trade_id,
                    'entry_time': entry.get('timestamp'),
                    'exit_time': exit_log.get('timestamp'),
                    'strategy': entry.get('strategy'),
                    'side': entry.get('side'),
                    'entry_price': float(entry_price),
                    'exit_price': float(exit_log.get('price', 0)),
                    'position_size_usdt': float(position_size_usdt),
                    'bybit_unrealized_pnl': float(bybit_pnl),
                    'estimated_fees': float(estimated_fees),
                    'estimated_net_pnl': float(bybit_pnl - estimated_fees),
                    'hold_seconds': exit_log.get('hold_seconds', 0),
                    'exit_trigger': exit_log.get('trigger'),
                    'market_condition': entry.get('market_condition'),
                    'confidence': entry.get('confidence'),
                    'rsi': entry.get('rsi', 0),
                    'adx': entry.get('adx', 0)
                }
                
                complete_trades.append(complete_trade)
                del trades[trade_id]
        
        return sorted(complete_trades, key=lambda x: x['entry_time'])
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary"""
        complete_trades = self.get_complete_trades()
        
        if not complete_trades:
            return {"message": "No complete trades found"}
        
        total_trades = len(complete_trades)
        winning_trades = [t for t in complete_trades if t['bybit_unrealized_pnl'] > 0]
        win_rate = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0
        
        total_gross_pnl = sum(t['bybit_unrealized_pnl'] for t in complete_trades)
        total_estimated_fees = sum(t['estimated_fees'] for t in complete_trades)
        estimated_net_pnl = total_gross_pnl - total_estimated_fees
        
        range_trades = [t for t in complete_trades if t['strategy'] == 'RANGE']
        trend_trades = [t for t in complete_trades if t['strategy'] == 'TREND']
        
        return {
            'total_trades': total_trades,
            'win_rate': round(win_rate, 1),
            'winning_trades': len(winning_trades),
            'losing_trades': len([t for t in complete_trades if t['bybit_unrealized_pnl'] <= 0]),
            'total_gross_pnl': round(total_gross_pnl, 2),
            'total_estimated_fees': round(total_estimated_fees, 2),
            'estimated_net_pnl': round(estimated_net_pnl, 2),
            'range_strategy_trades': len(range_trades),
            'trend_strategy_trades': len(trend_trades),
            'avg_hold_time_seconds': round(sum(t['hold_seconds'] for t in complete_trades) / total_trades, 1) if total_trades > 0 else 0,
            'best_trade_pnl': max(t['bybit_unrealized_pnl'] for t in complete_trades) if complete_trades else 0,
            'worst_trade_pnl': min(t['bybit_unrealized_pnl'] for t in complete_trades) if complete_trades else 0
        }

# Initialize
app = FastAPI(title="Trading Bot Data API", version="1.0.0")
connector = TradingLogConnector()

# Enable CORS for Claude
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def health():
    """Health check endpoint"""
    return {
        "status": "Trading Bot Data API",
        "version": "1.0.0",
        "endpoints": [
            "/summary",
            "/trades",
            "/logs",
            "/analyze"
        ]
    }

@app.get("/summary")
async def get_performance_summary():
    """Get trading performance summary"""
    try:
        return connector.get_performance_summary()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/trades")
async def get_trades(
    limit: Optional[int] = Query(20, description="Limit number of trades"),
    strategy: Optional[str] = Query(None, description="Filter by strategy (RANGE/TREND)")
):
    """Get complete trades with optional filters"""
    try:
        trades = connector.get_complete_trades()
        
        if strategy:
            trades = [t for t in trades if t['strategy'] == strategy.upper()]
        
        if limit:
            trades = trades[-limit:]
        
        return {
            "trades": trades,
            "count": len(trades),
            "filters": {"limit": limit, "strategy": strategy}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/logs")
async def get_logs(
    limit: Optional[int] = Query(50, description="Limit number of log entries"),
    hours: Optional[int] = Query(None, description="Filter by hours back")
):
    """Get raw log entries with optional filters"""
    try:
        logs = connector.read_logs()
        
        if hours:
            cutoff = datetime.now().timestamp() - (hours * 3600)
            filtered_logs = []
            for log in logs:
                try:
                    log_time = datetime.fromisoformat(log['timestamp'].replace('Z', '+00:00')).timestamp()
                    if log_time >= cutoff:
                        filtered_logs.append(log)
                except:
                    continue
            logs = filtered_logs
        
        if limit:
            logs = logs[-limit:]
        
        return {
            "logs": logs,
            "count": len(logs),
            "filters": {"limit": limit, "hours": hours}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analyze")
async def analyze_performance(
    focus: str = Query("performance", description="Analysis focus: performance, strategy, risk"),
    limit: Optional[int] = Query(20, description="Limit trades for analysis")
):
    """Get detailed performance analysis"""
    try:
        trades = connector.get_complete_trades()
        summary = connector.get_performance_summary()
        
        if limit:
            recent_trades = trades[-limit:]
        else:
            recent_trades = trades
        
        analysis = {
            "focus": focus,
            "summary": summary,
            "recent_trades": recent_trades,
            "insights": _generate_insights(trades, summary, focus)
        }
        
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def _generate_insights(trades: List[Dict], summary: Dict, focus: str) -> Dict:
    """Generate insights based on focus area"""
    insights = {"focus": focus, "recommendations": []}
    
    if focus == "performance":
        win_rate = summary.get('win_rate', 0)
        net_pnl = summary.get('estimated_net_pnl', 0)
        
        insights["key_metrics"] = {
            "win_rate": win_rate,
            "net_pnl": net_pnl,
            "total_trades": summary.get('total_trades', 0)
        }
        
        if win_rate < 60:
            insights["recommendations"].append("Win rate below 60% - consider tightening entry criteria")
        if net_pnl < 0:
            insights["recommendations"].append("Negative net PnL - review risk management")
            
    elif focus == "strategy":
        range_trades = summary.get('range_strategy_trades', 0)
        trend_trades = summary.get('trend_strategy_trades', 0)
        
        insights["strategy_distribution"] = {
            "range_trades": range_trades,
            "trend_trades": trend_trades,
            "range_percentage": round((range_trades / (range_trades + trend_trades)) * 100, 1) if (range_trades + trend_trades) > 0 else 0
        }
        
        if range_trades > trend_trades * 2:
            insights["recommendations"].append("Heavy range bias - verify trend detection")
            
    elif focus == "risk":
        avg_hold = summary.get('avg_hold_time_seconds', 0)
        worst_trade = summary.get('worst_trade_pnl', 0)
        
        insights["risk_metrics"] = {
            "avg_hold_time": avg_hold,
            "worst_trade_pnl": worst_trade
        }
        
        if avg_hold > 1800:
            insights["recommendations"].append("High average hold time - consider tighter stops")
    
    return insights

if __name__ == "__main__":
    import uvicorn
    import ssl
    import os
    
    # Check if SSL certificates exist
    cert_file = "cert.pem"
    key_file = "key.pem"
    
    if not os.path.exists(cert_file) or not os.path.exists(key_file):
        print("🔒 Generating self-signed SSL certificate...")
        os.system(f'openssl req -x509 -newkey rsa:4096 -keyout {key_file} -out {cert_file} -days 365 -nodes -subj "/CN=localhost"')
        print("✅ SSL certificate generated!")
    
    # Run with HTTPS
    uvicorn.run(
        app, 
        host="127.0.0.1", 
        port=8000,
        ssl_keyfile=key_file,
        ssl_certfile=cert_file
    )