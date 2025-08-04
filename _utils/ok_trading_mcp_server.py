#!/usr/bin/env python3
"""
Simple Trading MCP Server using FastMCP
"""

from fastmcp import FastMCP
import sys
import logging

# Set up logging to stderr
logging.basicConfig(level=logging.INFO, stream=sys.stderr)

# Create MCP server
mcp = FastMCP("trading-bot-data")

@mcp.tool()
def test_connection(message: str = "Hello from MCP!") -> str:
    """Test if MCP connection is working"""
    return f"✅ MCP Connection Working!\nMessage: {message}"

@mcp.tool()
def get_python_info() -> str:
    """Get Python environment information"""
    import platform
    import os
    
    info = {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "working_directory": os.getcwd()
    }
    
    return f"🐍 Python Environment:\n" + "\n".join([f"{k}: {v}" for k, v in info.items()])

@mcp.tool()
def calculate_rsi(prices: list[float], period: int = 14) -> float:
    """Calculate RSI (Relative Strength Index) for given prices"""
    if len(prices) < period + 1:
        return 0.0
    
    gains = []
    losses = []
    
    for i in range(1, len(prices)):
        change = prices[i] - prices[i-1]
        if change > 0:
            gains.append(change)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(change))
    
    if len(gains) < period:
        return 0.0
    
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    
    if avg_loss == 0:
        return 100.0
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return round(rsi, 2)

if __name__ == "__main__":
    mcp.run()