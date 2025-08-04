import os
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

class TelegramNotifier:
    """Streamlined Telegram Notification System"""
    
    def __init__(self):
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.symbol = os.getenv('TRADING_SYMBOL', 'ETHUSDT')
        self.enabled = bool(self.bot_token and self.chat_id)
        self.timeout = 10
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
    
    async def send_message(self, message: str) -> bool:
        """Send message to Telegram chat"""
        if not self.enabled:
            return False
        
        try:
            response = requests.post(
                f"{self.base_url}/sendMessage",
                json={
                    'chat_id': self.chat_id,
                    'text': message,
                    'parse_mode': 'HTML',
                    'disable_web_page_preview': True
                },
                timeout=self.timeout
            )
            return response.status_code == 200
        except:
            return False
    
    async def send_trade_entry(self, signal_data, price, quantity, strategy_info):
        """Send trade entry notification"""
        action = signal_data.get('action', 'UNKNOWN')
        emoji = "🟢 LONG" if action == 'BUY' else "🔴 SHORT"
        
        message = f"""
📥 <b>TRADE ENTRY</b> {emoji}

<b>🔹 Symbol:</b> {self.symbol}
<b>💰 Entry Price:</b> ${price:.2f}
<b>📦 Quantity:</b> {quantity}
<b>🛑 Stop Loss:</b> ${signal_data.get('structure_stop', 0):.2f}

<b>📊 SIGNAL DETAILS</b>
<b>📈 RSI:</b> {signal_data.get('rsi', 0):.1f}
<b>🎯 Confidence:</b> {signal_data.get('confidence', 0):.0f}%
<b>🧠 Type:</b> {signal_data.get('signal_type', 'framework').title()}

🕒 <b>Time:</b> {datetime.now().strftime('%H:%M:%S')}
"""
        await self.send_message(message)
    
    async def send_trade_exit(self, exit_data, price, pnl, duration, strategy_info):
        """Send trade exit notification"""
        emoji = "🟢 WIN" if pnl >= 0 else "🔴 LOSS"
        pnl_text = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
        duration_text = self._format_duration(duration)
        trigger = exit_data.get('trigger', 'manual').replace('_', ' ').title()
        
        message = f"""
📤 <b>TRADE EXIT</b> {emoji}

<b>🔹 Symbol:</b> {self.symbol}
<b>💸 Exit Price:</b> ${price:.2f}
<b>📊 PnL:</b> {pnl_text}
<b>⏱ Duration:</b> {duration_text}
<b>🎯 Trigger:</b> {trigger}

🕒 <b>Time:</b> {datetime.now().strftime('%H:%M:%S')}
"""
        await self.send_message(message)
    
    def _format_duration(self, duration_seconds):
        """Format duration in human-readable format"""
        if duration_seconds < 60:
            return f"{duration_seconds:.1f}s"
        elif duration_seconds < 3600:
            return f"{duration_seconds / 60:.1f}m"
        else:
            return f"{duration_seconds / 3600:.1f}h"
    
    async def send_bot_status(self, status: str, message_text: str = ""):
        """Send bot status notification"""
        headlines = {
            'started': '🚀 BOT STARTED',
            'stopped': '🛑 BOT STOPPED',
            'error': '❌ ERROR',
            'warning': '⚠️ WARNING'
        }
        
        headline = headlines.get(status.lower(), '📊 BOT STATUS')
        info_line = f"\n📝 <b>Info:</b> {message_text}" if message_text else ""
        
        message = f"""
<b>{headline}</b>

<b>🔹 Symbol:</b> {self.symbol}
<b>🧠 Strategy:</b> Three Essential Conditions Framework{info_line}

🕒 <b>Time:</b> {datetime.now().strftime('%H:%M:%S')}
"""
        await self.send_message(message)
    
    async def send_error_notification(self, error_type: str, error_message: str):
        """Send error notification"""
        message = f"""
❌ <b>BOT ERROR</b>

<b>🔹 Symbol:</b> {self.symbol}
<b>⚠️ Type:</b> {error_type}
<b>📝 Details:</b> {error_message}

🕒 <b>Time:</b> {datetime.now().strftime('%H:%M:%S')}
"""
        await self.send_message(message)
    
    async def send_performance_update(self, trades_count: int, success_rate: float, total_pnl: float):
        """Send performance summary notification"""
        pnl_emoji = "📈" if total_pnl >= 0 else "📉"
        pnl_text = f"+${total_pnl:.2f}" if total_pnl >= 0 else f"-${abs(total_pnl):.2f}"
        
        message = f"""
📊 <b>PERFORMANCE UPDATE</b>

<b>🔹 Symbol:</b> {self.symbol}
<b>🔢 Trades:</b> {trades_count}
<b>🎯 Success Rate:</b> {success_rate:.1f}%
<b>{pnl_emoji} Total PnL:</b> {pnl_text}

🕒 <b>Time:</b> {datetime.now().strftime('%H:%M:%S')}
"""
        await self.send_message(message)
    
    def is_enabled(self):
        """Check if Telegram notifications are enabled"""
        return self.enabled
    
    def get_config_status(self):
        """Get configuration status for debugging"""
        return {
            'bot_token_configured': bool(self.bot_token),
            'chat_id_configured': bool(self.chat_id),
            'enabled': self.enabled,
            'symbol': self.symbol,
            'timeout': self.timeout
        }