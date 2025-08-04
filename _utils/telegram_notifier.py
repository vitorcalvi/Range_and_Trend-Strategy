import os
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

class TelegramNotifier:
    """Streamlined Telegram notifications"""
    
    def __init__(self):
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.symbol = os.getenv('TRADING_SYMBOL', 'ETHUSDT')
        self.enabled = bool(self.bot_token and self.chat_id)
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
    
    async def send_message(self, message: str) -> bool:
        """Send message to Telegram"""
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
                timeout=10
            )
            return response.status_code == 200
        except:
            return False
    
    async def send_trade_entry(self, signal_data, price, quantity, strategy_info):
        """Send trade entry notification"""
        action = signal_data.get('action', 'UNKNOWN')
        emoji = "🟢 LONG" if action == 'BUY' else "🔴 SHORT"
        
        message = f"""📥 <b>ENTRY</b> {emoji}

<b>Symbol:</b> {self.symbol}
<b>Price:</b> ${price:.2f}
<b>Size:</b> {quantity}
<b>Stop:</b> ${signal_data.get('structure_stop', 0):.2f}

<b>Signal:</b> RSI {signal_data.get('rsi', 0):.1f} | Conf: {signal_data.get('confidence', 0):.0f}%
<b>Strategy:</b> {signal_data.get('strategy', 'UNKNOWN')}

🕒 {datetime.now().strftime('%H:%M:%S')}"""
        
        await self.send_message(message)
    
    async def send_trade_exit(self, exit_data, price, pnl, duration, strategy_info):
        """Send trade exit notification"""
        emoji = "🟢 WIN" if pnl >= 0 else "🔴 LOSS"
        pnl_text = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
        duration_text = self._format_duration(duration)
        trigger = exit_data.get('trigger', 'manual').replace('_', ' ').title()
        
        message = f"""📤 <b>EXIT</b> {emoji}

<b>Symbol:</b> {self.symbol}
<b>Price:</b> ${price:.2f}
<b>PnL:</b> {pnl_text}
<b>Duration:</b> {duration_text}
<b>Trigger:</b> {trigger}

🕒 {datetime.now().strftime('%H:%M:%S')}"""
        
        await self.send_message(message)
    
    def _format_duration(self, duration_seconds):
        """Format duration"""
        if duration_seconds < 60:
            return f"{duration_seconds:.1f}s"
        elif duration_seconds < 3600:
            return f"{duration_seconds / 60:.1f}m"
        else:
            return f"{duration_seconds / 3600:.1f}h"
    
    async def send_bot_status(self, status: str, message_text: str = ""):
        """Send bot status"""
        headlines = {
            'started': '🚀 BOT STARTED',
            'stopped': '🛑 BOT STOPPED',
            'error': '❌ ERROR'
        }
        
        headline = headlines.get(status.lower(), '📊 STATUS')
        info_line = f"\n<b>Info:</b> {message_text}" if message_text else ""
        
        message = f"""<b>{headline}</b>

<b>Symbol:</b> {self.symbol}
<b>Strategy:</b> Dual Strategy System{info_line}

🕒 {datetime.now().strftime('%H:%M:%S')}"""
        
        await self.send_message(message)