import os
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

class TelegramNotifier:
    """
    Streamlined Telegram Notification System
    
    Handles:
    - Trade entry/exit notifications
    - Bot status updates
    - Error handling and rate limiting
    """
    
    def __init__(self):
        # Configuration
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.symbol = os.getenv('TRADING_SYMBOL', 'ETHUSDT')
        
        # Check if notifications are enabled
        self.enabled = bool(self.bot_token and self.chat_id)
        
        # Request settings
        self.timeout = 10
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
    
    async def send_message(self, message: str) -> bool:
        """
        Send message to Telegram chat
        
        Args:
            message: Message text to send
            
        Returns:
            bool: True if sent successfully, False otherwise
        """
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
            
        except requests.exceptions.RequestException:
            return False
        except Exception:
            return False
    
    async def send_trade_entry(self, signal_data, price, quantity, strategy_info):
        """
        Send trade entry notification
        
        Args:
            signal_data: Signal dictionary with trade details
            price: Entry price
            quantity: Position quantity
            strategy_info: Strategy information
        """
        # Determine trade direction
        action = signal_data.get('action', 'UNKNOWN')
        emoji = "🟢 LONG" if action == 'BUY' else "🔴 SHORT"
        
        # Format message
        message = self._format_entry_message(
            emoji, signal_data, price, quantity
        )
        
        await self.send_message(message)
    
    def _format_entry_message(self, emoji, signal_data, price, quantity):
        """Format trade entry message"""
        return f"""
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
    
    async def send_trade_exit(self, exit_data, price, pnl, duration, strategy_info):
        """
        Send trade exit notification
        
        Args:
            exit_data: Exit details dictionary
            price: Exit price
            pnl: Profit/Loss amount
            duration: Trade duration in seconds
            strategy_info: Strategy information
        """
        # Determine result
        emoji = "🟢 WIN" if pnl >= 0 else "🔴 LOSS"
        
        # Format message
        message = self._format_exit_message(
            emoji, exit_data, price, pnl, duration
        )
        
        await self.send_message(message)
    
    def _format_exit_message(self, emoji, exit_data, price, pnl, duration):
        """Format trade exit message"""
        # Format PnL
        pnl_text = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
        
        # Format duration
        duration_text = self._format_duration(duration)
        
        # Format trigger reason
        trigger = exit_data.get('trigger', 'manual')
        trigger_text = trigger.replace('_', ' ').title()
        
        return f"""
📤 <b>TRADE EXIT</b> {emoji}

<b>🔹 Symbol:</b> {self.symbol}
<b>💸 Exit Price:</b> ${price:.2f}
<b>📊 PnL:</b> {pnl_text}
<b>⏱ Duration:</b> {duration_text}
<b>🎯 Trigger:</b> {trigger_text}

🕒 <b>Time:</b> {datetime.now().strftime('%H:%M:%S')}
"""
    
    def _format_duration(self, duration_seconds):
        """Format duration in human-readable format"""
        if duration_seconds < 60:
            return f"{duration_seconds:.1f}s"
        elif duration_seconds < 3600:
            minutes = duration_seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = duration_seconds / 3600
            return f"{hours:.1f}h"
    
    async def send_bot_status(self, status: str, message_text: str = ""):
        """
        Send bot status notification
        
        Args:
            status: Status type ('started', 'stopped', 'error', 'warning')
            message_text: Additional message text
        """
        # Status headlines
        headlines = {
            'started': '🚀 BOT STARTED',
            'stopped': '🛑 BOT STOPPED',
            'error': '❌ ERROR',
            'warning': '⚠️ WARNING'
        }
        
        headline = headlines.get(status.lower(), '📊 BOT STATUS')
        
        # Format message
        message = self._format_status_message(headline, message_text)
        
        await self.send_message(message)
    
    def _format_status_message(self, headline, message_text):
        """Format bot status message"""
        # Additional info line
        info_line = f"\n📝 <b>Info:</b> {message_text}" if message_text else ""
        
        return f"""
<b>{headline}</b>

<b>🔹 Symbol:</b> {self.symbol}
<b>🧠 Strategy:</b> Three Essential Conditions Framework{info_line}

🕒 <b>Time:</b> {datetime.now().strftime('%H:%M:%S')}
"""
    
    async def send_error_notification(self, error_type: str, error_message: str):
        """
        Send error notification with details
        
        Args:
            error_type: Type of error
            error_message: Error description
        """
        message = f"""
❌ <b>BOT ERROR</b>

<b>🔹 Symbol:</b> {self.symbol}
<b>⚠️ Type:</b> {error_type}
<b>📝 Details:</b> {error_message}

🕒 <b>Time:</b> {datetime.now().strftime('%H:%M:%S')}
"""
        
        await self.send_message(message)
    
    async def send_performance_update(self, trades_count: int, success_rate: float, 
                                    total_pnl: float):
        """
        Send performance summary notification
        
        Args:
            trades_count: Number of trades executed
            success_rate: Win rate percentage
            total_pnl: Total profit/loss
        """
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
        """
        Get configuration status for debugging
        
        Returns:
            dict: Configuration status
        """
        return {
            'bot_token_configured': bool(self.bot_token),
            'chat_id_configured': bool(self.chat_id),
            'enabled': self.enabled,
            'symbol': self.symbol,
            'timeout': self.timeout
        }