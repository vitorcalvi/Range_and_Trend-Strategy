import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, Any

class MarketConditionDetector:
    """Detects market conditions using ADX and supporting indicators"""
    
    def __init__(self):
        self.adx_period = 14
        self.bb_period = 20
        self.ma_period = 15
        
    def calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series) -> float:
        """Calculate ADX - try simple method first, then detailed if needed"""
        # Try simple method first (more reliable)
        simple_result = self.calculate_adx_simple(high, low, close)
        if simple_result != 25.0:  # If we got a real calculation (not default)
            return simple_result
        
        # Fall back to detailed method
        return self.calculate_adx_detailed(high, low, close)
    
    def calculate_adx_simple(self, high: pd.Series, low: pd.Series, close: pd.Series) -> float:
        """Simple, robust ADX calculation"""
        try:
            if len(close) < 20:
                print(f"🔍 ADX: Insufficient data for simple calc ({len(close)} < 20)")
                return 25.0
            
            print("🔍 ADX: Using simple calculation method")
            
            # Use last 50 periods for calculation
            h = high.iloc[-50:] if len(high) >= 50 else high
            l = low.iloc[-50:] if len(low) >= 50 else low
            c = close.iloc[-50:] if len(close) >= 50 else close
            
            # Simple True Range and Directional Movement
            tr = []
            dm_plus = []
            dm_minus = []
            
            for i in range(1, len(h)):
                # True Range
                tr_val = max(
                    h.iloc[i] - l.iloc[i],
                    abs(h.iloc[i] - c.iloc[i-1]),
                    abs(l.iloc[i] - c.iloc[i-1])
                )
                tr.append(tr_val)
                
                # Directional Movement
                h_move = h.iloc[i] - h.iloc[i-1]
                l_move = l.iloc[i-1] - l.iloc[i]
                
                if h_move > l_move and h_move > 0:
                    dm_plus.append(h_move)
                else:
                    dm_plus.append(0)
                    
                if l_move > h_move and l_move > 0:
                    dm_minus.append(l_move)
                else:
                    dm_minus.append(0)
            
            if len(tr) < 14:
                print(f"🔍 ADX: Not enough TR data ({len(tr)} < 14)")
                return 25.0
            
            # Simple moving averages (more stable than EWM)
            tr_avg = sum(tr[-14:]) / 14
            dm_plus_avg = sum(dm_plus[-14:]) / 14
            dm_minus_avg = sum(dm_minus[-14:]) / 14
            
            if tr_avg == 0:
                print("🔍 ADX: Zero TR average")
                return 25.0
            
            # Calculate DI
            di_plus = 100 * dm_plus_avg / tr_avg
            di_minus = 100 * dm_minus_avg / tr_avg
            
            # Calculate DX (simplified ADX)
            di_diff = abs(di_plus - di_minus)
            di_sum = di_plus + di_minus
            
            if di_sum == 0:
                print("🔍 ADX: Zero DI sum")
                return 25.0
            
            dx = 100 * di_diff / di_sum
            
            # Return DX as ADX approximation
            adx_result = max(0, min(100, dx))
            print(f"🔍 ADX: ✅ Simple calculation = {adx_result:.2f} (DI+: {di_plus:.1f}, DI-: {di_minus:.1f})")
            return adx_result
            
        except Exception as e:
            print(f"🔍 ADX: Simple calculation failed: {str(e)}")
            return 25.0
    
    def calculate_adx_detailed(self, high: pd.Series, low: pd.Series, close: pd.Series) -> float:
        """Calculate ADX for trend strength detection with detailed debugging"""
        try:
            if len(close) < self.adx_period + 10:
                print(f"🔍 ADX: Insufficient data ({len(close)} < {self.adx_period + 10})")
                return 25.0  # Neutral default
            
            # Validate input data
            if high.isna().any() or low.isna().any() or close.isna().any():
                print("🔍 ADX: NaN values detected in input data")
                return 25.0
            
            print(f"🔍 ADX: Starting calculation with {len(close)} candles")
            
            # Calculate True Range (robust method)
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            
            print(f"🔍 ADX: TR components - tr1: {tr1.iloc[-5:].tolist()}")
            print(f"🔍 ADX: TR components - tr2: {tr2.iloc[-5:].tolist()}")
            print(f"🔍 ADX: TR components - tr3: {tr3.iloc[-5:].tolist()}")
            
            # Use pandas concat with proper axis
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Validate TR
            if tr.isna().any():
                print(f"🔍 ADX: NaN values in True Range - count: {tr.isna().sum()}")
                # Fill NaN values for TR (first value will be NaN due to shift)
                tr = tr.fillna(0)
            
            print(f"🔍 ADX: TR calculated - last 5 values: {tr.iloc[-5:].tolist()}")
            
            # Calculate Directional Movement with detailed debugging
            high_diff = high.diff()
            low_diff = -low.diff()
            
            print(f"🔍 ADX: High diff - last 5: {high_diff.iloc[-5:].tolist()}")
            print(f"🔍 ADX: Low diff - last 5: {low_diff.iloc[-5:].tolist()}")
            
            # Create DM series with proper indexing
            dm_plus_values = []
            dm_minus_values = []
            
            for i in range(len(high)):
                if pd.isna(high_diff.iloc[i]) or pd.isna(low_diff.iloc[i]):
                    dm_plus_values.append(0.0)
                    dm_minus_values.append(0.0)
                else:
                    h_diff = high_diff.iloc[i]
                    l_diff = low_diff.iloc[i]
                    
                    if h_diff > l_diff and h_diff > 0:
                        dm_plus_values.append(h_diff)
                    else:
                        dm_plus_values.append(0.0)
                        
                    if l_diff > h_diff and l_diff > 0:
                        dm_minus_values.append(l_diff)
                    else:
                        dm_minus_values.append(0.0)
            
            dm_plus = pd.Series(dm_plus_values, index=high.index)
            dm_minus = pd.Series(dm_minus_values, index=high.index)
            
            print(f"🔍 ADX: DM+ last 5: {dm_plus.iloc[-5:].tolist()}")
            print(f"🔍 ADX: DM- last 5: {dm_minus.iloc[-5:].tolist()}")
            
            # Validate DM series
            if dm_plus.isna().any() or dm_minus.isna().any():
                print(f"🔍 ADX: NaN in DM - DM+: {dm_plus.isna().sum()}, DM-: {dm_minus.isna().sum()}")
                return 25.0
            
            # Smooth with Wilder's EMA - use more conservative settings
            alpha = 1.0 / self.adx_period
            min_periods = max(1, self.adx_period // 2)  # More lenient min_periods
            
            print(f"🔍 ADX: Smoothing with alpha={alpha:.4f}, min_periods={min_periods}")
            
            try:
                tr_smooth = tr.ewm(alpha=alpha, min_periods=min_periods, adjust=False).mean()
                dm_plus_smooth = dm_plus.ewm(alpha=alpha, min_periods=min_periods, adjust=False).mean()
                dm_minus_smooth = dm_minus.ewm(alpha=alpha, min_periods=min_periods, adjust=False).mean()
                
                print(f"🔍 ADX: Smoothed TR last 5: {tr_smooth.iloc[-5:].tolist()}")
                print(f"🔍 ADX: Smoothed DM+ last 5: {dm_plus_smooth.iloc[-5:].tolist()}")
                print(f"🔍 ADX: Smoothed DM- last 5: {dm_minus_smooth.iloc[-5:].tolist()}")
                
            except Exception as smooth_error:
                print(f"🔍 ADX: Smoothing error: {smooth_error}")
                return 25.0
            
            # Validate smoothed values
            if tr_smooth.isna().any() or dm_plus_smooth.isna().any() or dm_minus_smooth.isna().any():
                nan_tr = tr_smooth.isna().sum()
                nan_plus = dm_plus_smooth.isna().sum()
                nan_minus = dm_minus_smooth.isna().sum()
                print(f"🔍 ADX: NaN after smoothing - TR: {nan_tr}, DM+: {nan_plus}, DM-: {nan_minus}")
                
                # Try to recover by using only the non-NaN portion
                valid_idx = ~(tr_smooth.isna() | dm_plus_smooth.isna() | dm_minus_smooth.isna())
                if valid_idx.sum() < 5:
                    print("🔍 ADX: Too few valid values after smoothing")
                    return 25.0
                
                tr_smooth = tr_smooth[valid_idx]
                dm_plus_smooth = dm_plus_smooth[valid_idx]
                dm_minus_smooth = dm_minus_smooth[valid_idx]
                
                print(f"🔍 ADX: Using {len(tr_smooth)} valid values for calculation")
            
            # Calculate DI with robust zero division protection
            tr_smooth_safe = tr_smooth.replace(0, 1e-8)
            di_plus = 100 * dm_plus_smooth / tr_smooth_safe
            di_minus = 100 * dm_minus_smooth / tr_smooth_safe
            
            print(f"🔍 ADX: DI+ last 3: {di_plus.iloc[-3:].tolist()}")
            print(f"🔍 ADX: DI- last 3: {di_minus.iloc[-3:].tolist()}")
            
            # Calculate DX
            di_sum = di_plus + di_minus
            di_sum_safe = di_sum.replace(0, 1e-8)
            dx = 100 * abs(di_plus - di_minus) / di_sum_safe
            
            print(f"🔍 ADX: DX last 3: {dx.iloc[-3:].tolist()}")
            
            # Calculate ADX with same settings
            try:
                adx = dx.ewm(alpha=alpha, min_periods=min_periods, adjust=False).mean()
                print(f"🔍 ADX: Final ADX last 3: {adx.iloc[-3:].tolist()}")
            except Exception as adx_error:
                print(f"🔍 ADX: Final ADX calculation error: {adx_error}")
                return 25.0
            
            # Final validation
            if adx.empty or adx.isna().all():
                print("🔍 ADX: Final ADX calculation resulted in empty/NaN series")
                return 25.0
            
            final_adx = float(adx.iloc[-1])
            if pd.isna(final_adx) or not np.isfinite(final_adx):
                print(f"🔍 ADX: Invalid final value: {final_adx}")
                return 25.0
            
            # Clamp to reasonable range
            final_adx = max(0, min(100, final_adx))
            print(f"🔍 ADX: ✅ Calculated successfully = {final_adx:.2f}")
            return final_adx
            
        except Exception as e:
            print(f"🔍 ADX: ❌ Calculation error: {str(e)}")
            import traceback
            print(f"🔍 ADX: Traceback: {traceback.format_exc()}")
            print("🔍 ADX: Trying simple fallback calculation...")
            return self.calculate_adx_simple(high, low, close)
    
    def calculate_adx_simple(self, high: pd.Series, low: pd.Series, close: pd.Series) -> float:
        """Simple, robust ADX calculation as fallback"""
        try:
            if len(close) < 20:
                return 25.0
            
            print("🔍 ADX: Using simple fallback calculation")
            
            # Use last 50 periods for calculation
            h = high.iloc[-50:]
            l = low.iloc[-50:]
            c = close.iloc[-50:]
            
            # Simple True Range
            tr = []
            dm_plus = []
            dm_minus = []
            
            for i in range(1, len(h)):
                # True Range
                tr_val = max(
                    h.iloc[i] - l.iloc[i],
                    abs(h.iloc[i] - c.iloc[i-1]),
                    abs(l.iloc[i] - c.iloc[i-1])
                )
                tr.append(tr_val)
                
                # Directional Movement
                h_move = h.iloc[i] - h.iloc[i-1]
                l_move = l.iloc[i-1] - l.iloc[i]
                
                if h_move > l_move and h_move > 0:
                    dm_plus.append(h_move)
                else:
                    dm_plus.append(0)
                    
                if l_move > h_move and l_move > 0:
                    dm_minus.append(l_move)
                else:
                    dm_minus.append(0)
            
            if len(tr) < 14:
                return 25.0
            
            # Simple moving averages instead of EWM
            tr_avg = sum(tr[-14:]) / 14
            dm_plus_avg = sum(dm_plus[-14:]) / 14
            dm_minus_avg = sum(dm_minus[-14:]) / 14
            
            if tr_avg == 0:
                return 25.0
            
            # Calculate DI
            di_plus = 100 * dm_plus_avg / tr_avg
            di_minus = 100 * dm_minus_avg / tr_avg
            
            # Calculate DX
            di_diff = abs(di_plus - di_minus)
            di_sum = di_plus + di_minus
            
            if di_sum == 0:
                return 25.0
            
            dx = 100 * di_diff / di_sum
            
            # Return DX as ADX approximation
            adx_result = max(0, min(100, dx))
            print(f"🔍 ADX: Simple calculation result = {adx_result:.2f}")
            return adx_result
            
        except Exception as e:
            print(f"🔍 ADX: Simple calculation also failed: {str(e)}")
            return 25.0
            
    def calculate_adx_simple(self, high: pd.Series, low: pd.Series, close: pd.Series) -> float:
        """Simple, robust ADX calculation as fallback"""
        try:
            if len(close) < 20:
                return 25.0
            
            # Use last 50 periods for calculation
            h = high.iloc[-50:]
            l = low.iloc[-50:]
            c = close.iloc[-50:]
            
            # Simple True Range
            tr = []
            dm_plus = []
            dm_minus = []
            
            for i in range(1, len(h)):
                # True Range
                tr_val = max(
                    h.iloc[i] - l.iloc[i],
                    abs(h.iloc[i] - c.iloc[i-1]),
                    abs(l.iloc[i] - c.iloc[i-1])
                )
                tr.append(tr_val)
                
                # Directional Movement
                h_move = h.iloc[i] - h.iloc[i-1]
                l_move = l.iloc[i-1] - l.iloc[i]
                
                if h_move > l_move and h_move > 0:
                    dm_plus.append(h_move)
                else:
                    dm_plus.append(0)
                    
                if l_move > h_move and l_move > 0:
                    dm_minus.append(l_move)
                else:
                    dm_minus.append(0)
            
            if len(tr) < 14:
                return 25.0
            
            # Simple moving averages instead of EWM
            tr_avg = sum(tr[-14:]) / 14
            dm_plus_avg = sum(dm_plus[-14:]) / 14
            dm_minus_avg = sum(dm_minus[-14:]) / 14
            
            if tr_avg == 0:
                return 25.0
            
            # Calculate DI
            di_plus = 100 * dm_plus_avg / tr_avg
            di_minus = 100 * dm_minus_avg / tr_avg
            
            # Calculate DX
            di_diff = abs(di_plus - di_minus)
            di_sum = di_plus + di_minus
            
            if di_sum == 0:
                return 25.0
            
            dx = 100 * di_diff / di_sum
            
            # Return DX as ADX approximation
            return max(0, min(100, dx))
            
        except:
            return 25.0
    
    def calculate_volatility_regime(self, close: pd.Series) -> str:
        """Calculate volatility regime using Bollinger Bands with validation"""
        try:
            if len(close) < self.bb_period:
                print(f"🔍 Volatility: Insufficient data ({len(close)} < {self.bb_period})")
                return "NORMAL"
            
            # Validate input data
            if close.isna().any():
                print("🔍 Volatility: NaN values in price data")
                close = close.fillna(close.iloc[-1])  # Forward fill with last valid price
            
            sma = close.rolling(self.bb_period, min_periods=self.bb_period).mean()
            std = close.rolling(self.bb_period, min_periods=self.bb_period).std()
            
            # Validate calculations
            if sma.isna().all() or std.isna().all():
                print("🔍 Volatility: SMA or STD calculation failed")
                return "NORMAL"
            
            # Calculate Bollinger Band width with safety checks
            current_sma = sma.iloc[-1]
            current_std = std.iloc[-1]
            
            if pd.isna(current_sma) or pd.isna(current_std) or current_sma == 0:
                print("🔍 Volatility: Invalid SMA or STD values")
                return "NORMAL"
            
            bb_width = (current_std * 2) / current_sma
            
            # Need at least 50 periods for average width calculation
            if len(std) >= 50:
                avg_width = (std * 2 / sma).rolling(50, min_periods=25).mean().iloc[-1]
                
                if pd.isna(avg_width) or avg_width == 0:
                    print("🔍 Volatility: Invalid average width")
                    return "NORMAL"
                
                width_ratio = bb_width / avg_width
                
                if width_ratio > 1.5:
                    return "HIGH_VOL"
                elif width_ratio < 0.5:
                    return "LOW_VOL"
                else:
                    return "NORMAL"
            else:
                print(f"🔍 Volatility: Not enough data for avg width ({len(std)} < 50)")
                return "NORMAL"
                
        except Exception as e:
            print(f"🔍 Volatility: Calculation error: {str(e)}")
            return "NORMAL"
    
    def detect_market_condition(self, data_1m: pd.DataFrame, data_15m: pd.DataFrame) -> Dict[str, Any]:
        """Detect current market condition with robust validation"""
        if len(data_1m) < 50 or len(data_15m) < 30:
            print(f"🔍 Market Detection: Insufficient data - 1m:{len(data_1m)}, 15m:{len(data_15m)}")
            return {"condition": "INSUFFICIENT_DATA", "adx": 25.0, "confidence": 0, "timestamp": datetime.now()}
        
        try:
            # Calculate ADX on 15-minute data for better trend detection
            print(f"🔍 Market Detection: Calculating ADX on 15m data ({len(data_15m)} candles)")
            adx_15m = self.calculate_adx(data_15m['high'], data_15m['low'], data_15m['close'])
            
            # Validate ADX result
            if pd.isna(adx_15m) or not np.isfinite(adx_15m):
                print(f"🔍 Market Detection: Invalid ADX result: {adx_15m}, using default")
                adx_15m = 25.0
            
            # Calculate volatility regime on 1-minute data
            vol_regime = self.calculate_volatility_regime(data_1m['close'])
            
            # Determine market condition with validated ADX
            if adx_15m < 20:
                condition = "STRONG_RANGE"
                confidence = 0.9
            elif adx_15m < 25:
                condition = "WEAK_RANGE" 
                confidence = 0.7
            elif adx_15m < 40:
                condition = "TRENDING"
                confidence = 0.8
            else:
                condition = "STRONG_TREND"
                confidence = 0.95
            
            result = {
                "condition": condition,
                "adx": adx_15m,
                "volatility": vol_regime,
                "confidence": confidence,
                "timestamp": datetime.now()
            }
            
            print(f"🔍 Market Detection: {condition} (ADX: {adx_15m:.2f}, Vol: {vol_regime}, Conf: {confidence*100:.0f}%)")
            return result
            
        except Exception as e:
            print(f"🔍 Market Detection: Error - {str(e)}")
            return {
                "condition": "WEAK_RANGE",  # Safe default
                "adx": 25.0,
                "volatility": "NORMAL",
                "confidence": 0.5,
                "timestamp": datetime.now()
            }

class StrategyManager:
    """Manages dual strategy system with automatic switching"""
    
    def __init__(self):
        self.detector = MarketConditionDetector()
        self.current_strategy = None
        self.last_switch_time = None
        self.switch_cooldown = 300  # 5 minutes cooldown between switches
        self.market_condition = {"condition": "UNKNOWN", "adx": 25.0}
        
    def should_switch_strategy(self, new_condition: str) -> bool:
        """Determine if strategy should be switched"""
        if self.current_strategy is None:
            return True
            
        # Cooldown check
        if (self.last_switch_time and 
            (datetime.now() - self.last_switch_time).total_seconds() < self.switch_cooldown):
            return False
            
        # Strategy mapping
        current_type = "RANGE" if self.current_strategy in ["STRONG_RANGE", "WEAK_RANGE"] else "TREND"
        new_type = "RANGE" if new_condition in ["STRONG_RANGE", "WEAK_RANGE"] else "TREND"
        
        return current_type != new_type
    
    def select_strategy(self, data_1m: pd.DataFrame, data_15m: pd.DataFrame) -> Tuple[str, Dict[str, Any]]:
        """Select appropriate strategy based on market conditions"""
        # Detect market condition
        market_info = self.detector.detect_market_condition(data_1m, data_15m)
        condition = market_info["condition"]
        
        if condition == "INSUFFICIENT_DATA":
            return "RANGE", market_info
            
        # Check if strategy switch is needed
        if self.should_switch_strategy(condition):
            self.current_strategy = condition
            self.last_switch_time = datetime.now()
            
        # Map condition to strategy
        if condition in ["STRONG_RANGE", "WEAK_RANGE"]:
            strategy_type = "RANGE"
        else:
            strategy_type = "TREND"
            
        self.market_condition = market_info
        return strategy_type, market_info
    
    def get_active_timeframe(self, strategy_type: str) -> str:
        """Get the appropriate timeframe for the strategy"""
        return "1m" if strategy_type == "RANGE" else "15m"
    
    def get_position_sizing_multiplier(self, strategy_type: str, market_info: Dict[str, Any]) -> float:
        """Get position sizing multiplier based on strategy and market conditions"""
        base_multiplier = 1.0
        
        if strategy_type == "TREND":
            # Larger positions for trending markets
            if market_info["condition"] == "STRONG_TREND":
                base_multiplier = 1.5
            else:
                base_multiplier = 1.2
        else:
            # Smaller positions for ranging markets
            if market_info["condition"] == "STRONG_RANGE":
                base_multiplier = 0.8
            else:
                base_multiplier = 1.0
                
        # Adjust for volatility
        if market_info.get("volatility") == "HIGH_VOL":
            base_multiplier *= 0.7
        elif market_info.get("volatility") == "LOW_VOL":
            base_multiplier *= 1.2
            
        return base_multiplier
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get current strategy information"""
        return {
            "current_strategy": self.current_strategy,
            "market_condition": self.market_condition,
            "last_switch": self.last_switch_time,
            "switch_cooldown_remaining": max(0, self.switch_cooldown - 
                (datetime.now() - self.last_switch_time).total_seconds()) 
                if self.last_switch_time else 0
        }