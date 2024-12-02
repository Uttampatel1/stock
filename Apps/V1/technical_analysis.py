import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import datetime
import os

class AdvancedTechnicalAnalysis:
    def __init__(self, data: pd.DataFrame):
        """
        Initialize with OHLCV DataFrame
        
        Args:
            data (pd.DataFrame): DataFrame with columns ['Open', 'High', 'Low', 'Close', 'Volume']
        """
        self._validate_input_data(data)
        self.data = data.copy()
        
    def _validate_input_data(self, data: pd.DataFrame) -> None:
        """Validate input data has required columns"""
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

    def _calculate_sma(self, series: pd.Series, periods: int) -> pd.Series:
        """Calculate Simple Moving Average"""
        return pd.Series(
            series.rolling(window=periods, min_periods=1).mean(),
            index=series.index
        )

    def _calculate_ema(self, series: pd.Series, periods: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return pd.Series(
            series.ewm(span=periods, min_periods=1, adjust=False).mean(),
            index=series.index
        )

    def _calculate_macd(self, series: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD and Signal line"""
        exp1 = self._calculate_ema(series, 12)
        exp2 = self._calculate_ema(series, 26)
        macd = exp1 - exp2
        signal = self._calculate_ema(macd, 9)
        return macd, signal

    def _calculate_rsi(self, series: pd.Series, periods: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=periods, min_periods=1).mean()
        avg_loss = loss.rolling(window=periods, min_periods=1).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return pd.Series(rsi.fillna(50), index=series.index)

    def _calculate_bollinger_bands(self, series: pd.Series, periods: int = 20) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        middle_band = self._calculate_sma(series, periods)
        std = series.rolling(window=periods, min_periods=1).std()
        upper_band = middle_band + (std * 2)
        lower_band = middle_band - (std * 2)
        return upper_band, middle_band, lower_band

    def _calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                            k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_period, min_periods=1).min()
        highest_high = high.rolling(window=k_period, min_periods=1).max()
        
        denominator = (highest_high - lowest_low)
        k = 100 * ((close - lowest_low) / denominator.where(denominator != 0, 1))
        d = k.rolling(window=d_period, min_periods=1).mean()
        
        k = pd.Series(k.fillna(50), index=close.index)
        d = pd.Series(d.fillna(50), index=close.index)
        return k, d

    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, periods: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return pd.Series(tr.rolling(window=periods, min_periods=1).mean(), index=close.index)

    def _calculate_obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate On Balance Volume"""
        close_diff = close.diff()
        direction = np.where(close_diff > 0, 1, np.where(close_diff < 0, -1, 0))
        obv = pd.Series((volume * direction).cumsum(), index=close.index)
        return obv

    def _calculate_price_rate_of_change(self, series: pd.Series, periods: int = 14) -> pd.Series:
        """Calculate Price Rate of Change"""
        return pd.Series(series.pct_change(periods=periods).fillna(0) * 100, index=series.index)

    def calculate_all_indicators(self) -> pd.DataFrame:
        """Calculate all technical indicators"""
        df = pd.DataFrame(index=self.data.index)
        
        # Copy original data
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = self.data[col]
        
        # Trend Indicators
        df['SMA_20'] = self._calculate_sma(df['Close'], 20)
        df['SMA_50'] = self._calculate_sma(df['Close'], 50)
        df['EMA_20'] = self._calculate_ema(df['Close'], 20)
        macd, signal = self._calculate_macd(df['Close'])
        df['MACD'] = macd
        df['MACD_Signal'] = signal
        
        # Momentum Indicators
        df['RSI'] = self._calculate_rsi(df['Close'])
        df['ROC'] = self._calculate_price_rate_of_change(df['Close'])
        k, d = self._calculate_stochastic(df['High'], df['Low'], df['Close'])
        df['STOCH_K'] = k
        df['STOCH_D'] = d
        
        # Volatility Indicators
        df['ATR'] = self._calculate_atr(df['High'], df['Low'], df['Close'])
        upper, middle, lower = self._calculate_bollinger_bands(df['Close'])
        df['BOLLINGER_UPPER'] = upper
        df['BOLLINGER_MIDDLE'] = middle
        df['BOLLINGER_LOWER'] = lower
        
        # Volume Indicators
        df['OBV'] = self._calculate_obv(df['Close'], df['Volume'])
        
        # Additional Features
        df['HL_PCT'] = ((df['High'] - df['Low']) / df['Close']).fillna(0) * 100
        df['PCT_CHANGE'] = df['Close'].pct_change().fillna(0) * 100
        df['VOLUME_PCT_CHANGE'] = df['Volume'].pct_change().fillna(0) * 100
        
        # Price Momentum Features
        for period in [5, 10, 20]:
            df[f'RETURN_{period}D'] = df['Close'].pct_change(periods=period).fillna(0) * 100
            df[f'VOLUME_MOMENTUM_{period}D'] = df['Volume'].pct_change(periods=period).fillna(0) * 100
        
        # Moving Average Crossovers (ensuring alignment)
        sma_20 = pd.Series(df['SMA_20'], index=df.index)
        sma_50 = pd.Series(df['SMA_50'], index=df.index)
        close = pd.Series(df['Close'], index=df.index)
        
        df['SMA_20_50_CROSS'] = (sma_20 > sma_50).astype(int)
        df['PRICE_SMA20_CROSS'] = (close > sma_20).astype(int)
        
        # Remove any remaining NaN values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        return df