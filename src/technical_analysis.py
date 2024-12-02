import pandas as pd
import numpy as np
from typing import Dict

class CustomTechnicalAnalysis:
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()

    def calculate_all_indicators(self) -> pd.DataFrame:
        df = self.data.copy()
        
        print("Calculating moving averages...")
        self._add_moving_averages(df)
        
        print("Calculating MACD...")
        self._add_macd(df)
        
        print("Calculating RSI...")
        self._add_rsi(df)
        
        print("Calculating Bollinger Bands...")
        self._add_bollinger_bands(df)
        
        print("Calculating volume indicators...")
        self._add_volume_indicators(df)
        
        print("Calculating momentum indicators...")
        self._add_momentum_indicators(df)
        
        df = df.ffill().bfill()
        self.data = df
        
        return df
    
    def _add_moving_averages(self, df: pd.DataFrame) -> None:
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
            df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
    
    def _add_macd(self, df: pd.DataFrame) -> None:
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        
        df['MACD_line'] = exp1 - exp2
        df['MACD_signal'] = df['MACD_line'].ewm(span=9, adjust=False).mean()
        df['MACD_hist'] = df['MACD_line'] - df['MACD_signal']
    
    def _add_rsi(self, df: pd.DataFrame, period: int = 14) -> None:
        delta = df['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
    
    def _add_bollinger_bands(self, df: pd.DataFrame, period: int = 20) -> None:
        middle_band = df['Close'].rolling(window=period).mean()
        std_dev = df['Close'].rolling(window=period).std()
        
        df['BB_middle'] = middle_band
        df['BB_upper'] = middle_band + (std_dev * 2)
        df['BB_lower'] = middle_band - (std_dev * 2)
    
    def _add_volume_indicators(self, df: pd.DataFrame) -> None:
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
    
    def _add_momentum_indicators(self, df: pd.DataFrame) -> None:
        df['ROC'] = df['Close'].pct_change(periods=10) * 100
        df['Momentum'] = df['Close'] - df['Close'].shift(10)
        
        low_min = df['Low'].rolling(window=14).min()
        high_max = df['High'].rolling(window=14).max()
        
        df['Stoch_K'] = ((df['Close'] - low_min) / (high_max - low_min)) * 100
        df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()

    def get_signals(self) -> Dict:
        latest_row = self.data.iloc[-1]
        prev_row = self.data.iloc[-2]

        signals = {
            'price': latest_row['Close'].iloc[0],
            'trend': self._calculate_trend_signal(latest_row),
            'momentum': self._calculate_momentum_signal(latest_row),
            'volume': self._calculate_volume_signal(latest_row, prev_row),
            'volatility': self._calculate_volatility_signal(latest_row)
        }

        signal_values = [v for k, v in signals.items() if k != 'price']
        signals['overall'] = sum(signal_values)

        return signals

    def _calculate_trend_signal(self, row) -> int:
        signal = 0
        
        if row['SMA_20'].iloc[0] > row['SMA_50'].iloc[0]:
            signal += 1
        if row['SMA_50'].iloc[0] > row['SMA_200'].iloc[0]:
            signal += 1
            
        if row['MACD_hist'].iloc[0] > 0:
            signal += 1
        if row['MACD_line'].iloc[0] > row['MACD_signal'].iloc[0]:
            signal += 1
            
        return signal

    def _calculate_momentum_signal(self, row) -> int:
        signal = 0
        
        rsi = row['RSI'].iloc[0]
        if rsi < 30:
            signal += 2
        elif rsi > 70:
            signal -= 2
        
        stoch_k = row['Stoch_K'].iloc[0]
        if stoch_k < 20:
            signal += 1
        elif stoch_k > 80:
            signal -= 1
            
        return signal

    def _calculate_volume_signal(self, current_row, prev_row) -> int:
        signal = 0
        
        if current_row['Volume'].iloc[0] > current_row['Volume_SMA'].iloc[0]:
            if current_row['Close'].iloc[0] > prev_row['Close'].iloc[0]:
                signal += 1
            else:
                signal -= 1
        
        if current_row['OBV'].iloc[0] > prev_row['OBV'].iloc[0]:
            signal += 1
            
        return signal

    def _calculate_volatility_signal(self, row) -> int:
        close = row['Close'].iloc[0]
        upper = row['BB_upper'].iloc[0]
        lower = row['BB_lower'].iloc[0]
        
        if close < lower:
            return 1
        elif close > upper:
            return -1
        return 0

if __name__ == "__main__":
    import yfinance as yf
    
    # Fetch data
    symbol = "AAPL"
    data = yf.download(symbol, start="2023-01-01")
    
    # Run analysis
    analyzer = CustomTechnicalAnalysis(data)
    data_with_indicators = analyzer.calculate_all_indicators()
    signals = analyzer.get_signals()
    
    # Print results
    print(f"\nTechnical Analysis Results for {symbol}")
    print(f"Current Price: ${signals['price']:.2f}")
    print("\nSignals:")
    for key, value in signals.items():
        if key == 'price':
            continue
        signal_type = "BULLISH" if value > 0 else "BEARISH" if value < 0 else "NEUTRAL"
        strength = "STRONG " if abs(value) >= 2 else ""
        print(f"{key.upper()}: {strength}{signal_type} ({value})")