import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import talib
from datetime import datetime, timedelta

class StockPredictor:
    def __init__(self, symbol, lookback_period=365):
        """
        Initialize the stock predictor with a symbol and lookback period
        
        Args:
            symbol (str): Stock ticker symbol
            lookback_period (int): Number of days of historical data to use
        """
        self.symbol = symbol
        self.lookback_period = lookback_period
        self.data = None
        self.model = None
        self.scaler = StandardScaler()
        
    def fetch_data(self):
        """Fetch historical data and calculate technical indicators"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_period)
        
        # Fetch data using yfinance
        stock = yf.Ticker(self.symbol)
        df = stock.history(start=start_date, end=end_date)
        
        # Calculate technical indicators
        df['RSI'] = talib.RSI(df['Close'])
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        
        # MACD
        macd, signal, _ = talib.MACD(df['Close'])
        df['MACD'] = macd
        df['Signal'] = signal
        
        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(df['Close'])
        df['BB_upper'] = upper
        df['BB_middle'] = middle
        df['BB_lower'] = lower
        
        # Volume indicators
        df['OBV'] = talib.OBV(df['Close'], df['Volume'])
        df['ADL'] = talib.AD(df['High'], df['Low'], df['Close'], df['Volume'])
        
        # Momentum indicators
        df['MOM'] = talib.MOM(df['Close'])
        df['ROC'] = talib.ROC(df['Close'])
        
        # Create target variable (1 if tomorrow's price is higher, 0 if lower)
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        
        # Remove NaN values
        df = df.dropna()
        self.data = df
        return df
    
    def create_features(self):
        """Create feature matrix for machine learning"""
        df = self.data.copy()
        
        features = [
            'RSI', 'MA20', 'MA50', 'MACD', 'Signal',
            'BB_upper', 'BB_middle', 'BB_lower',
            'OBV', 'ADL', 'MOM', 'ROC',
            'Open', 'High', 'Low', 'Close', 'Volume'
        ]
        
        X = df[features]
        y = df['Target']
        
        return X, y
    
    def backtest_strategy(self, window_size=50):
        """
        Perform rolling window backtest of the prediction strategy
        
        Args:
            window_size (int): Size of rolling window for training
            
        Returns:
            float: Strategy performance metrics
        """
        df = self.data.copy()
        predictions = []
        actual_returns = []
        
        for i in range(window_size, len(df)):
            # Get training data
            train_data = df.iloc[i-window_size:i]
            
            # Prepare features
            X_train = train_data[self.features]
            y_train = train_data['Target']
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            
            # Train model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train_scaled, y_train)
            
            # Make prediction for next day
            X_pred = df.iloc[[i]][self.features]
            X_pred_scaled = self.scaler.transform(X_pred)
            pred = model.predict(X_pred_scaled)[0]
            
            predictions.append(pred)
            actual_returns.append(df.iloc[i]['Target'])
        
        # Calculate performance metrics
        accuracy = np.mean(np.array(predictions) == np.array(actual_returns))
        return accuracy
    
    def train_model(self):
        """Train the prediction model on full dataset"""
        X, y = self.create_features()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train_scaled, y_train)
        
        # Calculate accuracy
        train_accuracy = self.model.score(X_train_scaled, y_train)
        test_accuracy = self.model.score(X_test_scaled, y_test)
        
        return train_accuracy, test_accuracy
    
    def predict_tomorrow(self):
        """Predict if tomorrow's price will be higher or lower"""
        if self.model is None:
            self.fetch_data()
            self.train_model()
        
        # Get latest data point
        latest_data = self.data.iloc[-1:]
        features = [
            'RSI', 'MA20', 'MA50', 'MACD', 'Signal',
            'BB_upper', 'BB_middle', 'BB_lower',
            'OBV', 'ADL', 'MOM', 'ROC',
            'Open', 'High', 'Low', 'Close', 'Volume'
        ]
        X_pred = latest_data[features]
        
        # Scale features
        X_pred_scaled = self.scaler.transform(X_pred)
        
        # Make prediction
        prediction = self.model.predict(X_pred_scaled)[0]
        probabilities = self.model.predict_proba(X_pred_scaled)[0]
        
        return {
            'prediction': 'Higher' if prediction == 1 else 'Lower',
            'confidence': float(max(probabilities)),
            'current_price': float(latest_data['Close'].values[0]),
            'date': latest_data.index[0].strftime('%Y-%m-%d')
        }

def analyze_stock(symbol):
    """
    Main function to analyze a stock and make prediction
    
    Args:
        symbol (str): Stock ticker symbol
        
    Returns:
        dict: Prediction results and analysis metrics
    """
    predictor = StockPredictor(symbol)
    predictor.fetch_data()
    
    # Train model and get accuracies
    train_accuracy, test_accuracy = predictor.train_model()
    
    # Perform backtest
    backtest_accuracy = predictor.backtest_strategy()
    
    # Make prediction for tomorrow
    prediction_result = predictor.predict_tomorrow()
    
    # Combine results
    analysis_results = {
        'symbol': symbol,
        'prediction': prediction_result,
        'model_performance': {
            'training_accuracy': train_accuracy,
            'testing_accuracy': test_accuracy,
            'backtest_accuracy': backtest_accuracy
        }
    }
    
    return analysis_results