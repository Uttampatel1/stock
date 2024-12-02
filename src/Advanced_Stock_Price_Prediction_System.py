import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.svm import SVC
from xgboost import XGBClassifier
from finta import TA
from datetime import datetime, timedelta
from scipy.signal import find_peaks
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import warnings
warnings.filterwarnings('ignore')


class EnhancedStockPredictor:
    def __init__(self, symbol, lookback_period=365, output_dir='stock_analysis'):
        self.symbol = symbol
        self.lookback_period = lookback_period
        self.data = None
        self.model = None
        self.scaler = StandardScaler()
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def calculate_technical_indicators(self, df):
        """Enhanced technical indicators calculation"""
        # Trend Indicators
        df['ema20'] = TA.EMA(df, 20)
        df['ema50'] = TA.EMA(df, 50)
        df['ema200'] = TA.EMA(df, 200)
        
        # RSI with multiple timeframes
        df['rsi'] = TA.RSI(df)
        df['rsi_short'] = TA.RSI(df, period=7)
        df['rsi_long'] = TA.RSI(df, period=21)
        
        # MACD
        macd = TA.MACD(df)
        df['macd'] = macd['MACD']
        df['macd_signal'] = macd['SIGNAL']
        df['macd_hist'] = macd['MACD'] - macd['SIGNAL']
        
        # Volume Indicators
        df['mfi'] = TA.MFI(df)
        df['vwap'] = TA.VWAP(df)
        
        # Bollinger Bands
        bb = TA.BBANDS(df)
        df['bb_upper'] = bb['BB_UPPER']
        df['bb_middle'] = bb['BB_MIDDLE']
        df['bb_lower'] = bb['BB_LOWER']
        
        # Advanced Indicators
        try:
            df['adx'] = TA.ADX(df)  # Average Directional Index
        except:
            df['adx'] = np.nan
            
        try:
            df['cci'] = TA.CCI(df)  # Commodity Channel Index
        except:
            df['cci'] = np.nan
            
        try:
            df['stoch_k'] = TA.STOCH(df)['STOCH_K']
            df['stoch_d'] = TA.STOCH(df)['STOCH_D']
        except:
            df['stoch_k'] = np.nan
            df['stoch_d'] = np.nan
        
        # Custom Momentum Indicators
        df['price_momentum'] = df['Close'].diff(periods=1)
        df['price_momentum_5'] = df['Close'].diff(periods=5)
        df['volume_momentum'] = df['Volume'].diff(periods=1)
        
        # Price Channels
        df['upper_channel'] = df['High'].rolling(window=20).max()
        df['lower_channel'] = df['Low'].rolling(window=20).min()
        df['channel_width'] = df['upper_channel'] - df['lower_channel']
        
        # Volatility Indicators
        df['price_volatility'] = df['Close'].rolling(window=20).std()
        df['volume_volatility'] = df['Volume'].rolling(window=20).std()
        
        # Custom Indicators
        df['price_to_vwap'] = df['Close'] / df['vwap']
        df['volume_price_trend'] = (df['Close'] - df['Close'].shift(1)) * df['Volume']
        
        # Add candlestick patterns
        self.add_candlestick_patterns(df)
        
        return df
    
    def add_candlestick_patterns(self, df):
        """Add custom candlestick pattern detection"""
        # Calculate candlestick properties
        df['body'] = df['Close'] - df['Open']
        df['body_pct'] = (df['Close'] - df['Open']) / df['Open'] * 100
        df['upper_shadow'] = df['High'] - df[['Open', 'Close']].max(axis=1)
        df['lower_shadow'] = df[['Open', 'Close']].min(axis=1) - df['Low']
        df['total_range'] = df['High'] - df['Low']
        
        # Detect doji patterns
        df['doji'] = ((abs(df['body']) <= 0.1 * df['total_range']).astype(int))
        
        # Detect engulfing patterns
        df['bullish_engulfing'] = (
            (df['body'].shift(1) < 0) & 
            (df['body'] > 0) & 
            (df['Close'] > df['Open'].shift(1)) & 
            (df['Open'] < df['Close'].shift(1))
        ).astype(int)
        
        # Detect trend days
        df['trend_day'] = (abs(df['body_pct']) > 1).astype(int)
        
        # Add support/resistance levels
        self.add_support_resistance(df)
        
        return df
        
    def add_support_resistance(self, df, window=20):
        """Add support and resistance levels using peak detection"""
        # Find peaks for resistance levels
        peaks, _ = find_peaks(df['High'].values, distance=window)
        df['resistance'] = np.nan
        df.iloc[peaks, df.columns.get_loc('resistance')] = df.iloc[peaks]['High']
        
        # Find troughs for support levels
        troughs, _ = find_peaks(-df['Low'].values, distance=window)
        df['support'] = np.nan
        df.iloc[troughs, df.columns.get_loc('support')] = df.iloc[troughs]['Low']
        
        # Fill forward support/resistance levels
        df['resistance'] = df['resistance'].fillna(method='ffill')
        df['support'] = df['support'].fillna(method='ffill')
        
        # Calculate distance to support/resistance
        df['dist_to_resistance'] = (df['resistance'] - df['Close']) / df['Close'] * 100
        df['dist_to_support'] = (df['Close'] - df['support']) / df['Close'] * 100
        
        return df

# class EnhancedStockPredictor:
#     def __init__(self, symbol, lookback_period=365, output_dir='stock_analysis'):
#         self.symbol = symbol
#         self.lookback_period = lookback_period
#         self.data = None
#         self.model = None
#         self.scaler = StandardScaler()
#         self.output_dir = output_dir
#         os.makedirs(output_dir, exist_ok=True)

    def fetch_data(self):
        """Fetch historical data and calculate technical indicators"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.lookback_period)
            
            # Fetch data using yfinance
            stock = yf.Ticker(self.symbol)
            df = stock.history(start=start_date, end=end_date)
            
            if df.empty:
                raise ValueError(f"No data found for symbol {self.symbol}")
            
            print(f"Downloaded {len(df)} rows of data for {self.symbol}")
            
            # Calculate all technical indicators
            df = self.calculate_technical_indicators(df)
            
            # Create target variables with multiple horizons
            for horizon in [1, 3, 5, 10]:
                df[f'Target_{horizon}d'] = (
                    df['Close'].shift(-horizon) > df['Close']
                ).astype(int)
                
                # Add return magnitude for regression
                df[f'Return_{horizon}d'] = df['Close'].shift(-horizon) / df['Close'] - 1
            
            # Remove NaN values
            df = df.dropna()
            
            if len(df) < 100:  # Minimum required data points
                raise ValueError(f"Insufficient data points ({len(df)}) after processing")
            
            print(f"Final dataset size: {len(df)} rows")
            self.data = df
            return df
            
        except Exception as e:
            print(f"Error fetching data: {str(e)}")
            raise

    def calculate_technical_indicators(self, df):
        """Calculate technical indicators with error handling"""
        try:
            # Base calculations
            df['ema20'] = df['Close'].ewm(span=20, adjust=False).mean()
            df['ema50'] = df['Close'].ewm(span=50, adjust=False).mean()
            df['ema200'] = df['Close'].ewm(span=200, adjust=False).mean()
            
            # RSI calculation
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = df['Close'].ewm(span=12, adjust=False).mean()
            exp2 = df['Close'].ewm(span=26, adjust=False).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
            
            # Bollinger Bands
            df['bb_middle'] = df['Close'].rolling(window=20).mean()
            df['bb_upper'] = df['bb_middle'] + 2 * df['Close'].rolling(window=20).std()
            df['bb_lower'] = df['bb_middle'] - 2 * df['Close'].rolling(window=20).std()
            
            # Volume indicators
            df['volume_ma'] = df['Volume'].rolling(window=20).mean()
            df['volume_std'] = df['Volume'].rolling(window=20).std()
            
            # Price momentum
            for period in [1, 5, 10, 20]:
                df[f'momentum_{period}'] = df['Close'].pct_change(periods=period)
            
            # Volatility
            df['volatility'] = df['Close'].rolling(window=20).std()
            
            # Price channels
            df['upper_channel'] = df['High'].rolling(window=20).max()
            df['lower_channel'] = df['Low'].rolling(window=20).min()
            df['channel_width'] = df['upper_channel'] - df['lower_channel']
            
            # Custom indicators
            df['price_position'] = (df['Close'] - df['lower_channel']) / (df['upper_channel'] - df['lower_channel'])
            df['volume_price_ratio'] = df['Volume'] / df['Close']
            
            return df
            
        except Exception as e:
            print(f"Error calculating indicators: {str(e)}")
            raise

    def create_features(self):
        """Create feature matrix for machine learning with validation"""
        try:
            if self.data is None or self.data.empty:
                raise ValueError("No data available for feature creation")
            
            df = self.data.copy()
            
            base_features = [
                'ema20', 'ema50', 'ema200',
                'rsi', 'macd', 'macd_signal', 'macd_hist',
                'bb_upper', 'bb_middle', 'bb_lower',
                'volume_ma', 'volume_std',
                'momentum_1', 'momentum_5', 'momentum_10', 'momentum_20',
                'volatility', 'channel_width', 'price_position',
                'volume_price_ratio'
            ]
            
            # Verify all features exist
            available_features = [f for f in base_features if f in df.columns]
            
            if not available_features:
                raise ValueError("No valid features found in the dataset")
            
            print(f"Using {len(available_features)} features for prediction")
            
            X = df[available_features]
            y = df['Target_1d']
            
            # Verify data quality
            if X.isnull().any().any():
                raise ValueError("Features contain NULL values")
            
            return X, y
            
        except Exception as e:
            print(f"Error creating features: {str(e)}")
            raise

    def train_enhanced_model(self):
        """Train the model with proper validation"""
        try:
            X, y = self.create_features()
            
            if len(X) < 100:  # Minimum required samples
                raise ValueError(f"Insufficient samples ({len(X)}) for training")
            
            # Configure cross-validation
            n_splits = min(5, len(X) // 20)  # Ensure we have enough data for splits
            tscv = TimeSeriesSplit(n_splits=n_splits)
            
            # Create base models
            rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
            gb = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
            
            # Simple ensemble instead of voting classifier
            models = {'rf': rf, 'gb': gb}
            
            results = self._evaluate_model(models, X, y, tscv)
            
            # Train final model on all data
            self.model = rf  # Use RF as final model
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled, y)
            
            return results
            
        except Exception as e:
            print(f"Error training model: {str(e)}")
            raise

    def _evaluate_model(self, models, X, y, tscv):
        """Evaluate models with proper error handling"""
        scores = {name: {'precision': [], 'recall': [], 'f1': []} for name in models.keys()}
        
        try:
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Scale features
                X_train_scaled = self.scaler.fit_transform(X_train)
                X_val_scaled = self.scaler.transform(X_val)
                
                # Evaluate each model
                for name, model in models.items():
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_val_scaled)
                    
                    scores[name]['precision'].append(precision_score(y_val, y_pred))
                    scores[name]['recall'].append(recall_score(y_val, y_pred))
                    scores[name]['f1'].append(f1_score(y_val, y_pred))
            
            return scores
            
        except Exception as e:
            print(f"Error in model evaluation: {str(e)}")
            raise
        
    def predict_tomorrow(self):
        """Enhanced prediction with detailed analysis"""
        if self.model is None:
            self.fetch_data()
            results = self.train_enhanced_model()
        
        # Get latest data point
        latest_data = self.data.iloc[-1:]
        X_pred = self.create_features()[0].iloc[-1:] 
        
        # Scale features
        X_pred_scaled = self.scaler.transform(X_pred)
        
        # Make prediction
        prediction = self.model.predict(X_pred_scaled)[0]
        probabilities = self.model.predict_proba(X_pred_scaled)[0]
        
        # Get feature importances
        feature_imp = pd.DataFrame({
            'feature': X_pred.columns,
            'importance': self.model.estimators_[0].feature_importances_
        }).sort_values('importance', ascending=False)
        
        return {
            'prediction': 'Higher' if prediction == 1 else 'Lower',
            'confidence': float(max(probabilities)),
            'probability_up': float(probabilities[1]),
            'probability_down': float(probabilities[0]),
            'current_price': float(latest_data['Close'].values[0]),
            'date': latest_data.index[0].strftime('%Y-%m-%d'),
            'top_signals': feature_imp.head(5).to_dict('records'),
            'technical_signals': {
                'trend': 'Bullish' if latest_data['ema20'].values[0] > latest_data['ema50'].values[0] else 'Bearish',
                'rsi': float(latest_data['rsi'].values[0]),
                'macd': float(latest_data['macd'].values[0]),
                'support_level': float(latest_data['support'].values[0]),
                'resistance_level': float(latest_data['resistance'].values[0]),
                'volatility': float(latest_data['price_volatility'].values[0])
            }
        }


def analyze_stock(symbol):
    """Analyze stock with proper error handling"""
    try:
        predictor = EnhancedStockPredictor(symbol)
        print(f"Analyzing {symbol}...")
        
        # Fetch and process data
        predictor.fetch_data()
        
        # Train model
        results = predictor.train_enhanced_model()
        
        # Generate prediction
        prediction = predictor.predict_tomorrow()
        
        return {
            'symbol': symbol,
            'prediction': prediction,
            'model_performance': results
        }
        
    except Exception as e:
        print(f"Error analyzing {symbol}: {str(e)}")
        raise

    # def fetch_data(self):
    #     """Fetch historical data and calculate technical indicators"""
    #     end_date = datetime.now()
    #     start_date = end_date - timedelta(days=self.lookback_period)
        
    #     # Fetch data using yfinance
    #     stock = yf.Ticker(self.symbol)
    #     df = stock.history(start=start_date, end=end_date)
        
    #     # Calculate all technical indicators
    #     df = self.calculate_technical_indicators(df)
        
    #     # Create target variables with multiple horizons
    #     for horizon in [1, 3, 5, 10]:
    #         df[f'Target_{horizon}d'] = (
    #             df['Close'].shift(-horizon) > df['Close']
    #         ).astype(int)
            
    #         # Add return magnitude for regression
    #         df[f'Return_{horizon}d'] = df['Close'].shift(-horizon) / df['Close'] - 1
        
    #     # Remove NaN values
    #     df = df.dropna()
    #     self.data = df
    #     return df

    # class EnhancedStockPredictor:
    # def __init__(self, symbol, lookback_period=365, output_dir='stock_analysis'):
    #     self.symbol = symbol
    #     self.lookback_period = lookback_period
    #     self.data = None
    #     self.model = None
    #     self.scaler = StandardScaler()
    #     self.output_dir = output_dir
    #     os.makedirs(output_dir, exist_ok=True)

#     def create_features(self):
#         """Enhanced feature creation with feature interactions"""
#         df = self.data.copy()
        
#         # Basic features
#         features = [
#             'ema20', 'ema50', 'ema200', 
#             'rsi', 'rsi_short', 'rsi_long',
#             'macd', 'macd_signal', 'macd_hist',
#             'mfi', 'vwap',
#             'bb_upper', 'bb_middle', 'bb_lower',
#             'price_momentum', 'price_momentum_5',
#             'volume_momentum',
#             'upper_channel', 'lower_channel',
#             'price_volatility', 'volume_volatility',
#             'body', 'upper_shadow', 'lower_shadow',
#             'doji', 'bullish_engulfing',
#             'support', 'resistance',
#             'adx', 'cci', 'stoch_k', 'stoch_d',
#             'channel_width',
#             'dist_to_resistance', 'dist_to_support',
#             'body_pct', 'trend_day'
#         ]
        
#         # Create interaction features
#         df['rsi_macd_cross'] = (df['rsi'] - 50) * df['macd']
#         df['trend_strength'] = df['adx'] * (df['ema20'] - df['ema50'])
#         df['price_to_bb_width'] = (df['Close'] - df['bb_middle']) / (df['bb_upper'] - df['bb_lower'])
        
#         # Volume/Price interactions
#         df['volume_price_ratio'] = df['Volume'] / df['Close']
#         df['volume_trend'] = df['Volume'] * df['price_momentum']
        
#         # Add interaction features to the list
#         features.extend([
#             'rsi_macd_cross', 'trend_strength', 'price_to_bb_width',
#             'volume_price_ratio', 'volume_trend'
#         ])
        
#         # Select only features that exist in the dataframe
#         available_features = [f for f in features if f in df.columns]
        
#         X = df[available_features]
#         y = df['Target_1d']  # Using 1-day horizon as default
        
#         return X, y
    
#     def train_enhanced_model(self):
#         """Train an advanced ensemble model with multiple algorithms"""
#         X, y = self.create_features()
        
#         # Create base models
#         models = {
#             'rf': RandomForestClassifier(
#                 n_estimators=200, 
#                 max_depth=10, 
#                 min_samples_split=5,
#                 min_samples_leaf=2,
#                 random_state=42
#             ),
#             'gb': GradientBoostingClassifier(
#                 n_estimators=200, 
#                 max_depth=5,
#                 learning_rate=0.1,
#                 subsample=0.8,
#                 random_state=42
#             ),
#             'xgb': XGBClassifier(
#                 n_estimators=200,
#                 max_depth=5,
#                 learning_rate=0.1,
#                 subsample=0.8,
#                 random_state=42
#             ),
#             'svm': SVC(
#                 probability=True,
#                 kernel='rbf',
#                 C=1.0,
#                 random_state=42
#             )
#         }
        
#         # Create voting classifier
#         voting_clf = VotingClassifier(
#             estimators=[(name, model) for name, model in models.items()],
#             voting='soft'
#         )
        
#         # Time series cross-validation
#         tscv = TimeSeriesSplit(n_splits=5)
#         results = self._evaluate_model(voting_clf, X, y, tscv)
        
#         self.model = voting_clf
#         return results


#     def _evaluate_model(self, model, X, y, tscv):
#         """Enhanced model evaluation with detailed metrics"""
#         scores = {
#             'precision': [], 'recall': [], 'f1': [],
#             'confusion_matrices': [], 'predictions': []
#         }
        
#         for train_idx, val_idx in tscv.split(X):
#             X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
#             y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
#             # Scale features
#             X_train_scaled = self.scaler.fit_transform(X_train)
#             X_val_scaled = self.scaler.transform(X_val)
            
#             # Train and evaluate
#             model.fit(X_train_scaled, y_train)
#             y_pred = model.predict(X_val_scaled)
            
#             # Store metrics
#             scores['precision'].append(precision_score(y_val, y_pred))
#             scores['recall'].append(recall_score(y_val, y_pred))
#             scores['f1'].append(f1_score(y_val, y_pred))
#             scores['confusion_matrices'].append(confusion_matrix(y_val, y_pred))
#             scores['predictions'].extend(list(zip(y_val, y_pred)))
        
#         return scores

    
#     def generate_analysis_charts(self):
#         """Generate and save analysis charts"""
#         # Price and Indicators Chart
#         fig = make_subplots(rows=3, cols=1, shared_xaxis=True,
#                            subplot_titles=('Price and MA', 'RSI', 'MACD'))
        
#         # Add price and MA
#         fig.add_trace(go.Candlestick(
#             x=self.data.index,
#             open=self.data['Open'],
#             high=self.data['High'],
#             low=self.data['Low'],
#             close=self.data['Close'],
#             name='Price'
#         ), row=1, col=1)
        
#         # Add EMAs
#         for period in [20, 50, 200]:
#             fig.add_trace(go.Scatter(
#                 x=self.data.index,
#                 y=self.data[f'ema{period}'],
#                 name=f'EMA {period}',
#                 line=dict(width=1)
#             ), row=1, col=1)
        
#         # Add RSI
#         fig.add_trace(go.Scatter(
#             x=self.data.index,
#             y=self.data['rsi'],
#             name='RSI'
#         ), row=2, col=1)
        
#         # Add MACD
#         fig.add_trace(go.Scatter(
#             x=self.data.index,
#             y=self.data['macd'],
#             name='MACD'
#         ), row=3, col=1)
        
#         fig.update_layout(height=1000, title_text=f"{self.symbol} Technical Analysis")
#         fig.write_html(f"{self.output_dir}/{self.symbol}_technical_analysis.html")
        
#         # Generate feature importance plot
#         self._plot_feature_importance()
        
#         # Generate performance metrics plot
#         self._plot_performance_metrics()
    
#     def _plot_feature_importance(self):
#         """Plot and save feature importance"""
#         feature_imp = pd.DataFrame({
#             'feature': self.create_features()[0].columns,
#             'importance': self.model.estimators_[0].feature_importances_
#         }).sort_values('importance', ascending=False)
        
#         plt.figure(figsize=(12, 6))
#         sns.barplot(x='importance', y='feature', data=feature_imp.head(15))
#         plt.title(f'{self.symbol} - Top 15 Feature Importance')
#         plt.tight_layout()
#         plt.savefig(f"{self.output_dir}/{self.symbol}_feature_importance.png")
#         plt.close()
    
#     def _plot_performance_metrics(self):
#         """Plot and save model performance metrics"""
#         metrics = self.predict_tomorrow()['model_metrics']
        
#         plt.figure(figsize=(10, 6))
#         sns.boxplot(data=pd.DataFrame(metrics))
#         plt.title(f'{self.symbol} - Model Performance Metrics')
#         plt.tight_layout()
#         plt.savefig(f"{self.output_dir}/{self.symbol}_performance_metrics.png")
#         plt.close()


# def analyze_stock(symbol):
#     """Enhanced main analysis function"""
#     predictor = EnhancedStockPredictor(symbol)
#     predictor.fetch_data()
    
#     # Train enhanced model
#     training_results = predictor.train_enhanced_model()
    
#     # Generate analysis charts
#     predictor.generate_analysis_charts()
    
#     # Get prediction and analysis
#     prediction_result = predictor.predict_tomorrow()
    
#     # Return comprehensive results
#     return {
#         'symbol': symbol,
#         'prediction': prediction_result,
#         'training_results': training_results,
#         'charts_location': predictor.output_dir
#     }
