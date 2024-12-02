# technical_analysis.py
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import datetime
import os
import matplotlib.pyplot as plt
import yfinance as yf

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

    def _create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all features consistently"""
        df = pd.DataFrame(index=data.index)
        
        # Basic price and returns
        df['Returns'] = data['Close'].pct_change()
        returns = df['Returns']
        volume = data['Volume'].astype(float)  # Convert to float Series
        
        # Trend Indicators
        df['SMA_20'] = data['Close'].rolling(window=20, min_periods=1).mean()
        df['SMA_50'] = data['Close'].rolling(window=50, min_periods=1).mean()
        df['EMA_20'] = data['Close'].ewm(span=20, min_periods=1, adjust=False).mean()
        
        # MACD
        exp1 = data['Close'].ewm(span=12, min_periods=1, adjust=False).mean()
        exp2 = data['Close'].ewm(span=26, min_periods=1, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, min_periods=1, adjust=False).mean()
        
        # Trend Strength and Direction
        df['TREND_STRENGTH'] = abs(df['SMA_20'] - df['SMA_50']) / df['SMA_50']
        df['TREND_DIRECTION'] = (df['SMA_20'] > df['SMA_50']).astype(float)
        
        # Volatility
        df['VOLATILITY'] = returns.rolling(window=20, min_periods=1).std()
        df['VOLATILITY_MA'] = df['VOLATILITY'].rolling(window=10, min_periods=1).mean()
        
        # RSI
        delta = returns
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14, min_periods=1).mean()
        avg_loss = loss.rolling(window=14, min_periods=1).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        df['RSI_TREND'] = (df['RSI'] > df['RSI'].shift(1)).astype(float)
        
        # MACD Crossover
        df['MACD_CROSS'] = ((df['MACD'] > df['MACD_Signal']) & 
                            (df['MACD'].shift(1) <= df['MACD_Signal'].shift(1))).astype(float)
        
        # Volume Price Trend
        # Using vectorized operations instead of loop
        df['VOLUME_PRICE_TREND'] = (volume * returns).cumsum().rolling(window=20, min_periods=1).mean()
        
        # Bollinger Bands
        middle_band = df['SMA_20']
        std = data['Close'].rolling(window=20, min_periods=1).std()
        df['BOLLINGER_UPPER'] = middle_band + (std * 2)
        df['BOLLINGER_LOWER'] = middle_band - (std * 2)
        df['BB_WIDTH'] = (df['BOLLINGER_UPPER'] - df['BOLLINGER_LOWER']) / middle_band
        df['BB_POSITION'] = (data['Close'] - df['BOLLINGER_LOWER']) / (df['BOLLINGER_UPPER'] - df['BOLLINGER_LOWER'])
        
        # Price to MA Ratios
        df['PRICE_TO_SMA20'] = data['Close'] / df['SMA_20'] - 1
        df['PRICE_TO_SMA50'] = data['Close'] / df['SMA_50'] - 1
        
        # Volume Indicators
        volume_ma = volume.rolling(window=20, min_periods=1).mean()
        df['VOLUME_TO_MA'] = volume / volume_ma
        df['VOLUME_TREND'] = volume.pct_change().rolling(window=5, min_periods=1).mean()
        
        # Fill NaN values
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        return df
 
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

# advanced_ml_model.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import optuna
from typing import Dict, List, Tuple, Optional
import joblib
import datetime
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AdvancedStockPredictor:
    def __init__(self, feature_data: Optional[pd.DataFrame] = None):
        """
        Initialize with feature DataFrame
        
        Args:
            feature_data (Optional[pd.DataFrame]): DataFrame with technical indicators
        """
        self.data = feature_data.copy() if feature_data is not None else None
        self.models: Dict = {}
        self.feature_importance: Optional[pd.DataFrame] = None
        self.scaler = StandardScaler()
        
    def prepare_data(self, target_column: str, test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Prepare data for training with improved feature engineering
        
        Args:
            target_column (str): Column name for prediction target
            test_size (float): Proportion of data to use for testing
        """
        if self.data is None:
            raise ValueError("No data available. Initialize with feature_data or load a saved model.")
            
        # Create a copy of the data for feature engineering
        X = self.data.copy()
        
        # Calculate returns for target and features
        returns = X[target_column].pct_change()
        std_returns = returns.rolling(window=20).std()
        threshold = 0.5 * std_returns  # Dynamic threshold based on volatility
        y = (returns.shift(-1) > threshold).astype(int)

        # Print class distribution
        positive_ratio = y.mean()
        logger.info(f"Positive class ratio: {positive_ratio:.2%}")
        
        # Add trend strength indicators
        X['TREND_STRENGTH'] = abs(X['SMA_20'] - X['SMA_50']) / X['SMA_50']
        X['TREND_DIRECTION'] = (X['SMA_20'] > X['SMA_50']).astype(int)
        
        # Volatility indicators
        X['VOLATILITY'] = returns.rolling(window=20).std()
        X['VOLATILITY_MA'] = X['VOLATILITY'].rolling(window=10).mean()
        
        # Advanced momentum features
        X['RSI_TREND'] = (X['RSI'] > X['RSI'].shift(1)).astype(int)
        X['MACD_CROSS'] = ((X['MACD'] > X['MACD_Signal']) & 
                        (X['MACD'].shift(1) <= X['MACD_Signal'].shift(1))).astype(int)
        
        # Volume price trend
        X['VOLUME_PRICE_TREND'] = (X['Volume'] * returns).rolling(window=20).mean()
        
        # Bollinger Band indicators
        X['BB_WIDTH'] = (X['BOLLINGER_UPPER'] - X['BOLLINGER_LOWER']) / X['BOLLINGER_MIDDLE']
        X['BB_POSITION'] = (X[target_column] - X['BOLLINGER_LOWER']) / (X['BOLLINGER_UPPER'] - X['BOLLINGER_LOWER'])
        
        # Price relative to moving averages
        X['PRICE_TO_SMA20'] = X[target_column] / X['SMA_20'] - 1
        X['PRICE_TO_SMA50'] = X[target_column] / X['SMA_50'] - 1
        
        # Advanced volume features
        X['VOLUME_TO_MA'] = X['Volume'] / X['Volume'].rolling(window=20).mean()
        X['VOLUME_TREND'] = X['Volume'].pct_change().rolling(window=5).mean()
        
        # Remove price columns and other unnecessary features after calculations
        columns_to_drop = ['Open', 'High', 'Low', target_column]
        X = X.drop(columns=columns_to_drop)
        
        # Remove NaN values
        X = X.dropna()
        y = y[X.index]
        
        # Split data ensuring no lookahead bias
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:-1]
        y_train, y_test = y[:split_idx], y[split_idx:-1]
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, list(X.columns)

    def _create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all features consistently"""
        df = pd.DataFrame(index=data.index)
        
        # Basic price and returns
        df['Returns'] = data['Close'].pct_change()
        returns = df['Returns']
        
        # Trend Indicators
        df['SMA_20'] = data['Close'].rolling(window=20, min_periods=1).mean()
        df['SMA_50'] = data['Close'].rolling(window=50, min_periods=1).mean()
        df['EMA_20'] = data['Close'].ewm(span=20, min_periods=1, adjust=False).mean()
        
        # MACD
        exp1 = data['Close'].ewm(span=12, min_periods=1, adjust=False).mean()
        exp2 = data['Close'].ewm(span=26, min_periods=1, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, min_periods=1, adjust=False).mean()
        
        # Trend Strength and Direction
        df['TREND_STRENGTH'] = abs(df['SMA_20'] - df['SMA_50']) / df['SMA_50']
        df['TREND_DIRECTION'] = (df['SMA_20'] > df['SMA_50']).astype(float)
        
        # Volatility
        df['VOLATILITY'] = returns.rolling(window=20, min_periods=1).std()
        df['VOLATILITY_MA'] = df['VOLATILITY'].rolling(window=10, min_periods=1).mean()
        
        # RSI
        delta = returns
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14, min_periods=1).mean()
        avg_loss = loss.rolling(window=14, min_periods=1).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        df['RSI_TREND'] = (df['RSI'] > df['RSI'].shift(1)).astype(float)
        
        # MACD Crossover
        df['MACD_CROSS'] = ((df['MACD'] > df['MACD_Signal']) & 
                            (df['MACD'].shift(1) <= df['MACD_Signal'].shift(1))).astype(float)
        
        # Volume Price Trend
        df['VOLUME_PRICE_TREND'] = (data['Volume'] * returns).rolling(window=20, min_periods=1).mean()
        
        # Bollinger Bands
        middle_band = df['SMA_20']
        std = data['Close'].rolling(window=20, min_periods=1).std()
        df['BOLLINGER_UPPER'] = middle_band + (std * 2)
        df['BOLLINGER_LOWER'] = middle_band - (std * 2)
        df['BB_WIDTH'] = (df['BOLLINGER_UPPER'] - df['BOLLINGER_LOWER']) / middle_band
        df['BB_POSITION'] = (data['Close'] - df['BOLLINGER_LOWER']) / (df['BOLLINGER_UPPER'] - df['BOLLINGER_LOWER'])
        
        # Price to MA Ratios
        df['PRICE_TO_SMA20'] = data['Close'] / df['SMA_20'] - 1
        df['PRICE_TO_SMA50'] = data['Close'] / df['SMA_50'] - 1
        
        # Volume Indicators
        volume_ma = data['Volume'].rolling(window=20, min_periods=1).mean()
        df['VOLUME_TO_MA'] = data['Volume'] / volume_ma
        df['VOLUME_TREND'] = data['Volume'].pct_change().rolling(window=5, min_periods=1).mean()
        
        # Fill NaN values
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        return df


    def optimize_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray, n_trials: int = 50) -> Dict:
        """Optimize hyperparameters with improved class balance handling"""
        def objective(trial):
            # Calculate class weights
            n_samples = len(y_train)
            n_positive = np.sum(y_train)
            class_weight = {0: 1.0, 1: n_samples / (2 * n_positive) if n_positive > 0 else 1.0}
            scale_pos_weight = (n_samples - n_positive) / n_positive if n_positive > 0 else 1.0

            # XGBoost parameters with class balance
            xgb_params = {
                'n_estimators': trial.suggest_int('xgb_n_estimators', 200, 2000),
                'max_depth': trial.suggest_int('xgb_max_depth', 3, 8),
                'learning_rate': trial.suggest_float('xgb_learning_rate', 1e-4, 1e-1, log=True),
                'subsample': trial.suggest_float('xgb_subsample', 0.7, 1.0),
                'colsample_bytree': trial.suggest_float('xgb_colsample_bytree', 0.7, 1.0),
                'min_child_weight': trial.suggest_int('xgb_min_child_weight', 1, 5),
                'gamma': trial.suggest_float('xgb_gamma', 1e-8, 0.5, log=True),
                'reg_alpha': trial.suggest_float('xgb_reg_alpha', 1e-8, 1.0, log=True),
                'reg_lambda': trial.suggest_float('xgb_reg_lambda', 1e-8, 1.0, log=True),
                'scale_pos_weight': scale_pos_weight
            }
            
            # LightGBM parameters with class balance
            lgb_params = {
                'n_estimators': trial.suggest_int('lgb_n_estimators', 200, 2000),
                'max_depth': trial.suggest_int('lgb_max_depth', 3, 8),
                'learning_rate': trial.suggest_float('lgb_learning_rate', 1e-4, 1e-1, log=True),
                'subsample': trial.suggest_float('lgb_subsample', 0.7, 1.0),
                'colsample_bytree': trial.suggest_float('lgb_colsample_bytree', 0.7, 1.0),
                'min_child_samples': trial.suggest_int('lgb_min_child_samples', 10, 50),
                'reg_alpha': trial.suggest_float('lgb_reg_alpha', 1e-8, 1.0, log=True),
                'reg_lambda': trial.suggest_float('lgb_reg_lambda', 1e-8, 1.0, log=True),
                'min_split_gain': trial.suggest_float('lgb_min_split_gain', 1e-8, 0.5, log=True),
                'num_leaves': trial.suggest_int('lgb_num_leaves', 16, 96),
                'is_unbalance': True  # Handle unbalanced datasets
            }
            
            cv = TimeSeriesSplit(n_splits=5)
            
            try:
                # Initialize models with class weights
                xgb = XGBClassifier(**xgb_params, random_state=42)
                lgb = LGBMClassifier(**lgb_params, random_state=42, verbose=-1)
                
                scores = []
                for train_idx, val_idx in cv.split(X_train):
                    X_train_cv = X_train[train_idx]
                    X_val_cv = X_train[val_idx]
                    y_train_cv = y_train.iloc[train_idx] if isinstance(y_train, pd.Series) else y_train[train_idx]
                    y_val_cv = y_train.iloc[val_idx] if isinstance(y_train, pd.Series) else y_train[val_idx]
                    
                    # Train models
                    xgb.fit(X_train_cv, y_train_cv)
                    lgb.fit(X_train_cv, y_train_cv)
                    
                    # Get predictions
                    xgb_proba = xgb.predict_proba(X_val_cv)[:, 1]
                    lgb_proba = lgb.predict_proba(X_val_cv)[:, 1]
                    
                    # Ensemble predictions with adjusted threshold
                    ensemble_proba = 0.6 * xgb_proba + 0.4 * lgb_proba
                    predictions = (ensemble_proba > 0.4).astype(int)  # Lower threshold to encourage more positive predictions
                    
                    # Calculate metrics with zero_division parameter
                    accuracy = accuracy_score(y_val_cv, predictions)
                    precision = precision_score(y_val_cv, predictions, zero_division=0)
                    recall = recall_score(y_val_cv, predictions, zero_division=0)
                    f1 = f1_score(y_val_cv, predictions, zero_division=0)
                    
                    # Custom score that balances different metrics
                    score = (
                        accuracy * 0.2 +  # Reduce weight of accuracy
                        precision * 0.3 +  # Emphasize precision
                        recall * 0.3 +     # Emphasize recall
                        f1 * 0.2           # Keep F1 score
                    )
                    
                    # Add penalty for no predictions
                    if np.sum(predictions) == 0:
                        score *= 0.5  # Penalize for making no positive predictions
                    
                    scores.append(score)
                
                return np.mean(scores)
                
            except Exception as e:
                logger.error(f"Error in hyperparameter optimization: {str(e)}")
                return float('-inf')
        
        # Use TPE sampler with multivariate optimization
        sampler = optuna.samplers.TPESampler(seed=42, multivariate=True)
        study = optuna.create_study(direction='maximize', sampler=sampler)
        
        try:
            study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
            
            logger.info(f"Best trial: {study.best_trial.number}. Best value: {study.best_trial.value:.6f}")
            logger.info("Best hyperparameters:")
            for key, value in study.best_trial.params.items():
                logger.info(f"    {key}: {value}")
                
            return study.best_trial.params
            
        except Exception as e:
            logger.error(f"Error in hyperparameter optimization: {str(e)}")
            raise

    def train_ensemble(self, X_train: np.ndarray, y_train: np.ndarray, 
                    X_test: np.ndarray, y_test: np.ndarray, 
                    best_params: Dict) -> Dict:
        """Train ensemble of models and return performance metrics"""
        try:
            # Extract parameters
            xgb_params = {k[4:]: v for k, v in best_params.items() if k.startswith('xgb_')}
            lgb_params = {k[4:]: v for k, v in best_params.items() if k.startswith('lgb_')}
            
            # Initialize models with suppress_warnings for LightGBM
            models = {
                'xgboost': XGBClassifier(**xgb_params, random_state=42),
                'lightgbm': LGBMClassifier(**lgb_params, random_state=42, verbose=-1),
                'random_forest': RandomForestClassifier(n_estimators=500, random_state=42),
                'gradient_boosting': GradientBoostingClassifier(n_estimators=200, random_state=42)
            }
            
            results = {}
            predictions = {}
            
            for name, model in models.items():
                logger.info(f"Training {name}...")
                
                # Convert target to numpy array if it's a pandas Series
                y_train_np = y_train.values if isinstance(y_train, pd.Series) else y_train
                y_test_np = y_test.values if isinstance(y_test, pd.Series) else y_test
                
                model.fit(X_train, y_train_np)
                y_pred = model.predict(X_test)
                pred_proba = model.predict_proba(X_test)
                
                predictions[name] = pred_proba[:, 1]
                results[name] = {
                    'accuracy': accuracy_score(y_test_np, y_pred),
                    'precision': precision_score(y_test_np, y_pred),
                    'recall': recall_score(y_test_np, y_pred),
                    'f1': f1_score(y_test_np, y_pred)
                }
                
                self.models[name] = model
            
            ensemble_pred_proba = np.mean([pred for pred in predictions.values()], axis=0)
            ensemble_pred = (ensemble_pred_proba > 0.5).astype(int)
            
            results['ensemble'] = {
                'accuracy': accuracy_score(y_test_np, ensemble_pred),
                'precision': precision_score(y_test_np, ensemble_pred),
                'recall': recall_score(y_test_np, ensemble_pred),
                'f1': f1_score(y_test_np, ensemble_pred)
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in ensemble training: {str(e)}")
            raise

        def calculate_feature_importance(self, feature_names: List[str]) -> pd.DataFrame:
            """Calculate and aggregate feature importance from all models"""
            try:
                importance_dict = {}
                
                for name, model in self.models.items():
                    if hasattr(model, 'feature_importances_'):
                        importance_dict[name] = model.feature_importances_
                
                importance_df = pd.DataFrame(importance_dict, index=feature_names)
                importance_df['mean_importance'] = importance_df.mean(axis=1)
                importance_df = importance_df.sort_values('mean_importance', ascending=False)
                
                self.feature_importance = importance_df
                return importance_df
                
            except Exception as e:
                logger.error(f"Error calculating feature importance: {str(e)}")
                raise

        def predict_next_day(self, latest_data: pd.DataFrame) -> Dict:
            """Make prediction for the next day using the ensemble"""
            try:
                if not self.models:
                    raise ValueError("No trained models available. Train models first.")
                    
                scaled_data = self.scaler.transform(latest_data)
                
                predictions = {}
                probabilities = {}
                
                for name, model in self.models.items():
                    pred = model.predict(scaled_data)
                    prob = model.predict_proba(scaled_data)
                    predictions[name] = pred[0]
                    probabilities[name] = prob[0][1]
                
                ensemble_prob = np.mean(list(probabilities.values()))
                ensemble_pred = 1 if ensemble_prob > 0.5 else 0
                
                return {
                    'ensemble_prediction': 'INCREASE' if ensemble_pred == 1 else 'DECREASE',
                    'ensemble_probability': float(ensemble_prob),
                    'individual_predictions': {
                        name: {
                            'prediction': 'INCREASE' if pred == 1 else 'DECREASE',
                            'probability': float(prob)
                        }
                        for name, pred, prob in zip(predictions.keys(),
                                                predictions.values(),
                                                probabilities.values())
                    }
                }
                
            except Exception as e:
                logger.error(f"Error making prediction: {str(e)}")
                raise

        def save_model(self, filepath: str) -> None:
            """Save the entire model state"""
            try:
                state = {
                    'models': self.models,
                    'scaler': self.scaler,
                    'feature_importance': self.feature_importance
                }
                joblib.dump(state, filepath)
                logger.info(f"Model saved successfully to {filepath}")
                
            except Exception as e:
                logger.error(f"Error saving model: {str(e)}")
                raise

        @classmethod
        def load_model(cls, filepath: str) -> 'AdvancedStockPredictor':
            """Load a saved model"""
            try:
                state = joblib.load(filepath)
                predictor = cls()
                predictor.models = state['models']
                predictor.scaler = state['scaler']
                predictor.feature_importance = state['feature_importance']
                logger.info(f"Model loaded successfully from {filepath}")
                return predictor
                
            except Exception as e:
                logger.error(f"Error loading model: {str(e)}")
                raise

    def calculate_feature_importance(self, feature_names: List[str]) -> pd.DataFrame:
        """
        Calculate and aggregate feature importance from all models
        
        Args:
            feature_names (List[str]): List of feature names
            
        Returns:
            pd.DataFrame: DataFrame with feature importance scores
        """
        try:
            if not self.models:
                raise ValueError("No trained models available. Train models first.")
                
            importance_dict = {}
            
            # Calculate feature importance for each model that supports it
            for name, model in self.models.items():
                if hasattr(model, 'feature_importances_'):
                    importance_dict[name] = model.feature_importances_
                elif hasattr(model, 'get_score'):  # For XGBoost
                    importance_dict[name] = np.array([model.get_score().get(f, 0) for f in feature_names])
                    importance_dict[name] = importance_dict[name] / importance_dict[name].sum()  # Normalize
            
            # Create DataFrame with feature importance
            importance_df = pd.DataFrame(importance_dict, index=feature_names)
            
            # Calculate mean importance across all models
            importance_df['mean_importance'] = importance_df.mean(axis=1)
            
            # Sort by mean importance
            importance_df = importance_df.sort_values('mean_importance', ascending=False)
            
            # Add rank column
            importance_df['rank'] = range(1, len(importance_df) + 1)
            
            # Add relative importance (percentage)
            importance_df['importance_percentage'] = (importance_df['mean_importance'] / 
                                                    importance_df['mean_importance'].sum() * 100)
            
            return importance_df
            
        except Exception as e:
            logger.error(f"Error calculating feature importance: {str(e)}")
            raise

    def predict_next_day(self, latest_data: pd.DataFrame) -> Dict:
        """
        Make prediction for the next day using the ensemble
        
        Args:
            latest_data (pd.DataFrame): DataFrame containing the latest feature values
        """
        try:
            if not self.models:
                raise ValueError("No trained models available. Train models first.")
                
            # Ensure latest_data is 2D
            if len(latest_data.shape) == 1:
                latest_data = latest_data.to_frame().T

            # Scale the data
            scaled_data = self.scaler.transform(latest_data)
            
            # Get predictions from each model
            predictions = {}
            probabilities = {}
            
            for name, model in self.models.items():
                try:
                    # Ensure scaled_data is 2D numpy array
                    if len(scaled_data.shape) == 1:
                        scaled_data = scaled_data.reshape(1, -1)
                        
                    pred = model.predict(scaled_data)
                    prob = model.predict_proba(scaled_data)
                    predictions[name] = pred[0]
                    probabilities[name] = prob[0][1]  # Probability of price increase
                except Exception as e:
                    logger.error(f"Error in model {name} prediction: {str(e)}")
                    predictions[name] = 0
                    probabilities[name] = 0.5
            
            # Calculate ensemble prediction
            ensemble_prob = np.mean([v for v in probabilities.values() if v != 0.5])
            if np.isnan(ensemble_prob):
                ensemble_prob = 0.5
            ensemble_pred = 1 if ensemble_prob > 0.5 else 0
            
            # Calculate confidence metrics
            valid_probs = [v for v in probabilities.values() if v != 0.5]
            prob_std = np.std(valid_probs) if valid_probs else 0
            model_agreement = sum(1 for p in predictions.values() if p == ensemble_pred) / len(predictions)
            
            result = {
                'ensemble_prediction': 'INCREASE' if ensemble_pred == 1 else 'DECREASE',
                'ensemble_probability': float(ensemble_prob),
                'prediction_confidence': {
                    'probability_std': float(prob_std),
                    'model_agreement': float(model_agreement)
                },
                'individual_predictions': {
                    name: {
                        'prediction': 'INCREASE' if pred == 1 else 'DECREASE',
                        'probability': float(prob)
                    }
                    for name, pred, prob in zip(predictions.keys(),
                                            predictions.values(),
                                            probabilities.values())
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise
    
    
    def save_model(self, filepath: str) -> None:
        """
        Save the entire model state to disk
        
        Args:
            filepath (str): Path to save the model
        """
        try:
            if not self.models:
                raise ValueError("No trained models available to save.")
                
            # Prepare state dictionary
            state = {
                'models': self.models,
                'scaler': self.scaler,
                'feature_importance': self.feature_importance,
                'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'version': '1.0.0'
            }
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save the state
            joblib.dump(state, filepath)
            logger.info(f"Model saved successfully to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    @classmethod
    def load_model(cls, filepath: str) -> 'AdvancedStockPredictor':
        """
        Load a saved model from disk
        
        Args:
            filepath (str): Path to the saved model
            
        Returns:
            AdvancedStockPredictor: Loaded model instance
        """
        try:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Model file not found: {filepath}")
                
            # Load the state
            state = joblib.load(filepath)
            
            # Create new instance
            predictor = cls()
            
            # Restore state
            predictor.models = state['models']
            predictor.scaler = state['scaler']
            predictor.feature_importance = state.get('feature_importance', None)
            
            logger.info(f"Model loaded successfully from {filepath}")
            logger.info(f"Model version: {state.get('version', 'unknown')}")
            logger.info(f"Saved on: {state.get('timestamp', 'unknown')}")
            
            return predictor
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise



# main.py (continued)
def main():
    """Main function to run the stock prediction system"""
    try:
        import yfinance as yf
        from datetime import timedelta
        # from datetime import datetime, timedelta
        
        # Get stock symbol from user
        symbol = input("Enter stock symbol (e.g., AAPL): ").upper()
        
        # Create output directory
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        # Setup logging to file
        log_file = output_dir / f"stock_prediction_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        
        logger.info(f"Starting prediction process for {symbol}")
        
        # Download historical data
        end_date = datetime.datetime.now()
        start_date = end_date - timedelta(days=365*2)  # 2 years of data
        data = yf.download(symbol, start=start_date, end=end_date)
        
        if data.empty:
            raise ValueError(f"No data found for symbol {symbol}")
        
        logger.info(f"Downloaded {len(data)} days of historical data")
        
        # Calculate technical indicators
        ta = AdvancedTechnicalAnalysis(data)
        feature_data = ta.calculate_all_indicators()
        logger.info("Calculated technical indicators")
        
        # Initialize predictor
        predictor = AdvancedStockPredictor(feature_data)
        
        # Prepare data
        logger.info("Preparing data for training...")
        X_train, X_test, y_train, y_test, feature_names = predictor.prepare_data('Close')
        
        # Optimize hyperparameters
        logger.info("Optimizing hyperparameters...")
        best_params = predictor.optimize_hyperparameters(X_train, y_train)
        logger.info(f"Best parameters: {best_params}")
        
        # Train models
        logger.info("Training ensemble models...")
        results = predictor.train_ensemble(X_train, y_train, X_test, y_test, best_params)
        
        # Calculate feature importance
        importance = predictor.calculate_feature_importance(feature_names)
        
        # Make prediction for next day
        latest_data = feature_data.iloc[-1:].drop('Close', axis=1)
        prediction = predictor.predict_next_day(latest_data)
        
        # Save results
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = output_dir / f"{symbol}_model_{timestamp}.joblib"
        predictor.save_model(model_path)
        
        # Save feature importance plot
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))
        importance['mean_importance'].plot(kind='bar')
        plt.title(f'Feature Importance for {symbol}')
        plt.tight_layout()
        plt.savefig(output_dir / f"{symbol}_feature_importance_{timestamp}.png")
        
        # Print results
        print(f"\nPrediction Results for {symbol}:")
        print(f"Ensemble Prediction: {prediction['ensemble_prediction']}")
        print(f"Confidence: {prediction['ensemble_probability']:.2%}")
        
        print("\nModel Performance:")
        for model, metrics in results.items():
            print(f"\n{model.upper()}:")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")
        
        print("\nTop 10 Most Important Features:")
        print(importance['mean_importance'].head(10))
        
        logger.info("Analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
        raise

if __name__ == "__main__":
    # main()
    pass


# def backtest_strategy(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
#     """
#     Backtest the prediction strategy with proper feature calculation
#     """
#     try:
#         # Download data
#         data = yf.download(symbol, start=start_date, end=end_date)
#         if data.empty:
#             raise ValueError(f"No data found for symbol {symbol}")
            
#         logger.info(f"Downloaded {len(data)} days of historical data for {symbol}")
        
#         # Calculate initial indicators for training
#         ta = AdvancedTechnicalAnalysis(data)
#         feature_data = ta.calculate_all_indicators()
        
#         # Initialize predictor and train models
#         predictor = AdvancedStockPredictor(feature_data)
#         X_train, X_test, y_train, y_test, feature_names = predictor.prepare_data('Close', test_size=0.3)
        
#         logger.info("Optimizing hyperparameters...")
#         best_params = predictor.optimize_hyperparameters(X_train, y_train, n_trials=30)
#         logger.info("Training models...")
#         predictor.train_ensemble(X_train, y_train, X_test, y_test, best_params)
        
#         # Create backtest DataFrame
#         backtest_data = pd.DataFrame(index=data.index)
#         backtest_data['Close'] = data['Close']
#         backtest_data['Returns'] = data['Close'].pct_change()
        
#         # Initialize prediction columns
#         backtest_data['Prediction'] = np.nan
#         backtest_data['Probability'] = np.nan
        
#         # Calculate features and make predictions using rolling window
#         window_size = len(X_train)
#         for i in range(window_size, len(data)-1):
#             try:
#                 # Get data up to current point for feature calculation
#                 current_window = data.iloc[:i+1]
                
#                 # Calculate features for current window
#                 ta_current = AdvancedTechnicalAnalysis(current_window)
#                 current_features = ta_current.calculate_all_indicators()
                
#                 # Get the last row of features
#                 latest_features = current_features.iloc[-1:].copy()  # Ensure we have a copy
#                 latest_features = latest_features[feature_names]  # Select only the required features
                
#                 # Make prediction
#                 pred = predictor.predict_next_day(latest_features)
                
#                 # Store prediction and probability
#                 backtest_data.iloc[i, backtest_data.columns.get_loc('Prediction')] = 1 if pred['ensemble_prediction'] == 'INCREASE' else 0
#                 backtest_data.iloc[i, backtest_data.columns.get_loc('Probability')] = pred['ensemble_probability']
                
#             except Exception as e:
#                 logger.error(f"Error in prediction for day {i}: {str(e)}")
#                 continue
        
#         # Calculate strategy performance
#         valid_predictions = backtest_data['Prediction'].notna()
#         if not valid_predictions.any():
#             raise ValueError("No valid predictions generated during backtesting")
            
#         backtest_data = backtest_data[valid_predictions]
        
#         # Calculate strategy performance
#         backtest_data['Actual'] = (backtest_data['Returns'].shift(-1) > 0).astype(int)
#         backtest_data['Correct'] = (backtest_data['Prediction'] == backtest_data['Actual']).astype(int)
        
#         # Calculate position sizing and returns
#         backtest_data['Position'] = backtest_data['Prediction'].map({1: 1, 0: -1})
#         backtest_data['Position_Size'] = abs(backtest_data['Probability'] - 0.5) * 2
#         backtest_data['Strategy_Returns'] = backtest_data['Position'] * backtest_data['Position_Size'] * backtest_data['Returns'].shift(-1)
        
#         # Calculate cumulative returns
#         backtest_data['Cumulative_Market_Returns'] = (1 + backtest_data['Returns']).cumprod()
#         backtest_data['Cumulative_Strategy_Returns'] = (1 + backtest_data['Strategy_Returns']).cumprod()
        
#         # Calculate performance metrics
#         trading_days = 252
#         backtest_metrics = {
#             'Total_Returns': backtest_data['Strategy_Returns'].sum(),
#             'Annual_Return': ((1 + backtest_data['Strategy_Returns'].sum()) ** (trading_days / len(backtest_data)) - 1),
#             'Sharpe_Ratio': backtest_data['Strategy_Returns'].mean() / backtest_data['Strategy_Returns'].std() * np.sqrt(trading_days),
#             'Max_Drawdown': (backtest_data['Cumulative_Strategy_Returns'] / 
#                            backtest_data['Cumulative_Strategy_Returns'].cummax() - 1).min(),
#             'Accuracy': backtest_data['Correct'].mean(),
#             'Win_Rate': len(backtest_data[backtest_data['Strategy_Returns'] > 0]) / len(backtest_data[backtest_data['Strategy_Returns'] != 0]),
#             'Profit_Factor': abs(backtest_data[backtest_data['Strategy_Returns'] > 0]['Strategy_Returns'].sum() / 
#                                backtest_data[backtest_data['Strategy_Returns'] < 0]['Strategy_Returns'].sum())
#         }
        
#         # Add metrics to the data
#         for metric, value in backtest_metrics.items():
#             backtest_data[metric] = value
        
#         logger.info("Backtesting completed successfully")
#         logger.info(f"Backtest metrics: {backtest_metrics}")
        
#         return backtest_data.dropna()
        
#     except Exception as e:
#         logger.error(f"Error in backtesting: {str(e)}")
#         raise
    

# def backtest_strategy(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
#     """Backtest the prediction strategy"""
#     try:
#         # Download data
#         data = yf.download(symbol, start=start_date, end=end_date)
#         if data.empty:
#             raise ValueError(f"No data found for symbol {symbol}")
            
#         logger.info(f"Downloaded {len(data)} days of historical data for {symbol}")
        
#         # Initialize technical analysis
#         ta = AdvancedTechnicalAnalysis(data)
        
#         # Calculate all features at once
#         feature_data = ta._create_features(data)
        
#         # Initialize predictor and train models
#         predictor = AdvancedStockPredictor(data)  # Initialize with raw data
#         X_train, X_test, y_train, y_test, feature_names = predictor.prepare_data('Close', test_size=0.3)
        
#         logger.info("Optimizing hyperparameters...")
#         best_params = predictor.optimize_hyperparameters(X_train, y_train, n_trials=30)
#         logger.info("Training models...")
#         predictor.train_ensemble(X_train, y_train, X_test, y_test, best_params)
        
#         # Create backtest DataFrame
#         backtest_data = pd.DataFrame(index=data.index)
#         backtest_data['Close'] = data['Close']
#         backtest_data['Returns'] = data['Close'].pct_change()
        
#         # Initialize prediction columns
#         backtest_data['Prediction'] = np.nan
#         backtest_data['Probability'] = np.nan
        
#         # Calculate features and make predictions
#         window_size = len(X_train)
#         for i in range(window_size, len(data)-1):
#             try:
#                 # Get features for current window
#                 current_features = feature_data.iloc[i:i+1][feature_names]
                
#                 # Make prediction
#                 pred = predictor.predict_next_day(current_features)
                
#                 # Store prediction and probability
#                 backtest_data.iloc[i, backtest_data.columns.get_loc('Prediction')] = 1 if pred['ensemble_prediction'] == 'INCREASE' else 0
#                 backtest_data.iloc[i, backtest_data.columns.get_loc('Probability')] = pred['ensemble_probability']
                
#             except Exception as e:
#                 logger.error(f"Error in prediction for day {i}: {str(e)}")
#                 continue
        
#         # Remove rows without predictions
#         backtest_data = backtest_data.dropna(subset=['Prediction'])
        
#         if len(backtest_data) == 0:
#             raise ValueError("No valid predictions generated during backtesting")
            
#         # Calculate strategy performance
#         backtest_data['Actual'] = (backtest_data['Returns'].shift(-1) > 0).astype(int)
#         backtest_data['Correct'] = (backtest_data['Prediction'] == backtest_data['Actual']).astype(int)
        
#         # Position sizing and returns
#         backtest_data['Position'] = backtest_data['Prediction'].map({1: 1, 0: -1})
#         backtest_data['Position_Size'] = abs(backtest_data['Probability'] - 0.5) * 2
#         backtest_data['Strategy_Returns'] = backtest_data['Position'] * backtest_data['Position_Size'] * backtest_data['Returns'].shift(-1)
        
#         # Performance metrics
#         metrics = calculate_performance_metrics(backtest_data)
        
#         return backtest_data
        
#     except Exception as e:
#         logger.error(f"Error in backtesting: {str(e)}")
#         raise

def backtest_strategy(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Backtest the prediction strategy"""
    try:
        # Download data
        data = yf.download(symbol, start=start_date, end=end_date)
        if data.empty:
            raise ValueError(f"No data found for symbol {symbol}")
            
        logger.info(f"Downloaded {len(data)} days of historical data for {symbol}")
        
        # Initialize technical analysis
        ta = AdvancedTechnicalAnalysis(data)
        
        # Calculate all features at once
        feature_data = ta._create_features(data)
        
        # Initialize predictor and train models
        predictor = AdvancedStockPredictor(data)  # Initialize with raw data
        X_train, X_test, y_train, y_test, feature_names = predictor.prepare_data('Close', test_size=0.3)
        
        logger.info("Optimizing hyperparameters...")
        best_params = predictor.optimize_hyperparameters(X_train, y_train, n_trials=30)
        logger.info("Training models...")
        predictor.train_ensemble(X_train, y_train, X_test, y_test, best_params)
        
        # Create backtest DataFrame
        backtest_data = pd.DataFrame(index=data.index)
        backtest_data['Close'] = data['Close']
        backtest_data['Returns'] = data['Close'].pct_change()
        
        # Initialize prediction columns
        backtest_data['Prediction'] = np.nan
        backtest_data['Probability'] = np.nan
        
        # Calculate features and make predictions
        window_size = len(X_train)
        for i in range(window_size, len(data)-1):
            try:
                # Get features for current window
                current_features = feature_data.iloc[i:i+1][feature_names].copy()
                
                # Make prediction
                pred = predictor.predict_next_day(current_features)
                
                # Store prediction and probability
                backtest_data.iloc[i, backtest_data.columns.get_loc('Prediction')] = 1 if pred['ensemble_prediction'] == 'INCREASE' else 0
                backtest_data.iloc[i, backtest_data.columns.get_loc('Probability')] = pred['ensemble_probability']
                
            except Exception as e:
                logger.error(f"Error in prediction for day {i}: {str(e)}")
                continue
        
        # Remove rows without predictions
        backtest_data = backtest_data.dropna(subset=['Prediction'])
        
        if len(backtest_data) == 0:
            raise ValueError("No valid predictions generated during backtesting")
            
        # Calculate strategy performance
        backtest_data['Actual'] = (backtest_data['Returns'].shift(-1) > 0).astype(int)
        backtest_data['Correct'] = (backtest_data['Prediction'] == backtest_data['Actual']).astype(int)
        
        # Position sizing and returns
        backtest_data['Position'] = backtest_data['Prediction'].map({1: 1, 0: -1})
        backtest_data['Position_Size'] = abs(backtest_data['Probability'] - 0.5) * 2
        backtest_data['Strategy_Returns'] = backtest_data['Position'] * backtest_data['Position_Size'] * backtest_data['Returns'].shift(-1)
        
        # Calculate cumulative returns
        backtest_data['Cumulative_Market_Returns'] = (1 + backtest_data['Returns']).cumprod()
        backtest_data['Cumulative_Strategy_Returns'] = (1 + backtest_data['Strategy_Returns']).cumprod()
        
        # Performance metrics
        metrics = calculate_performance_metrics(backtest_data)
        for key, value in metrics.items():
            backtest_data[key] = value
            logger.info(f"{key}: {value:.4f}")
        
        return backtest_data
        
    except Exception as e:
        logger.error(f"Error in backtesting: {str(e)}")
        raise



def calculate_performance_metrics(backtest_data: pd.DataFrame) -> Dict:
    """Calculate performance metrics for the backtest"""
    trading_days = 252
    return {
        'Total_Returns': backtest_data['Strategy_Returns'].sum(),
        'Annual_Return': ((1 + backtest_data['Strategy_Returns'].sum()) ** (trading_days / len(backtest_data)) - 1),
        'Sharpe_Ratio': backtest_data['Strategy_Returns'].mean() / backtest_data['Strategy_Returns'].std() * np.sqrt(trading_days),
        'Max_Drawdown': (backtest_data['Cumulative_Strategy_Returns'] / 
                        backtest_data['Cumulative_Strategy_Returns'].cummax() - 1).min(),
        'Accuracy': backtest_data['Correct'].mean(),
        'Win_Rate': len(backtest_data[backtest_data['Strategy_Returns'] > 0]) / len(backtest_data[backtest_data['Strategy_Returns'] != 0]),
        'Profit_Factor': abs(backtest_data[backtest_data['Strategy_Returns'] > 0]['Strategy_Returns'].sum() / 
                           backtest_data[backtest_data['Strategy_Returns'] < 0]['Strategy_Returns'].sum())
    }
    
def run_backtest_example():
    """Run a backtest example"""
    try:
        symbol = "AAPL"
        start_date = "2022-01-01"
        end_date = "2023-12-31"
        
        results = backtest_strategy(symbol, start_date, end_date)
        
        # Print performance metrics
        print(f"\nBacktest Results for {symbol}:")
        print(f"Total Return: {results['Total_Returns'].iloc[-1]:.2%}")
        print(f"Sharpe Ratio: {results['Sharpe_Ratio'].iloc[-1]:.2f}")
        print(f"Maximum Drawdown: {results['Max_Drawdown'].iloc[-1]:.2%}")
        print(f"Prediction Accuracy: {results['Accuracy'].iloc[-1]:.2%}")
        print(f"Profit Factor: {results['Profit_Factor'].iloc[-1]:.2f}")
        
        # Plot returns
        plt.figure(figsize=(15, 7))
        plt.plot(results.index, results['Cumulative_Market_Returns'], label='Buy & Hold')
        plt.plot(results.index, results['Cumulative_Strategy_Returns'], label='Strategy')
        plt.title(f'Cumulative Returns: Strategy vs Buy & Hold ({symbol})')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        logger.error(f"Error in backtest example: {str(e)}")
        raise


if __name__ == "__main__":
    # Run either the main prediction system or backtest
    choice = input("Enter 1 for prediction or 2 for backtest: ")
    if choice == "1":
        main()
    elif choice == "2":
        run_backtest_example()
    else:
        print("Invalid choice")