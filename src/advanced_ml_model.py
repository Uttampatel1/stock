# technical_analysis.py
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple

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
        Prepare data for training
        
        Args:
            target_column (str): Column name for prediction target
            test_size (float): Proportion of data to use for testing
            
        Returns:
            Tuple containing training and test data arrays and feature names
        """
        if self.data is None:
            raise ValueError("No data available. Initialize with feature_data or load a saved model.")
            
        # Create target (1 if price increases, 0 if decreases)
        y = (self.data[target_column].shift(-1) > self.data[target_column]).astype(int)
        
        # Remove target column from features
        X = self.data.drop(columns=[target_column])
        
        # Remove NaN values
        X = X.dropna()
        y = y[X.index]
        
        # Split data
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:-1]
        y_train, y_test = y[:split_idx], y[split_idx:-1]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, list(X.columns)


    def optimize_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray, n_trials: int = 50) -> Dict:
        """
        Optimize hyperparameters using Optuna
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training targets
            n_trials (int): Number of optimization trials
            
        Returns:
            Dict of best parameters
        """
        def objective(trial):
            # XGBoost parameters
            xgb_params = {
                'n_estimators': trial.suggest_int('xgb_n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('xgb_max_depth', 3, 10),
                'learning_rate': trial.suggest_float('xgb_learning_rate', 1e-3, 1e-1, log=True),
                'subsample': trial.suggest_float('xgb_subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('xgb_colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('xgb_min_child_weight', 1, 7),
                'gamma': trial.suggest_float('xgb_gamma', 1e-8, 1.0, log=True),
                'reg_alpha': trial.suggest_float('xgb_reg_alpha', 1e-8, 1.0, log=True),
                'reg_lambda': trial.suggest_float('xgb_reg_lambda', 1e-8, 1.0, log=True)
            }
            
            # LightGBM parameters
            lgb_params = {
                'n_estimators': trial.suggest_int('lgb_n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('lgb_max_depth', 3, 10),
                'learning_rate': trial.suggest_float('lgb_learning_rate', 1e-3, 1e-1, log=True),
                'subsample': trial.suggest_float('lgb_subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('lgb_colsample_bytree', 0.6, 1.0),
                'min_child_samples': trial.suggest_int('lgb_min_child_samples', 5, 30),
                'reg_alpha': trial.suggest_float('lgb_reg_alpha', 1e-8, 1.0, log=True),
                'reg_lambda': trial.suggest_float('lgb_reg_lambda', 1e-8, 1.0, log=True),
                'min_split_gain': trial.suggest_float('lgb_min_split_gain', 1e-8, 1.0, log=True)
            }
            
            cv = TimeSeriesSplit(n_splits=5)
            
            try:
                # XGBoost CV score
                xgb = XGBClassifier(**xgb_params, random_state=42)
                xgb_scores = []
                
                # LightGBM CV score
                lgb = LGBMClassifier(**lgb_params, 
                                random_state=42,
                                verbose=-1)  # Suppress LightGBM warnings
                lgb_scores = []
                
                # Perform cross-validation
                for train_idx, val_idx in cv.split(X_train):
                    X_train_cv = X_train[train_idx]
                    X_val_cv = X_train[val_idx]
                    y_train_cv = y_train.iloc[train_idx] if isinstance(y_train, pd.Series) else y_train[train_idx]
                    y_val_cv = y_train.iloc[val_idx] if isinstance(y_train, pd.Series) else y_train[val_idx]
                    
                    # Train and evaluate XGBoost
                    xgb.fit(X_train_cv, y_train_cv)
                    xgb_scores.append(xgb.score(X_val_cv, y_val_cv))
                    
                    # Train and evaluate LightGBM
                    lgb.fit(X_train_cv, y_train_cv)
                    lgb_scores.append(lgb.score(X_val_cv, y_val_cv))
                
                # Weight the scores (0.6 for XGBoost, 0.4 for LightGBM)
                xgb_mean = np.mean(xgb_scores)
                lgb_mean = np.mean(lgb_scores)
                
                # Add a small penalty for complexity to prevent overfitting
                complexity_penalty = (xgb_params['max_depth'] + lgb_params['max_depth']) / 200
                
                return xgb_mean * 0.6 + lgb_mean * 0.4 - complexity_penalty
                
            except Exception as e:
                logger.error(f"Error in hyperparameter optimization: {str(e)}")
                return float('-inf')
        
        # Create study with different sampler for better exploration
        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(direction='maximize', sampler=sampler)
        
        # Run optimization
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        logger.info(f"Best trial: {study.best_trial.params}")
        logger.info(f"Best score: {study.best_trial.value}")
        
        return study.best_trial.params

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
            
        Returns:
            Dict: Prediction results including ensemble and individual model predictions
        """
        try:
            if not self.models:
                raise ValueError("No trained models available. Train models first.")
                
            # Scale the data
            scaled_data = self.scaler.transform(latest_data)
            
            # Get predictions from each model
            predictions = {}
            probabilities = {}
            
            for name, model in self.models.items():
                pred = model.predict(scaled_data)
                prob = model.predict_proba(scaled_data)
                predictions[name] = pred[0]
                probabilities[name] = prob[0][1]  # Probability of price increase
            
            # Calculate ensemble prediction
            ensemble_prob = np.mean(list(probabilities.values()))
            ensemble_pred = 1 if ensemble_prob > 0.5 else 0
            
            # Calculate confidence metrics
            prob_std = np.std(list(probabilities.values()))
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
            
            # Add confidence level based on probability and agreement
            if model_agreement > 0.8 and abs(ensemble_prob - 0.5) > 0.2:
                confidence_level = 'HIGH'
            elif model_agreement > 0.6 and abs(ensemble_prob - 0.5) > 0.1:
                confidence_level = 'MEDIUM'
            else:
                confidence_level = 'LOW'
                
            result['confidence_level'] = confidence_level
            
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
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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
        from datetime import datetime, timedelta
        
        # Get stock symbol from user
        symbol = input("Enter stock symbol (e.g., AAPL): ").upper()
        
        # Create output directory
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        # Setup logging to file
        log_file = output_dir / f"stock_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        
        logger.info(f"Starting prediction process for {symbol}")
        
        # Download historical data
        end_date = datetime.now()
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
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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
    main()

# Example usage for backtesting
def backtest_strategy(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Backtest the prediction strategy
    
    Args:
        symbol (str): Stock symbol
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        
    Returns:
        pd.DataFrame: DataFrame with backtesting results
    """
    try:
        # Download data
        data = yf.download(symbol, start=start_date, end=end_date)
        
        # Calculate indicators
        ta = AdvancedTechnicalAnalysis(data)
        feature_data = ta.calculate_all_indicators()
        
        # Initialize predictor
        predictor = AdvancedStockPredictor(feature_data)
        
        # Prepare data with smaller test size for more backtesting data
        X_train, X_test, y_train, y_test, feature_names = predictor.prepare_data('Close', test_size=0.3)
        
        # Train models
        best_params = predictor.optimize_hyperparameters(X_train, y_train, n_trials=30)
        predictor.train_ensemble(X_train, y_train, X_test, y_test, best_params)
        
        # Create backtest DataFrame
        backtest_data = data[int(len(data) * 0.7):].copy()  # Use test portion
        predictions = []
        probabilities = []
        
        # Make predictions for each day
        for i in range(len(backtest_data) - 1):
            current_features = feature_data.iloc[i:i+1].drop('Close', axis=1)
            pred = predictor.predict_next_day(current_features)
            predictions.append(1 if pred['ensemble_prediction'] == 'INCREASE' else 0)
            probabilities.append(pred['ensemble_probability'])
        
        backtest_data = backtest_data[:-1]  # Remove last day since we don't have next day's price
        backtest_data['Predicted_Direction'] = predictions
        backtest_data['Prediction_Probability'] = probabilities
        backtest_data['Actual_Direction'] = (backtest_data['Close'].shift(-1) > backtest_data['Close']).astype(int)
        backtest_data['Correct_Prediction'] = (backtest_data['Predicted_Direction'] == backtest_data['Actual_Direction']).astype(int)
        
        # Calculate returns
        backtest_data['Strategy_Return'] = backtest_data['Correct_Prediction'] * abs(backtest_data['Close'].pct_change())
        backtest_data['Buy_Hold_Return'] = backtest_data['Close'].pct_change()
        
        return backtest_data
        
    except Exception as e:
        logger.error(f"Error in backtesting: {str(e)}")
        raise

# Example of running backtest
def run_backtest_example():
    """Run a backtest example"""
    symbol = "AAPL"
    start_date = "2022-01-01"
    end_date = "2023-12-31"
    
    results = backtest_strategy(symbol, start_date, end_date)
    
    # Calculate performance metrics
    strategy_return = results['Strategy_Return'].sum()
    buy_hold_return = results['Buy_Hold_Return'].sum()
    accuracy = results['Correct_Prediction'].mean()
    
    print(f"\nBacktest Results for {symbol}:")
    print(f"Strategy Return: {strategy_return:.2%}")
    print(f"Buy & Hold Return: {buy_hold_return:.2%}")
    print(f"Prediction Accuracy: {accuracy:.2%}")
    
    # Plot cumulative returns
    plt.figure(figsize=(12, 6))
    ((1 + results['Strategy_Return']).cumprod()).plot(label='Strategy')
    ((1 + results['Buy_Hold_Return']).cumprod()).plot(label='Buy & Hold')
    plt.title(f'Cumulative Returns: Strategy vs Buy & Hold ({symbol})')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Run either the main prediction system or backtest
    choice = input("Enter 1 for prediction or 2 for backtest: ")
    if choice == "1":
        main()
    elif choice == "2":
        run_backtest_example()
    else:
        print("Invalid choice")