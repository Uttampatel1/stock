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


