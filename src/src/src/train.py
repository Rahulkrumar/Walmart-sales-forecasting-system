"""
Model Training Module
Trains XGBoost and LightGBM models
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import joblib
from datetime import datetime

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WalmartSalesModel:
    """Sales forecasting model trainer"""
    
    def __init__(self, data_path='data/features/train_features.csv'):
        self.data_path = data_path
        self.df = None
        self.feature_cols = None
        self.models = {}
        
    def load_data(self):
        """Load engineered features"""
        logger.info(f"Loading data from {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        logger.info(f"Loaded {len(self.df)} records")
        return self
    
    def prepare_features(self, target_col='Weekly_Sales'):
        """Prepare feature matrix and target"""
        logger.info("Preparing features...")
        
        df = self.df.copy()
        
        # Exclude columns
        exclude_cols = [target_col, 'Date', 'Type']
        
        # Remove high-null features
        null_pct = df.isnull().mean()
        high_null_cols = null_pct[null_pct > 0.2].index.tolist()
        exclude_cols.extend(high_null_cols)
        
        # Get feature columns
        self.feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        logger.info(f"Using {len(self.feature_cols)} features")
        
        # Create X and y
        X = df[self.feature_cols].fillna(0)
        y = df[target_col]
        dates = df['Date']
        
        # Remove NaN targets
        valid_idx = ~y.isna()
        X = X[valid_idx]
        y = y[valid_idx]
        dates = dates[valid_idx]
        
        return X, y, dates
    
    def train_test_split(self, X, y, dates, cutoff_date='2012-08-01'):
        """Time-based train-test split"""
        logger.info(f"Splitting data at: {cutoff_date}")
        
        cutoff = pd.to_datetime(cutoff_date)
        train_mask = dates < cutoff
        test_mask = dates >= cutoff
        
        X_train = X[train_mask]
        X_test = X[test_mask]
        y_train = y[train_mask]
        y_test = y[test_mask]
        
        logger.info(f"Train: {len(X_train)} records")
        logger.info(f"Test: {len(X_test)} records")
        
        return X_train, X_test, y_train, y_test
    
    def train_xgboost(self, X_train, y_train, X_test, y_test):
        """Train XGBoost model"""
        logger.info("="*60)
        logger.info("TRAINING XGBOOST")
        logger.info("="*60)
        
        params = {
            'objective': 'reg:squarederror',
            'learning_rate': 0.05,
            'max_depth': 8,
            'min_child_weight': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'n_estimators': 500,
            'random_state': 42,
            'n_jobs': -1
        }
        
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        
        self.models['xgboost'] = model
        logger.info("XGBoost training complete!")
        
        return model
    
    def train_lightgbm(self, X_train, y_train, X_test, y_test):
        """Train LightGBM model"""
        logger.info("="*60)
        logger.info("TRAINING LIGHTGBM")
        logger.info("="*60)
        
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'learning_rate': 0.05,
            'num_leaves': 63,
            'max_depth': 10,
            'min_data_in_leaf': 20,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'n_estimators': 500,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
        
        self.models['lightgbm'] = model
        logger.info("LightGBM training complete!")
        
        return model
    
    def evaluate_model(self, model, X_test, y_test, model_name="Model"):
        """Evaluate model performance"""
        logger.info(f"\nEvaluating {model_name}...")
        
        y_pred = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        r2 = r2_score(y_test, y_pred)
        
        logger.info(f"\n{model_name} Performance:")
        logger.info(f"  RMSE: ${rmse:,.2f}")
        logger.info(f"  MAE: ${mae:,.2f}")
        logger.info(f"  MAPE: {mape:.2f}%")
        logger.info(f"  R²: {r2:.4f}")
        
        return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R²': r2}
    
    def save_models(self, output_dir='models'):
        """Save trained models"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving models to {output_dir}...")
        
        for model_name, model in self.models.items():
            model_path = output_dir / f'{model_name}_model.pkl'
            joblib.dump(model, model_path)
            logger.info(f"  Saved {model_name}")
        
        # Save feature columns
        feature_path = output_dir / 'feature_columns.pkl'
        joblib.dump(self.feature_cols, feature_path)
        logger.info(f"  Saved feature columns")
        
        logger.info("All models saved!")
    
    def train_all(self):
        """Complete training pipeline"""
        logger.info("="*80)
        logger.info("STARTING TRAINING PIPELINE")
        logger.info("="*80)
        
        # Load and prepare data
        self.load_data()
        X, y, dates = self.prepare_features()
        X_train, X_test, y_train, y_test = self.train_test_split(X, y, dates)
        
        # Train models
        xgb_model = self.train_xgboost(X_train, y_train, X_test, y_test)
        xgb_metrics = self.evaluate_model(xgb_model, X_test, y_test, "XGBoost")
        
        lgb_model = self.train_lightgbm(X_train, y_train, X_test, y_test)
        lgb_metrics = self.evaluate_model(lgb_model, X_test, y_test, "LightGBM")
        
        # Save models
        self.save_models()
        
        logger.info("="*80)
        logger.info("TRAINING COMPLETE")
        logger.info("="*80)
        
        return {'xgboost': xgb_metrics, 'lightgbm': lgb_metrics}


if __name__ == "__main__":
    trainer = WalmartSalesModel(data_path='data/features/train_features.csv')
    results = trainer.train_all()
    
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    for model_name, metrics in results.items():
        print(f"\n{model_name.upper()}:")
        for metric_name, value in metrics.items():
            if 'MAPE' in metric_name:
                print(f"  {metric_name}: {value:.2f}%")
            elif metric_name == 'R²':
                print(f"  {metric_name}: {value:.4f}")
            else:
                print(f"  {metric_name}: ${value:,.2f}")
