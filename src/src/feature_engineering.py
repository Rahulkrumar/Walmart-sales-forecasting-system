"""
Feature Engineering Module
Creates 80+ time-series features
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Advanced feature engineering for time series"""
    
    def __init__(self, df):
        self.df = df.copy()
        self.df = self.df.sort_values(['Store', 'Dept', 'Date']).reset_index(drop=True)
        
    def create_time_features(self):
        """Extract time-based features"""
        logger.info("Creating time features...")
        
        df = self.df
        
        # Basic datetime components
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Week'] = df['Date'].dt.isocalendar().week
        df['Day'] = df['Date'].dt.day
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['Quarter'] = df['Date'].dt.quarter
        
        # Binary indicators
        df['IsMonthEnd'] = df['Date'].dt.is_month_end.astype(int)
        df['IsQuarterEnd'] = df['Month'].isin([3, 6, 9, 12]).astype(int)
        df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
        
        # Cyclical encoding
        df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        df['Week_Sin'] = np.sin(2 * np.pi * df['Week'] / 52)
        df['Week_Cos'] = np.cos(2 * np.pi * df['Week'] / 52)
        
        self.df = df
        return self
    
    def create_lag_features(self, target_col='Weekly_Sales'):
        """Create lag features"""
        logger.info("Creating lag features...")
        
        df = self.df
        lags = [1, 2, 3, 4, 5, 8, 12, 26, 52]
        
        for lag in lags:
            df[f'{target_col}_Lag_{lag}'] = df.groupby(['Store', 'Dept'])[target_col].shift(lag)
        
        # Lag differences
        df[f'{target_col}_Diff_1'] = df[target_col] - df[f'{target_col}_Lag_1']
        
        self.df = df
        return self
    
    def create_rolling_features(self, target_col='Weekly_Sales'):
        """Create rolling window statistics"""
        logger.info("Creating rolling features...")
        
        df = self.df
        windows = [4, 8, 12, 26, 52]
        
        for window in windows:
            # Rolling mean
            df[f'{target_col}_RollingMean_{window}'] = (
                df.groupby(['Store', 'Dept'])[target_col]
                .transform(lambda x: x.rolling(window=window, min_periods=1).mean())
            )
            
            # Rolling std
            df[f'{target_col}_RollingStd_{window}'] = (
                df.groupby(['Store', 'Dept'])[target_col]
                .transform(lambda x: x.rolling(window=window, min_periods=1).std())
            )
            
            # Rolling min/max
            df[f'{target_col}_RollingMin_{window}'] = (
                df.groupby(['Store', 'Dept'])[target_col]
                .transform(lambda x: x.rolling(window=window, min_periods=1).min())
            )
            
            df[f'{target_col}_RollingMax_{window}'] = (
                df.groupby(['Store', 'Dept'])[target_col]
                .transform(lambda x: x.rolling(window=window, min_periods=1).max())
            )
        
        self.df = df
        return self
    
    def create_aggregated_features(self, target_col='Weekly_Sales'):
        """Create aggregated features"""
        logger.info("Creating aggregated features...")
        
        df = self.df
        
        # Store-level
        df['Store_Mean_Sales'] = df.groupby('Store')[target_col].transform('mean')
        df['Store_Std_Sales'] = df.groupby('Store')[target_col].transform('std')
        
        # Department-level
        df['Dept_Mean_Sales'] = df.groupby('Dept')[target_col].transform('mean')
        df['Dept_Std_Sales'] = df.groupby('Dept')[target_col].transform('std')
        
        # Store-Dept combination
        df['StoreDept_Mean_Sales'] = df.groupby(['Store', 'Dept'])[target_col].transform('mean')
        df['StoreDept_Std_Sales'] = df.groupby(['Store', 'Dept'])[target_col].transform('std')
        
        self.df = df
        return self
    
    def create_interaction_features(self):
        """Create interaction features"""
        logger.info("Creating interaction features...")
        
        df = self.df
        
        # Holiday interactions
        if 'IsHoliday' in df.columns:
            df['Holiday_Month'] = df['IsHoliday'] * df['Month']
            if 'Type_Encoded' in df.columns:
                df['Holiday_StoreType'] = df['IsHoliday'] * df['Type_Encoded']
        
        # Size interactions
        if 'Size' in df.columns:
            if 'Temperature' in df.columns:
                df['Size_Temperature'] = df['Size'] * df['Temperature'] / 1e6
            if 'Unemployment' in df.columns:
                df['Size_Unemployment'] = df['Size'] * df['Unemployment'] / 1e5
        
        self.df = df
        return self
    
    def create_holiday_features(self):
        """Create holiday proximity features"""
        logger.info("Creating holiday features...")
        
        df = self.df
        
        if 'IsHoliday' in df.columns:
            df['Holiday_Next_Week'] = df.groupby(['Store', 'Dept'])['IsHoliday'].shift(-1).fillna(0).astype(int)
            df['Holiday_Prev_Week'] = df.groupby(['Store', 'Dept'])['IsHoliday'].shift(1).fillna(0).astype(int)
        
        self.df = df
        return self
    
    def create_all_features(self):
        """Create all features"""
        logger.info("="*60)
        logger.info("CREATING ALL FEATURES")
        logger.info("="*60)
        
        initial_cols = len(self.df.columns)
        
        self.create_time_features()
        self.create_lag_features()
        self.create_rolling_features()
        self.create_aggregated_features()
        self.create_interaction_features()
        self.create_holiday_features()
        
        final_cols = len(self.df.columns)
        
        logger.info("="*60)
        logger.info(f"Initial features: {initial_cols}")
        logger.info(f"Final features: {final_cols}")
        logger.info(f"New features created: {final_cols - initial_cols}")
        logger.info("="*60)
        
        return self.df
    
    def save_features(self, output_path='data/features/train_features.csv'):
        """Save engineered features"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.df.to_csv(output_path, index=False)
        logger.info(f"Features saved to: {output_path}")


if __name__ == "__main__":
    # Load processed data
    logger.info("Loading processed data...")
    df = pd.read_csv('data/processed/processed_train.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Create features
    engineer = FeatureEngineer(df)
    df_features = engineer.create_all_features()
    
    # Save
    engineer.save_features('data/features/train_features.csv')
    
    print("\nFeature engineering complete!")
    print(f"Total features: {len(df_features.columns)}")
    print("\nFirst 5 rows:")
    print(df_features.head())
