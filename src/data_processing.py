"""
Data Processing Module for Walmart Sales Forecasting
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WalmartDataProcessor:
    """Data processing pipeline"""
    
    def __init__(self, data_path='data/raw'):
        self.data_path = Path(data_path)
        
    def load_data(self):
        """Load all CSV files"""
        logger.info("Loading datasets...")
        
        self.train_df = pd.read_csv(self.data_path / 'train.csv')
        self.stores_df = pd.read_csv(self.data_path / 'stores.csv')
        self.features_df = pd.read_csv(self.data_path / 'features.csv')
        
        logger.info(f"Train shape: {self.train_df.shape}")
        logger.info(f"Stores shape: {self.stores_df.shape}")
        logger.info(f"Features shape: {self.features_df.shape}")
        
        return self
    
    def convert_dates(self):
        """Convert date columns to datetime"""
        logger.info("Converting dates...")
        
        self.train_df['Date'] = pd.to_datetime(self.train_df['Date'])
        self.features_df['Date'] = pd.to_datetime(self.features_df['Date'])
        
        return self
    
    def handle_missing_values(self):
        """Handle missing values"""
        logger.info("Handling missing values...")
        
        # MarkDown columns - fill with 0
        markdown_cols = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
        for col in markdown_cols:
            if col in self.features_df.columns:
                self.features_df[col].fillna(0, inplace=True)
        
        # Economic indicators - forward fill
        if 'CPI' in self.features_df.columns:
            self.features_df['CPI'].fillna(method='ffill', inplace=True)
            
        if 'Unemployment' in self.features_df.columns:
            self.features_df['Unemployment'].fillna(method='ffill', inplace=True)
        
        # Temperature and Fuel Price - interpolate
        if 'Temperature' in self.features_df.columns:
            self.features_df['Temperature'].interpolate(method='linear', inplace=True)
            
        if 'Fuel_Price' in self.features_df.columns:
            self.features_df['Fuel_Price'].interpolate(method='linear', inplace=True)
        
        return self
    
    def remove_outliers(self):
        """Remove outliers"""
        logger.info("Removing outliers...")
        
        # Remove negative sales
        initial_len = len(self.train_df)
        self.train_df = self.train_df[self.train_df['Weekly_Sales'] >= 0]
        logger.info(f"Removed {initial_len - len(self.train_df)} negative sales records")
        
        # Cap markdown values at 99th percentile
        markdown_cols = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
        for col in markdown_cols:
            if col in self.features_df.columns:
                p99 = self.features_df[col].quantile(0.99)
                self.features_df[col] = self.features_df[col].clip(upper=p99)
        
        return self
    
    def encode_categorical(self):
        """Encode categorical variables"""
        logger.info("Encoding categorical variables...")
        
        # Store Type encoding
        type_mapping = {'A': 0, 'B': 1, 'C': 2}
        self.stores_df['Type_Encoded'] = self.stores_df['Type'].map(type_mapping)
        
        # IsHoliday to integer
        self.train_df['IsHoliday'] = self.train_df['IsHoliday'].astype(int)
        self.features_df['IsHoliday'] = self.features_df['IsHoliday'].astype(int)
        
        return self
    
    def merge_datasets(self):
        """Merge all datasets"""
        logger.info("Merging datasets...")
        
        # Merge train with stores
        merged = self.train_df.merge(self.stores_df, on='Store', how='left')
        
        # Merge with features
        merged = merged.merge(
            self.features_df, 
            on=['Store', 'Date', 'IsHoliday'], 
            how='left'
        )
        
        logger.info(f"Merged shape: {merged.shape}")
        
        return merged
    
    def add_derived_features(self, df):
        """Add simple derived features"""
        logger.info("Adding derived features...")
        
        # Total MarkDown
        markdown_cols = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
        if all(col in df.columns for col in markdown_cols):
            df['Total_MarkDown'] = df[markdown_cols].sum(axis=1)
            df['Has_MarkDown'] = (df['Total_MarkDown'] > 0).astype(int)
        
        return df
    
    def process_all(self, save_path='data/processed'):
        """Run complete processing pipeline"""
        logger.info("="*60)
        logger.info("STARTING DATA PROCESSING PIPELINE")
        logger.info("="*60)
        
        # Load and process
        self.load_data()
        self.convert_dates()
        self.handle_missing_values()
        self.remove_outliers()
        self.encode_categorical()
        
        # Merge and add features
        merged_df = self.merge_datasets()
        merged_df = self.add_derived_features(merged_df)
        
        # Save
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        output_file = save_path / 'processed_train.csv'
        merged_df.to_csv(output_file, index=False)
        logger.info(f"Saved to: {output_file}")
        
        logger.info("="*60)
        logger.info(f"Total records: {len(merged_df):,}")
        logger.info(f"Stores: {merged_df['Store'].nunique()}")
        logger.info(f"Departments: {merged_df['Dept'].nunique()}")
        logger.info(f"Avg weekly sales: ${merged_df['Weekly_Sales'].mean():,.2f}")
        logger.info("="*60)
        
        return merged_df


if __name__ == "__main__":
    processor = WalmartDataProcessor(data_path='data/raw')
    df = processor.process_all(save_path='data/processed')
    print("\nFirst 5 rows:")
    print(df.head())
