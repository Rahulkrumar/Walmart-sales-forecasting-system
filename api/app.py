"""
Flask REST API for Walmart Sales Forecasting
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from pydantic import BaseModel, validator, ValidationError
from datetime import datetime
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

MODEL_PATH = Path('models')


# Input validation
class PredictionInput(BaseModel):
    Store: int
    Dept: int
    Date: str
    IsHoliday: int
    Type: str
    Size: float
    Temperature: float
    Fuel_Price: float
    MarkDown1: float = 0.0
    MarkDown2: float = 0.0
    MarkDown3: float = 0.0
    MarkDown4: float = 0.0
    MarkDown5: float = 0.0
    CPI: float
    Unemployment: float
    
    @validator('Store')
    def validate_store(cls, v):
        if v < 1 or v > 45:
            raise ValueError('Store must be between 1 and 45')
        return v
    
    @validator('Date')
    def validate_date(cls, v):
        try:
            datetime.strptime(v, '%Y-%m-%d')
        except ValueError:
            raise ValueError('Date must be YYYY-MM-DD format')
        return v
    
    @validator('Type')
    def validate_type(cls, v):
        if v not in ['A', 'B', 'C']:
            raise ValueError('Type must be A, B, or C')
        return v


# Model handler
class ModelHandler:
    def __init__(self):
        self.model = None
        self.feature_cols = None
        self.load_model()
    
    def load_model(self):
        try:
            model_path = MODEL_PATH / 'xgboost_model.pkl'
            self.model = joblib.load(model_path)
            
            feature_path = MODEL_PATH / 'feature_columns.pkl'
            if feature_path.exists():
                self.feature_cols = joblib.load(feature_path)
            
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def preprocess_input(self, input_data):
        df = pd.DataFrame([input_data])
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Basic time features
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Week'] = df['Date'].dt.isocalendar().week
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['Quarter'] = df['Date'].dt.quarter
        
        # Type encoding
        type_mapping = {'A': 0, 'B': 1, 'C': 2}
        df['Type_Encoded'] = df['Type'].map(type_mapping)
        
        # Total markdown
        markdown_cols = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
        df['Total_MarkDown'] = df[markdown_cols].sum(axis=1)
        
        # Add missing features with 0
        if self.feature_cols:
            for col in self.feature_cols:
                if col not in df.columns:
                    df[col] = 0
            df = df[self.feature_cols]
        
        return df
    
    def predict(self, input_data):
        df = self.preprocess_input(input_data)
        prediction = self.model.predict(df)[0]
        return float(prediction)


model_handler = ModelHandler()


# API Endpoints
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'service': 'Walmart Sales Forecasting API',
        'status': 'running',
        'endpoints': {
            'health': '/health',
            'predict': '/v1/predict'
        }
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_handler.model is not None
    })


@app.route('/v1/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No input data'}), 400
        
        # Validate input
        try:
            validated_input = PredictionInput(**data)
        except ValidationError as e:
            return jsonify({'error': 'Invalid input', 'details': e.errors()}), 400
        
        # Make prediction
        prediction = model_handler.predict(validated_input.dict())
        
        response = {
            'prediction': prediction,
            'model': 'XGBoost',
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': 'Internal server error'}), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    logger.info("Starting Walmart Sales Forecasting API...")
    app.run(host='0.0.0.0', port=5000, debug=False)
