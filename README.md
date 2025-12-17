Walmart Weekly Sales Forecasting using Machine Learning (7.8% MAPE)
📋 Table of Contents

Overview

Dataset

Installation

Quick Start

Feature Engineering

Model Performance

API Usage

Project Structure

License

🎯 Overview

This project implements a machine learning–based time series forecasting system to predict weekly sales for Walmart stores.
It forecasts sales for 45 stores and 99 departments using historical sales data and external economic factors.

Key highlights:

Engineered 80+ time-series features

Used proper time-based train–test split to prevent data leakage

Trained XGBoost model achieving 7.8% MAPE

Exposed predictions via a Flask REST API

📊 Dataset

Source: Walmart Recruiting – Store Sales Forecasting (Kaggle)

Dataset Files
File	Description
train.csv	Historical weekly sales (2010–2012)
stores.csv	Store metadata (type, size)
features.csv	External factors (temperature, fuel price, CPI, unemployment, markdowns, holidays)
Target & Features

Target: Weekly_Sales

Features: Store, Dept, Date, IsHoliday, Type, Size, Temperature, Fuel_Price, CPI, Unemployment, MarkDown1–5

🚀 Installation
Prerequisites

Python 3.8+

Kaggle account (for dataset download)

Setup
git clone https://github.com/YOUR_USERNAME/walmart-sales-forecasting.git
cd walmart-sales-forecasting

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt

⚡ Quick Start
# Download data
kaggle competitions download -c walmart-recruiting-store-sales-forecasting
unzip walmart-recruiting-store-sales-forecasting.zip -d data/raw/

# Data processing
python src/data_processing.py

# Feature engineering
python src/feature_engineering.py

# Train model
python src/train.py

# Start API
python api/app.py

⚙️ Feature Engineering

Created 80+ features following time-series best practices:

1. Time-Based Features

Year, Month, Week, DayOfWeek, Quarter

Cyclical encoding: Month_Sin/Cos, Week_Sin/Cos

Period flags: IsWeekend, IsMonthEnd, IsQuarterEnd

2. Lag Features

Sales_Lag_1, 2, 3, 4, 5

Long-term lags: Sales_Lag_8, 12, 26, 52

3. Rolling Statistics

Rolling means and standard deviations (4–52 week windows)

Rolling min/max features

4. Aggregations

Store-level, department-level, and store–department historical averages

5. Holiday & Interaction Features

Holiday indicators and surrounding-week flags

Size and holiday interaction features

📈 Model Performance
Train–Test Split

Time-based split to avoid data leakage:

cutoff_date = "2012-08-01"
train = df[df["Date"] < cutoff_date]
test  = df[df["Date"] >= cutoff_date]

Results
Model	MAPE	R²
XGBoost	7.8%	0.94
LightGBM	8.1%	0.93

Final Model: XGBoost

Top Features

Sales_Lag_1

Sales_Rolling_Mean_4

Store_Dept_Mean

IsHoliday

Sales_Lag_52

🔌 API Usage
Start Server
python api/app.py


API runs at: http://localhost:5000

Health Check
GET /health

Prediction
POST /v1/predict
Content-Type: application/json

{
  "Store": 1,
  "Dept": 1,
  "Date": "2012-11-02",
  "IsHoliday": 0,
  "Type": "A",
  "Size": 151315,
  "Temperature": 58.5,
  "Fuel_Price": 3.69,
  "CPI": 211.096,
  "Unemployment": 8.106
}


Response

{
  "prediction": 24350.67,
  "model": "XGBoost"
}

📁 Project Structure
walmart_sales_forecasting/
├── src/
│   ├── data_processing.py
│   ├── feature_engineering.py
│   └── train.py
├── api/
│   └── app.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── features/
├── models/
│   ├── xgboost_model.pkl
│   └── feature_columns.pkl
├── requirements.txt
└── README.md

📄 License

This project is licensed under the MIT License.
