"""
Configuration settings for Flight Delay Prediction System
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# Data file paths
FLIGHT_DATA_PATH = RAW_DATA_DIR / "flight_data.csv"
WEATHER_DATA_PATH = RAW_DATA_DIR / "weather_data.csv"
FEATURES_PATH = PROCESSED_DATA_DIR / "features.parquet"

# Model settings
MODEL_CONFIG = {
    'xgboost': {
        'n_estimators': 200,
        'max_depth': 5,
        'scale_pos_weight': 2.3,
        'random_state': 42
    },
    'lightgbm': {
        'n_estimators': 200,
        'max_depth': 5,
        'class_weight': 'balanced',
        'random_state': 42
    },
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 10,
        'class_weight': 'balanced',
        'random_state': 42
    }
}

# Feature engineering settings
SEASON_MAP = {
    12: 'Winter', 1: 'Winter', 2: 'Winter',
    3: 'Spring', 4: 'Spring', 5: 'Spring',
    6: 'Summer', 7: 'Summer', 8: 'Summer',
    9: 'Fall', 10: 'Fall', 11: 'Fall'
}

TIME_CATEGORIES = {
    'Night': (0, 559),
    'Morning': (600, 1159),
    'Afternoon': (1200, 1759),
    'Evening': (1800, 2359)
}

# Target thresholds
DELAY_SEVERITY_THRESHOLDS = {
    'Low': 0.3,
    'Medium': 0.7,
    'High': 1.0
}

# Performance targets
TARGET_METRICS = {
    'f1_score': 0.78,
    'accuracy': 0.783,
    'inference_time_ms': 300,
    'throughput_rps': 50
}

# API settings
API_CONFIG = {
    'host': '0.0.0.0',
    'port': 8000,
    'reload': False
}

# Column definitions
FLIGHT_COLUMNS = [
    'FL_DATE', 'AIRLINE', 'ORIGIN', 'DEST', 'CRS_DEP_TIME', 
    'CRS_ARR_TIME', 'CRS_ELAPSED_TIME', 'DISTANCE', 'DEP_DELAY', 
    'ARR_DELAY', 'DELAY_DUE_WEATHER'
]

WEATHER_COLUMNS = [
    'EventDate', 'City', 'Type', 'Severity', 'Precipitation(in)'
]

FEATURE_COLUMNS = [
    'MONTH', 'DAY_OF_WEEK', 'SEASON', 'TIME_CATEGORY',
    'ORIGIN_DEST_PAIR', 'DISTANCE_GROUP', 'PRECIP_LEVEL', 'SEVERITY_SCORE'
]