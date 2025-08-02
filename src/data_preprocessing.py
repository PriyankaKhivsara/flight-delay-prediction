"""
Data preprocessing and feature engineering pipeline for flight delay prediction
"""
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, TargetEncoder
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
import pickle
from pathlib import Path

from config import (
    SEASON_MAP, TIME_CATEGORIES, FLIGHT_COLUMNS, WEATHER_COLUMNS,
    PROCESSED_DATA_DIR, MODELS_DIR
)


class FlightDataPreprocessor:
    """Handles preprocessing and feature engineering for flight delay data"""
    
    def __init__(self):
        self.target_encoder = TargetEncoder()
        self.scaler = MinMaxScaler()
        self.feature_selector = None
        self.selected_features = None
        
    def engineer_temporal_features(self, df):
        """Create temporal features from flight date and time"""
        df = df.copy()
        
        # Convert FL_DATE to datetime
        df['FL_DATE'] = pd.to_datetime(df['FL_DATE'])
        
        # Extract temporal features
        df['MONTH'] = df['FL_DATE'].dt.month
        df['DAY_OF_WEEK'] = df['FL_DATE'].dt.dayofweek
        
        # Map season
        df['SEASON'] = df['MONTH'].map(SEASON_MAP)
        
        # Categorize time of day
        df['TIME_CATEGORY'] = df['CRS_DEP_TIME'].apply(self.get_time_category)
        
        return df
    
    def get_time_category(self, time_val):
        """Convert time to category (Night/Morning/Afternoon/Evening)"""
        for category, (start, end) in TIME_CATEGORIES.items():
            if start <= time_val <= end:
                return category
        return 'Night'  # Default for edge cases
    
    def engineer_route_features(self, df):
        """Create route-based features"""
        df = df.copy()
        
        # Create origin-destination pair
        df['ORIGIN_DEST_PAIR'] = df['ORIGIN'] + '_' + df['DEST']
        
        # Categorize distance
        df['DISTANCE_GROUP'] = pd.cut(
            df['DISTANCE'], 
            bins=[0, 500, 1000, 2000, 5000], 
            labels=['Short', 'Medium', 'Long', 'Ultra_Long']
        )
        
        return df
    
    def engineer_weather_features(self, weather_df):
        """Process weather data and create severity features"""
        weather_df = weather_df.copy()
        
        # Create precipitation level categories
        weather_df['PRECIP_LEVEL'] = pd.cut(
            weather_df['Precipitation(in)'],
            bins=[-0.1, 0, 0.1, 0.5, 1.0, float('inf')],
            labels=['None', 'Light', 'Moderate', 'Heavy', 'Extreme']
        )
        
        # Create severity score based on weather type and precipitation
        severity_map = {
            'Clear': 1, 'Partly Cloudy': 1, 'Cloudy': 2,
            'Rain': 3, 'Snow': 4, 'Thunderstorm': 5, 'Fog': 3
        }
        weather_df['SEVERITY_SCORE'] = weather_df['Type'].map(severity_map).fillna(2)
        
        # Adjust severity based on precipitation
        precip_multiplier = weather_df['Precipitation(in)'].apply(
            lambda x: 1 if x == 0 else 1.5 if x < 0.5 else 2.0 if x < 1.0 else 2.5
        )
        weather_df['SEVERITY_SCORE'] *= precip_multiplier
        
        return weather_df
    
    def merge_flight_weather_data(self, flight_df, weather_df):
        """Merge flight and weather data based on date and airport"""
        # Convert dates to same format
        flight_df['FL_DATE'] = pd.to_datetime(flight_df['FL_DATE'])
        weather_df['EventDate'] = pd.to_datetime(weather_df['EventDate'])
        
        # Merge on origin airport and date
        merged_df = flight_df.merge(
            weather_df.rename(columns={'City': 'ORIGIN', 'EventDate': 'FL_DATE'}),
            on=['ORIGIN', 'FL_DATE'],
            how='left',
            suffixes=('', '_weather')
        )
        
        return merged_df
    
    def create_target_variable(self, df, delay_threshold=15):
        """Create binary target variable for weather-related delays"""
        # Consider a flight delayed if weather delay > threshold minutes
        df['IS_DELAYED'] = (
            (df['DELAY_DUE_WEATHER'] > delay_threshold) & 
            (df['DELAY_DUE_WEATHER'].notna())
        ).astype(int)
        
        return df
    
    def encode_categorical_features(self, df, target_col='IS_DELAYED', fit=True):
        """Apply target encoding to categorical features"""
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target and irrelevant columns
        categorical_cols = [col for col in categorical_cols if col not in [
            target_col, 'FL_DATE', 'AIRLINE'
        ]]
        
        if fit:
            df[categorical_cols] = self.target_encoder.fit_transform(
                df[categorical_cols], df[target_col]
            )
        else:
            df[categorical_cols] = self.target_encoder.transform(df[categorical_cols])
        
        return df
    
    def scale_numerical_features(self, df, fit=True):
        """Apply MinMax scaling to numerical features"""
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target column
        numerical_cols = [col for col in numerical_cols if col != 'IS_DELAYED']
        
        if fit:
            df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
        else:
            df[numerical_cols] = self.scaler.transform(df[numerical_cols])
        
        return df
    
    def select_features(self, X, y, n_features=12):
        """Select top features using Recursive Feature Elimination"""
        self.feature_selector = RFE(
            estimator=RandomForestClassifier(n_estimators=50, random_state=42),
            n_features_to_select=n_features
        )
        
        X_selected = self.feature_selector.fit_transform(X, y)
        self.selected_features = X.columns[self.feature_selector.support_].tolist()
        
        return X_selected, self.selected_features
    
    def fit_transform(self, flight_df, weather_df):
        """Complete preprocessing pipeline for training data"""
        print("Starting preprocessing pipeline...")
        
        # Feature engineering
        print("Engineering temporal features...")
        flight_df = self.engineer_temporal_features(flight_df)
        
        print("Engineering route features...")
        flight_df = self.engineer_route_features(flight_df)
        
        print("Processing weather features...")
        weather_df = self.engineer_weather_features(weather_df)
        
        print("Merging flight and weather data...")
        merged_df = self.merge_flight_weather_data(flight_df, weather_df)
        
        print("Creating target variable...")
        merged_df = self.create_target_variable(merged_df)
        
        # Remove rows with missing target
        merged_df = merged_df.dropna(subset=['IS_DELAYED'])
        
        print("Encoding categorical features...")
        merged_df = self.encode_categorical_features(merged_df, fit=True)
        
        print("Scaling numerical features...")
        merged_df = self.scale_numerical_features(merged_df, fit=True)
        
        # Prepare features and target
        feature_cols = [col for col in merged_df.columns if col not in [
            'IS_DELAYED', 'FL_DATE', 'AIRLINE', 'DEP_DELAY', 'ARR_DELAY'
        ]]
        
        X = merged_df[feature_cols]
        y = merged_df['IS_DELAYED']
        
        print("Selecting top features...")
        X_selected, selected_features = self.select_features(X, y)
        
        print(f"Selected features: {selected_features}")
        
        return X_selected, y, selected_features
    
    def transform(self, flight_df, weather_df):
        """Transform new data using fitted preprocessors"""
        # Feature engineering
        flight_df = self.engineer_temporal_features(flight_df)
        flight_df = self.engineer_route_features(flight_df)
        weather_df = self.engineer_weather_features(weather_df)
        
        # Merge data
        merged_df = self.merge_flight_weather_data(flight_df, weather_df)
        
        # Encoding and scaling (using fitted transformers)
        merged_df = self.encode_categorical_features(merged_df, fit=False)
        merged_df = self.scale_numerical_features(merged_df, fit=False)
        
        # Select features
        if self.selected_features:
            X = merged_df[self.selected_features]
        else:
            feature_cols = [col for col in merged_df.columns if col not in [
                'IS_DELAYED', 'FL_DATE', 'AIRLINE', 'DEP_DELAY', 'ARR_DELAY'
            ]]
            X = merged_df[feature_cols]
        
        return X
    
    def save_preprocessor(self, filepath):
        """Save fitted preprocessor to disk"""
        preprocessor_data = {
            'target_encoder': self.target_encoder,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'selected_features': self.selected_features
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(preprocessor_data, f)
        
        print(f"Preprocessor saved to {filepath}")
    
    def load_preprocessor(self, filepath):
        """Load fitted preprocessor from disk"""
        with open(filepath, 'rb') as f:
            preprocessor_data = pickle.load(f)
        
        self.target_encoder = preprocessor_data['target_encoder']
        self.scaler = preprocessor_data['scaler']
        self.feature_selector = preprocessor_data['feature_selector']
        self.selected_features = preprocessor_data['selected_features']
        
        print(f"Preprocessor loaded from {filepath}")


def create_sample_data():
    """Create sample data files for testing"""
    # Sample flight data
    flight_data = {
        'FL_DATE': ['2023-01-15', '2023-01-16', '2023-06-20', '2023-12-10'],
        'AIRLINE': ['AA', 'DL', 'UA', 'WN'],
        'ORIGIN': ['JFK', 'LAX', 'ORD', 'ATL'],
        'DEST': ['LAX', 'JFK', 'DFW', 'SEA'],
        'CRS_DEP_TIME': [800, 1400, 1800, 600],
        'CRS_ARR_TIME': [1100, 2200, 2100, 900],
        'CRS_ELAPSED_TIME': [180, 480, 180, 180],
        'DISTANCE': [2475, 2475, 925, 1890],
        'DEP_DELAY': [0, 45, 120, 5],
        'ARR_DELAY': [10, 60, 150, 0],
        'DELAY_DUE_WEATHER': [0, 30, 90, 0]
    }
    
    # Sample weather data
    weather_data = {
        'EventDate': ['2023-01-15', '2023-01-16', '2023-06-20', '2023-12-10'],
        'City': ['JFK', 'LAX', 'ORD', 'ATL'],
        'Type': ['Clear', 'Rain', 'Thunderstorm', 'Snow'],
        'Severity': ['Low', 'Medium', 'High', 'Medium'],
        'Precipitation(in)': [0.0, 0.3, 0.8, 0.4]
    }
    
    return pd.DataFrame(flight_data), pd.DataFrame(weather_data)


if __name__ == "__main__":
    # Create sample data for testing
    flight_df, weather_df = create_sample_data()
    
    # Initialize preprocessor
    preprocessor = FlightDataPreprocessor()
    
    # Fit and transform data
    X, y, features = preprocessor.fit_transform(flight_df, weather_df)
    
    print(f"Processed data shape: {X.shape}")
    print(f"Target distribution: {np.bincount(y)}")
    
    # Save preprocessor
    MODELS_DIR.mkdir(exist_ok=True)
    preprocessor.save_preprocessor(MODELS_DIR / "preprocessing.pkl")