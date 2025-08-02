"""
Prediction module for flight delay prediction system
"""
import pandas as pd
import numpy as np
import pickle
import time
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from data_preprocessing import FlightDataPreprocessor
from config import MODELS_DIR, DELAY_SEVERITY_THRESHOLDS


class FlightDelayPredictor:
    """Handles model loading and prediction for flight delays"""
    
    def __init__(self, model_name='xgboost'):
        self.model_name = model_name
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        self.is_loaded = False
        
    def load_model(self, model_path: Optional[Path] = None):
        """Load trained model from disk"""
        if model_path is None:
            model_path = MODELS_DIR / f"{self.model_name}_model.pkl"
        
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"Model loaded from {model_path}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found: {model_path}")
    
    def load_preprocessor(self, preprocessor_path: Optional[Path] = None):
        """Load fitted preprocessor from disk"""
        if preprocessor_path is None:
            preprocessor_path = MODELS_DIR / "preprocessing.pkl"
        
        try:
            self.preprocessor = FlightDataPreprocessor()
            self.preprocessor.load_preprocessor(preprocessor_path)
            print(f"Preprocessor loaded from {preprocessor_path}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Preprocessor file not found: {preprocessor_path}")
    
    def initialize(self):
        """Initialize predictor by loading model and preprocessor"""
        self.load_model()
        self.load_preprocessor()
        self.is_loaded = True
        print(f"Predictor initialized with {self.model_name} model")
    
    def predict_single_flight(self, flight_data: Dict) -> Dict:
        """
        Predict delay probability for a single flight
        
        Args:
            flight_data: Dictionary containing flight information
                {
                    "origin": "JFK",
                    "destination": "LAX", 
                    "date": "2023-12-15",
                    "crs_dep_time": 800,
                    "weather_type": "Snow",
                    "precipitation_in": 0.4
                }
        
        Returns:
            Dictionary containing prediction results
        """
        if not self.is_loaded:
            self.initialize()
        
        # Convert input to DataFrame format
        flight_df, weather_df = self._prepare_input_data(flight_data)
        
        # Preprocess data
        X = self.preprocessor.transform(flight_df, weather_df)
        
        # Make prediction
        start_time = time.time()
        probability = self.model.predict_proba(X)[0, 1]
        inference_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        # Classify severity
        severity = self._classify_severity(probability)
        
        # Get feature importance for this prediction
        key_factors = self._get_prediction_factors(X, flight_data)
        
        result = {
            "prediction": severity,
            "probability": round(probability, 3),
            "inference_time_ms": round(inference_time, 2),
            "key_factors": key_factors
        }
        
        return result
    
    def predict_batch(self, flight_list: List[Dict]) -> List[Dict]:
        """Predict delay probabilities for multiple flights"""
        if not self.is_loaded:
            self.initialize()
        
        results = []
        
        for flight_data in flight_list:
            try:
                result = self.predict_single_flight(flight_data)
                results.append(result)
            except Exception as e:
                # Handle individual prediction errors
                results.append({
                    "prediction": "Error",
                    "probability": 0.0,
                    "error": str(e)
                })
        
        return results
    
    def _prepare_input_data(self, flight_data: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Convert input dictionary to DataFrame format for preprocessing"""
        # Create flight DataFrame
        flight_df = pd.DataFrame({
            'FL_DATE': [flight_data['date']],
            'AIRLINE': ['AA'],  # Default airline
            'ORIGIN': [flight_data['origin']],
            'DEST': [flight_data['destination']],
            'CRS_DEP_TIME': [flight_data['crs_dep_time']],
            'CRS_ARR_TIME': [flight_data.get('crs_arr_time', flight_data['crs_dep_time'] + 180)],
            'CRS_ELAPSED_TIME': [flight_data.get('crs_elapsed_time', 180)],
            'DISTANCE': [flight_data.get('distance', 1000)],  # Default distance
            'DEP_DELAY': [0],
            'ARR_DELAY': [0],
            'DELAY_DUE_WEATHER': [0]
        })
        
        # Create weather DataFrame
        weather_df = pd.DataFrame({
            'EventDate': [flight_data['date']],
            'City': [flight_data['origin']],
            'Type': [flight_data.get('weather_type', 'Clear')],
            'Severity': [flight_data.get('weather_severity', 'Low')],
            'Precipitation(in)': [flight_data.get('precipitation_in', 0.0)]
        })
        
        return flight_df, weather_df
    
    def _classify_severity(self, probability: float) -> str:
        """Classify delay probability into severity categories"""
        if probability <= DELAY_SEVERITY_THRESHOLDS['Low']:
            return "Low"
        elif probability <= DELAY_SEVERITY_THRESHOLDS['Medium']:
            return "Medium"
        else:
            return "High"
    
    def _get_prediction_factors(self, X: pd.DataFrame, flight_data: Dict) -> List[Dict]:
        """Identify key factors contributing to the prediction"""
        # For tree-based models, get feature importance
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = self.model.feature_importances_
            feature_names = X.columns if hasattr(X, 'columns') else [f'feature_{i}' for i in range(X.shape[1])]
            
            # Create importance ranking
            importance_data = list(zip(feature_names, feature_importance))
            importance_data.sort(key=lambda x: x[1], reverse=True)
            
            # Return top 5 factors with human-readable names
            factors = []
            for i, (feature, importance) in enumerate(importance_data[:5]):
                factor_name = self._get_readable_factor_name(feature, flight_data)
                factors.append({
                    "feature": factor_name,
                    "impact": round(importance, 3)
                })
            
            return factors
        
        # Default factors if feature importance not available
        return [
            {"feature": "weather_conditions", "impact": 0.4},
            {"feature": "time_of_day", "impact": 0.3},
            {"feature": "route", "impact": 0.2},
            {"feature": "season", "impact": 0.1}
        ]
    
    def _get_readable_factor_name(self, feature: str, flight_data: Dict) -> str:
        """Convert technical feature names to human-readable names"""
        feature_mapping = {
            'PRECIP_LEVEL': 'precipitation',
            'SEVERITY_SCORE': 'weather_severity',
            'SEASON': 'season',
            'TIME_CATEGORY': 'time_of_day',
            'DISTANCE_GROUP': 'flight_distance',
            'ORIGIN_DEST_PAIR': 'route',
            'MONTH': 'month',
            'DAY_OF_WEEK': 'day_of_week'
        }
        
        return feature_mapping.get(feature, feature.lower())
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        if not self.is_loaded:
            return {"error": "Model not loaded"}
        
        info = {
            "model_type": self.model_name,
            "model_class": type(self.model).__name__,
            "is_loaded": self.is_loaded
        }
        
        # Add model-specific parameters
        if hasattr(self.model, 'get_params'):
            info["parameters"] = self.model.get_params()
        
        return info


class PredictionValidator:
    """Validates prediction inputs and outputs"""
    
    @staticmethod
    def validate_flight_input(flight_data: Dict) -> bool:
        """Validate flight input data"""
        required_fields = ['origin', 'destination', 'date', 'crs_dep_time']
        
        for field in required_fields:
            if field not in flight_data:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate airport codes (3 letters)
        if len(flight_data['origin']) != 3 or len(flight_data['destination']) != 3:
            raise ValueError("Airport codes must be 3 letters")
        
        # Validate time format (0-2359)
        if not (0 <= flight_data['crs_dep_time'] <= 2359):
            raise ValueError("Departure time must be between 0 and 2359")
        
        # Validate precipitation if provided
        if 'precipitation_in' in flight_data:
            if flight_data['precipitation_in'] < 0:
                raise ValueError("Precipitation cannot be negative")
        
        return True


def create_sample_prediction():
    """Create sample prediction for testing"""
    # Initialize predictor
    predictor = FlightDelayPredictor(model_name='xgboost')
    
    # Sample flight data
    sample_flight = {
        "origin": "JFK",
        "destination": "LAX",
        "date": "2023-12-15",
        "crs_dep_time": 800,
        "weather_type": "Snow",
        "precipitation_in": 0.4
    }
    
    try:
        # Validate input
        PredictionValidator.validate_flight_input(sample_flight)
        
        # Make prediction
        result = predictor.predict_single_flight(sample_flight)
        
        print("Sample Prediction Result:")
        print(f"Flight: {sample_flight['origin']} → {sample_flight['destination']}")
        print(f"Prediction: {result['prediction']}")
        print(f"Probability: {result['probability']}")
        print(f"Key Factors: {result['key_factors']}")
        
        return result
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return None


if __name__ == "__main__":
    create_sample_prediction()