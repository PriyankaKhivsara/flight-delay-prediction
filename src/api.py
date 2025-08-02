"""
FastAPI application for flight delay prediction system
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional
import time
from datetime import datetime
import logging

from predict import FlightDelayPredictor, PredictionValidator
from config import API_CONFIG, TARGET_METRICS


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Flight Delay Prediction API",
    description="API for predicting weather-related flight delays",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor instance
predictor = None


# Pydantic models for request/response validation
class FlightPredictionRequest(BaseModel):
    """Request model for flight delay prediction"""
    origin: str = Field(..., description="Origin airport code (3 letters)", min_length=3, max_length=3)
    destination: str = Field(..., description="Destination airport code (3 letters)", min_length=3, max_length=3)
    date: str = Field(..., description="Flight date in YYYY-MM-DD format")
    crs_dep_time: int = Field(..., description="Scheduled departure time (0-2359)", ge=0, le=2359)
    weather_type: Optional[str] = Field("Clear", description="Weather condition type")
    precipitation_in: Optional[float] = Field(0.0, description="Precipitation in inches", ge=0)
    distance: Optional[int] = Field(1000, description="Flight distance in miles", ge=0)
    
    @validator('origin', 'destination')
    def validate_airport_codes(cls, v):
        if not v.isupper():
            raise ValueError('Airport codes must be uppercase')
        return v
    
    @validator('date')
    def validate_date_format(cls, v):
        try:
            datetime.strptime(v, '%Y-%m-%d')
        except ValueError:
            raise ValueError('Date must be in YYYY-MM-DD format')
        return v


class KeyFactor(BaseModel):
    """Model for key factors in prediction"""
    feature: str = Field(..., description="Feature name")
    impact: float = Field(..., description="Impact score")


class FlightPredictionResponse(BaseModel):
    """Response model for flight delay prediction"""
    prediction: str = Field(..., description="Delay severity (Low/Medium/High)")
    probability: float = Field(..., description="Delay probability (0-1)")
    key_factors: List[KeyFactor] = Field(..., description="Key factors influencing prediction")
    inference_time_ms: Optional[float] = Field(None, description="Inference time in milliseconds")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions"""
    flights: List[FlightPredictionRequest] = Field(..., description="List of flights to predict")


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions"""
    predictions: List[FlightPredictionResponse] = Field(..., description="List of predictions")
    total_flights: int = Field(..., description="Total number of flights processed")
    successful_predictions: int = Field(..., description="Number of successful predictions")
    processing_time_ms: float = Field(..., description="Total processing time")


class HealthResponse(BaseModel):
    """Health check response model"""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    version: str = Field("1.0.0", description="API version")


class ModelInfoResponse(BaseModel):
    """Model information response"""
    model_type: str = Field(..., description="Type of model used")
    model_class: str = Field(..., description="Model class name")
    target_metrics: Dict = Field(..., description="Target performance metrics")
    is_loaded: bool = Field(..., description="Whether model is loaded")


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the predictor on startup"""
    global predictor
    try:
        predictor = FlightDelayPredictor(model_name='xgboost')
        # Don't initialize immediately to allow for faster startup
        logger.info("Flight Delay Prediction API started successfully")
    except Exception as e:
        logger.error(f"Failed to initialize predictor: {e}")


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    model_loaded = predictor is not None and predictor.is_loaded
    
    return HealthResponse(
        status="healthy" if model_loaded else "starting",
        model_loaded=model_loaded
    )


# Model information endpoint
@app.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get information about the loaded model"""
    global predictor
    
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor not initialized")
    
    try:
        if not predictor.is_loaded:
            predictor.initialize()
        
        model_info = predictor.get_model_info()
        
        return ModelInfoResponse(
            model_type=model_info.get("model_type", "unknown"),
            model_class=model_info.get("model_class", "unknown"),
            target_metrics=TARGET_METRICS,
            is_loaded=model_info.get("is_loaded", False)
        )
    
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail="Failed to get model information")


# Main prediction endpoint
@app.post("/predict", response_model=FlightPredictionResponse)
async def predict_flight_delay(request: FlightPredictionRequest):
    """
    Predict weather-related flight delay probability
    
    This endpoint predicts the likelihood of weather-related delays for a flight
    based on route, timing, and weather conditions.
    """
    global predictor
    
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor not initialized")
    
    try:
        # Initialize predictor if not already loaded
        if not predictor.is_loaded:
            predictor.initialize()
        
        # Convert request to dictionary
        flight_data = request.dict()
        
        # Validate input
        PredictionValidator.validate_flight_input(flight_data)
        
        # Make prediction
        start_time = time.time()
        result = predictor.predict_single_flight(flight_data)
        total_time = (time.time() - start_time) * 1000
        
        # Convert key factors to KeyFactor objects
        key_factors = [
            KeyFactor(feature=factor["feature"], impact=factor["impact"])
            for factor in result["key_factors"]
        ]
        
        response = FlightPredictionResponse(
            prediction=result["prediction"],
            probability=result["probability"],
            key_factors=key_factors,
            inference_time_ms=total_time
        )
        
        logger.info(f"Prediction made: {request.origin}->{request.destination}, "
                   f"Result: {result['prediction']} ({result['probability']:.3f})")
        
        return response
    
    except ValueError as ve:
        logger.warning(f"Invalid input: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    
    except FileNotFoundError as fe:
        logger.error(f"Model not found: {fe}")
        raise HTTPException(status_code=503, detail="Model not available")
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Batch prediction endpoint
@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch_flights(request: BatchPredictionRequest):
    """
    Predict weather-related flight delays for multiple flights
    
    This endpoint processes multiple flight predictions in a single request.
    """
    global predictor
    
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor not initialized")
    
    if len(request.flights) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 flights per batch")
    
    try:
        # Initialize predictor if not already loaded
        if not predictor.is_loaded:
            predictor.initialize()
        
        start_time = time.time()
        predictions = []
        successful_count = 0
        
        for flight_request in request.flights:
            try:
                flight_data = flight_request.dict()
                PredictionValidator.validate_flight_input(flight_data)
                
                result = predictor.predict_single_flight(flight_data)
                
                key_factors = [
                    KeyFactor(feature=factor["feature"], impact=factor["impact"])
                    for factor in result["key_factors"]
                ]
                
                prediction = FlightPredictionResponse(
                    prediction=result["prediction"],
                    probability=result["probability"],
                    key_factors=key_factors,
                    inference_time_ms=result.get("inference_time_ms", 0)
                )
                
                predictions.append(prediction)
                successful_count += 1
                
            except Exception as e:
                # Create error response for failed prediction
                error_prediction = FlightPredictionResponse(
                    prediction="Error",
                    probability=0.0,
                    key_factors=[],
                    inference_time_ms=0
                )
                predictions.append(error_prediction)
                logger.warning(f"Failed to predict flight: {e}")
        
        total_time = (time.time() - start_time) * 1000
        
        response = BatchPredictionResponse(
            predictions=predictions,
            total_flights=len(request.flights),
            successful_predictions=successful_count,
            processing_time_ms=total_time
        )
        
        logger.info(f"Batch prediction completed: {successful_count}/{len(request.flights)} successful")
        
        return response
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail="Batch prediction failed")


# Sample prediction endpoint
@app.get("/predict/sample")
async def get_sample_prediction():
    """Get a sample prediction for demonstration purposes"""
    sample_request = FlightPredictionRequest(
        origin="JFK",
        destination="LAX",
        date="2023-12-15",
        crs_dep_time=800,
        weather_type="Snow",
        precipitation_in=0.4
    )
    
    return await predict_flight_delay(sample_request)


# Performance metrics endpoint
@app.get("/metrics")
async def get_performance_metrics():
    """Get system performance metrics"""
    return {
        "target_metrics": TARGET_METRICS,
        "api_config": API_CONFIG,
        "timestamp": datetime.now().isoformat()
    }


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Flight Delay Prediction API",
        "version": "1.0.0",
        "description": "API for predicting weather-related flight delays",
        "endpoints": {
            "predict": "/predict",
            "batch_predict": "/predict/batch",
            "health": "/health",
            "model_info": "/model/info",
            "sample": "/predict/sample",
            "docs": "/docs"
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api:app",
        host=API_CONFIG["host"],
        port=API_CONFIG["port"],
        reload=API_CONFIG["reload"]
    )