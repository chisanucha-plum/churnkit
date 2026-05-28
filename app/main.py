"""
FastAPI main application for Customer Churn Prediction System.

Simple MVP with basic endpoints.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from app.config.settings import get_settings
from app.database.connection import get_db_manager, close_db
from app.services.sample_data import generate_sample_data, save_sample_data, load_sample_data
from app.services.simple_model import SimpleModelTrainer
import numpy as np
import pandas as pd


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instance
model_trainer = None


# ===== Lifespan Events =====
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application startup and shutdown.
    """
    # Startup
    logger.info("🚀 Starting application...")
    
    global model_trainer
    
    # Initialize database
    try:
        db_manager = get_db_manager()
        if db_manager.health_check():
            logger.info("✓ Database connection healthy")
        else:
            logger.warning("⚠️  Database connection unhealthy")
    except Exception as e:
        logger.error(f"✗ Database initialization failed: {str(e)}")
    
    # Load or train model
    try:
        model_trainer = SimpleModelTrainer()
        
        # Try to load existing model
        try:
            model_trainer.load('simple_model.pkl')
            logger.info("✓ Model loaded from disk")
        except FileNotFoundError:
            logger.info("⚠️  Model not found. Generating sample data and training...")
            
            # Generate sample data
            df = generate_sample_data(n_samples=1000)
            save_sample_data(df)
            
            # Train model
            model_trainer.train(df)
            model_trainer.save('simple_model.pkl')
            logger.info("✓ Model trained and saved")
    except Exception as e:
        logger.error(f"✗ Model initialization failed: {str(e)}")
    
    logger.info("✅ Application started successfully")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down application...")
    close_db()
    logger.info("✅ Application shutdown complete")


# ===== Create FastAPI App =====
settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Customer Churn Prediction System - MVP",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===== Pydantic Models =====
class CustomerInput(BaseModel):
    """Customer input for prediction."""
    tenure: int
    monthly_charges: float
    total_charges: float
    contract_type: str
    internet_service: str
    online_security: int
    online_backup: int
    device_protection: int
    tech_support: int
    streaming_tv: int
    streaming_movies: int
    payment_method: str
    paperless_billing: int
    senior_citizen: int
    partner: int
    dependents: int
    phone_service: int
    multiple_lines: int


class PredictionResponse(BaseModel):
    """Prediction response."""
    churn_probability: float
    risk_level: str
    risk_score: int
    recommendation: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    database_healthy: bool
    version: str


class MetricsResponse(BaseModel):
    """Model metrics response."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float


# ===== Helper Functions =====
def get_risk_level(probability: float) -> str:
    """Determine risk level from probability."""
    if probability < 0.33:
        return "LOW"
    elif probability < 0.67:
        return "MEDIUM"
    else:
        return "HIGH"


def get_recommendation(probability: float, contract_type: str) -> str:
    """Get retention recommendation."""
    if probability > 0.7:
        if contract_type == "Month-to-month":
            return "Offer contract upgrade incentive"
        else:
            return "Offer premium support package"
    elif probability > 0.4:
        return "Offer loyalty discount"
    else:
        return "Monitor customer satisfaction"


def preprocess_customer_data(customer: CustomerInput) -> np.ndarray:
    """Preprocess customer data for prediction."""
    payload = {
        "tenure": customer.tenure,
        "monthly_charges": customer.monthly_charges,
        "total_charges": customer.total_charges,
        "contract_type": customer.contract_type,
        "internet_service": customer.internet_service,
        "online_security": customer.online_security,
        "online_backup": customer.online_backup,
        "device_protection": customer.device_protection,
        "tech_support": customer.tech_support,
        "streaming_tv": customer.streaming_tv,
        "streaming_movies": customer.streaming_movies,
        "payment_method": customer.payment_method,
        "paperless_billing": customer.paperless_billing,
        "senior_citizen": customer.senior_citizen,
        "partner": customer.partner,
        "dependents": customer.dependents,
        "phone_service": customer.phone_service,
        "multiple_lines": customer.multiple_lines,
    }

    df = pd.DataFrame([payload])
    return model_trainer.transform_features(df)


# ===== API Endpoints =====

@app.get("/", tags=["Info"])
async def root():
    """Root endpoint."""
    return {
        "message": "Customer Churn Prediction System - MVP",
        "version": settings.app_version,
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    db_manager = get_db_manager()
    
    return HealthResponse(
        status="healthy",
        model_loaded=model_trainer is not None and model_trainer.model is not None,
        database_healthy=db_manager.health_check(),
        version=settings.app_version
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(customer: CustomerInput):
    """
    Predict churn probability for a customer.
    
    Example:
        {
            "tenure": 24,
            "monthly_charges": 89.5,
            "total_charges": 2148.0,
            "contract_type": "Month-to-month",
            "internet_service": "Fiber optic",
            "online_security": 1,
            "online_backup": 0,
            "device_protection": 1,
            "tech_support": 0,
            "streaming_tv": 1,
            "streaming_movies": 1,
            "payment_method": "Electronic check",
            "paperless_billing": 1,
            "senior_citizen": 0,
            "partner": 1,
            "dependents": 0,
            "phone_service": 1,
            "multiple_lines": 0
        }
    """
    if model_trainer is None or model_trainer.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Preprocess data
        X = preprocess_customer_data(customer)
        
        # Make prediction
        _, probability = model_trainer.predict(X)
        churn_probability = float(probability[0])
        
        # Determine risk level and score
        risk_level = get_risk_level(churn_probability)
        risk_score = int(churn_probability * 100)
        
        # Get recommendation
        recommendation = get_recommendation(churn_probability, customer.contract_type)
        
        return PredictionResponse(
            churn_probability=round(churn_probability, 3),
            risk_level=risk_level,
            risk_score=risk_score,
            recommendation=recommendation
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics", response_model=MetricsResponse, tags=["Metrics"])
async def get_metrics():
    """Get model metrics."""
    if model_trainer is None or not model_trainer.metrics:
        raise HTTPException(status_code=503, detail="Model metrics not available")
    
    return MetricsResponse(**model_trainer.metrics)


@app.get("/features", tags=["Model"])
async def get_features():
    """Get model features."""
    if model_trainer is None or model_trainer.feature_names is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "features": model_trainer.feature_names,
        "count": len(model_trainer.feature_names)
    }


@app.get("/feature-importance", tags=["Model"])
async def get_feature_importance():
    """Get feature importance."""
    if model_trainer is None or model_trainer.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    importance = model_trainer.get_feature_importance()
    
    return {
        "importance": importance,
        "top_10": dict(list(importance.items())[:10])
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers
    )
