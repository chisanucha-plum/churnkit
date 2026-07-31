"""Prediction routes - Route layer.

Layered Architecture: Route Layer
- Receives HTTP requests
- Validates input via Pydantic schemas
- Delegates business logic to service
- Returns HTTP responses
"""

import logging

from fastapi import APIRouter, Depends, HTTPException

from app.dependencies import get_prediction_service
from app.models.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    CustomerInput,
    PredictionResponse,
)
from app.services.prediction_service import PredictionService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predict", tags=["Prediction"])


@router.post("", response_model=PredictionResponse)
async def predict(
    customer: CustomerInput,
    service: PredictionService = Depends(get_prediction_service),
):
    """
    Predict churn probability for a single customer.

    Returns the churn probability, risk level, and retention recommendation.
    """
    if not service.is_model_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        return service.predict(customer)
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: BatchPredictionRequest,
    service: PredictionService = Depends(get_prediction_service),
):
    """
    Predict churn probability for multiple customers.

    Processes a batch of customers and returns predictions for each.
    """
    if not service.is_model_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        return service.predict_batch(request)
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
