"""Prediction routes."""

from fastapi import APIRouter, HTTPException

from app.controllers.prediction_controller import PredictionController
from app.models.schemas import (
    CustomerData,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
)

router = APIRouter()
prediction_controller = PredictionController(prediction_engine=None)


@router.post("/predict", response_model=PredictionResponse)
async def predict_single(customer: CustomerData):
    """Make prediction for a single customer."""
    try:
        return prediction_controller.predict_single(customer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Make predictions for batch of customers."""
    try:
        return prediction_controller.predict_batch(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
