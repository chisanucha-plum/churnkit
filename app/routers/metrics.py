"""Metrics routes."""

from fastapi import APIRouter

from app.controllers.metrics_controller import MetricsController
from app.models.schemas import ModelMetrics

router = APIRouter()
metrics_controller = MetricsController()


@router.get("/metrics/model", response_model=ModelMetrics)
async def get_model_metrics():
    """Get model performance metrics."""
    return metrics_controller.get_model_metrics()


@router.get("/metrics/predictions")
async def get_prediction_metrics():
    """Get prediction metrics."""
    return {
        "total_predictions": 0,
        "high_risk_count": 0,
        "average_confidence": 0.0,
    }
