"""Health routes - Route layer.

Layered Architecture: Route Layer
- Exposes system health and status endpoints
- Delegates business logic to service
"""

import logging

from fastapi import APIRouter, Depends

from app.config.settings import get_settings
from app.dependencies import get_prediction_service
from app.models.schemas import HealthResponse
from app.services.health_service import HealthService
from app.services.prediction_service import PredictionService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def health_check(
    prediction_service: PredictionService = Depends(get_prediction_service),
    health_service: HealthService = Depends(),
):
    """
    Health check endpoint.

    Returns the current health status of the API, model, and database.
    """
    settings = get_settings()

    return health_service.check_health(
        model_loaded=prediction_service.is_model_loaded(),
        app_version=settings.app_version,
    )


@router.get("/")
async def root():
    """Root endpoint with API information."""
    settings = get_settings()

    return {
        "message": "Customer Churn Prediction System",
        "version": settings.app_version,
        "docs": "/docs",
    }
