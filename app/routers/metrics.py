"""Metrics routes - Route layer.

Layered Architecture: Route Layer
- Exposes model metrics and feature information
"""

import logging

from fastapi import APIRouter, Depends, HTTPException

from app.models.schemas import ModelMetrics, FeatureImportanceResponse
from app.services.prediction_service import PredictionService
from app.dependencies import get_prediction_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Metrics"])


@router.get("/metrics", response_model=ModelMetrics)
async def get_metrics(
    service: PredictionService = Depends(get_prediction_service)
):
    """Get model performance metrics."""
    metrics = service.get_metrics()
    
    if not metrics:
        raise HTTPException(status_code=503, detail="Model metrics not available")
    
    return ModelMetrics(
        accuracy=metrics.get("accuracy", 0.0),
        precision=metrics.get("precision", 0.0),
        recall=metrics.get("recall", 0.0),
        f1_score=metrics.get("f1_score", 0.0),
        roc_auc=metrics.get("roc_auc", metrics.get("auc_roc", 0.0))
    )


@router.get("/features")
async def get_features(
    service: PredictionService = Depends(get_prediction_service)
):
    """Get model feature names."""
    if not service.is_model_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    feature_names = service.get_feature_names()
    
    return {
        "features": feature_names,
        "count": len(feature_names) if feature_names else 0
    }


@router.get("/feature-importance", response_model=FeatureImportanceResponse)
async def get_feature_importance(
    service: PredictionService = Depends(get_prediction_service)
):
    """Get feature importance from the model."""
    importance = service.get_feature_importance()
    
    if not importance:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return FeatureImportanceResponse(
        importance=importance,
        top_10=dict(list(importance.items())[:10])
    )
