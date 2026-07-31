"""Dependency injection module.

Layered Architecture: Infrastructure Layer
- Provides dependency injection for FastAPI
- Manages service instances and configuration
"""

from functools import lru_cache
from typing import Optional

from app.services.prediction_service import PredictionService

# Global model trainer instance (set during startup)
_model_trainer: Optional[object] = None


def set_model_trainer(model_trainer: object) -> None:
    """Set the global model trainer instance during application startup."""
    global _model_trainer
    _model_trainer = model_trainer


def get_model_trainer() -> Optional[object]:
    """Get the global model trainer instance."""
    return _model_trainer


@lru_cache()
def get_prediction_service() -> PredictionService:
    """
    Get prediction service instance (cached).

    Uses dependency injection pattern for FastAPI.
    """
    return PredictionService(model_trainer=_model_trainer)
