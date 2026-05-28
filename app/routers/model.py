"""Model management routes."""

from fastapi import APIRouter

from app.controllers.model_controller import ModelController

router = APIRouter()
model_controller = ModelController()


@router.get("/model/info")
async def get_model_info():
    """Get model information."""
    return model_controller.get_model_info()


@router.get("/model/versions")
async def get_model_versions():
    """Get available model versions."""
    return model_controller.get_model_versions()


@router.post("/model/retrain")
async def retrain_model():
    """Trigger model retraining."""
    return model_controller.retrain_model()
