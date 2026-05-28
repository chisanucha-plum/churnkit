"""Health check routes."""

from fastapi import APIRouter

from app.controllers.health_controller import HealthController
from app.models.schemas import HealthResponse

router = APIRouter()
health_controller = HealthController()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check application health."""
    return health_controller.check_health()


@router.get("/ready")
async def readiness_check():
    """Check application readiness."""
    return {"status": "ready"}
