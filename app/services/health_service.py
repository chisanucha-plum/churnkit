"""Health check service - Business logic layer.

Layered Architecture: Service Layer
- Contains business logic for health checks
- Orchestrates database and cache checks
- No direct HTTP framework dependencies
"""

import logging

from app.models.schemas import HealthResponse
from app.database.connection import get_db_manager

logger = logging.getLogger(__name__)


class HealthService:
    """Service for handling health check business logic."""

    def __init__(self, db_manager=None):
        """Initialize health service with database manager."""
        self.db_manager = db_manager or get_db_manager()

    def check_health(self, model_loaded: bool, app_version: str) -> HealthResponse:
        """
        Check application health.

        Args:
            model_loaded: Whether the ML model is loaded
            app_version: Application version string

        Returns:
            HealthResponse with system health status
        """
        logger.info("Checking application health")

        db_healthy = self._check_database()

        overall_status = "healthy" if db_healthy else "degraded"

        return HealthResponse(
            status=overall_status,
            model_loaded=model_loaded,
            database_healthy=db_healthy,
            version=app_version
        )

    def _check_database(self) -> bool:
        """
        Check database connectivity.

        Returns:
            True if database is healthy, False otherwise
        """
        try:
            if self.db_manager:
                return self.db_manager.health_check()
            return False
        except Exception as e:
            logger.error(f"Database check failed: {str(e)}")
            return False
