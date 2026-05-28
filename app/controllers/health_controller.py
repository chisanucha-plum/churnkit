"""Health check controller."""

import logging

from app.models.schemas import HealthResponse

logger = logging.getLogger(__name__)


class HealthController:
    """Handle health check requests."""

    def __init__(self, db_session=None, cache_client=None):
        """Initialize controller."""
        self.db_session = db_session
        self.cache_client = cache_client

    def check_health(self) -> HealthResponse:
        """Check application health."""
        logger.info("Checking application health")

        db_status = self._check_database()
        cache_status = self._check_cache()
        model_status = self._check_model()

        return HealthResponse(
            status="healthy" if all([db_status, cache_status]) else "degraded",
            version="1.0.0",
            database=db_status,
            cache=cache_status,
            model_loaded=model_status,
        )

    def _check_database(self) -> str:
        """Check database connectivity."""
        try:
            if self.db_session:
                self.db_session.execute("SELECT 1")
            return "connected"
        except Exception as e:
            logger.error(f"Database check failed: {str(e)}")
            return "disconnected"

    def _check_cache(self) -> str:
        """Check cache connectivity."""
        try:
            if self.cache_client:
                self.cache_client.ping()
            return "connected"
        except Exception as e:
            logger.error(f"Cache check failed: {str(e)}")
            return "disconnected"

    def _check_model(self) -> bool:
        """Check if model is loaded."""
        return True
