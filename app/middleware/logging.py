"""Request/response logging middleware."""

import logging
import time
from fastapi import FastAPI, Request

logger = logging.getLogger(__name__)


def setup_logging_middleware(app: FastAPI) -> None:
    """Setup logging middleware."""

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """Log HTTP requests and responses."""
        start_time = time.time()

        # Log request
        logger.info(
            f"Request: {request.method} {request.url.path} - "
            f"Client: {request.client.host if request.client else 'unknown'}"
        )

        response = await call_next(request)

        # Log response
        process_time = time.time() - start_time
        logger.info(
            f"Response: {request.method} {request.url.path} - "
            f"Status: {response.status_code} - Time: {process_time:.3f}s"
        )

        response.headers["X-Process-Time"] = str(process_time)
        return response
