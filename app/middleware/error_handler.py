"""Error handling middleware."""

import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


def setup_error_handlers(app: FastAPI) -> None:
    """Setup error handlers."""

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle general exceptions."""
        logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)

        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal Server Error",
                "detail": str(exc),
                "status_code": 500,
            },
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        """Handle value errors."""
        logger.warning(f"Value error: {str(exc)}")

        return JSONResponse(
            status_code=400,
            content={
                "error": "Bad Request",
                "detail": str(exc),
                "status_code": 400,
            },
        )
