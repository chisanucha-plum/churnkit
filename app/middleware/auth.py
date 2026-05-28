"""API authentication middleware."""

import logging
from fastapi import HTTPException, Depends, Header
from typing import Optional

from app.config.settings import settings

logger = logging.getLogger(__name__)


async def verify_api_key(x_api_key: Optional[str] = Header(None)) -> str:
    """Verify API key from request header."""
    if not x_api_key:
        logger.warning("Missing API key")
        raise HTTPException(status_code=401, detail="Missing API key")

    if x_api_key != settings.API_KEY:
        logger.warning(f"Invalid API key: {x_api_key[:10]}...")
        raise HTTPException(status_code=401, detail="Invalid API key")

    return x_api_key
