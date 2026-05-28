"""Rate limiting middleware."""

import logging
from datetime import datetime, timedelta
from typing import Dict

logger = logging.getLogger(__name__)


class RateLimiter:
    """Simple in-memory rate limiter."""

    def __init__(self, requests_per_minute: int = 60):
        """Initialize rate limiter."""
        self.requests_per_minute = requests_per_minute
        self.requests: Dict[str, list] = {}

    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed."""
        now = datetime.utcnow()
        minute_ago = now - timedelta(minutes=1)

        if client_id not in self.requests:
            self.requests[client_id] = []

        # Remove old requests
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id] if req_time > minute_ago
        ]

        if len(self.requests[client_id]) >= self.requests_per_minute:
            logger.warning(f"Rate limit exceeded for client: {client_id}")
            return False

        self.requests[client_id].append(now)
        return True
