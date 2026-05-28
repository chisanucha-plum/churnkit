"""Cache management."""

import logging
import json
from typing import Any, Optional

logger = logging.getLogger(__name__)


class CacheManager:
    """In-memory cache manager."""

    def __init__(self, ttl: int = 3600):
        """Initialize cache manager."""
        self.cache = {}
        self.ttl = ttl
        logger.info(f"Cache manager initialized with TTL: {ttl}s")

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key in self.cache:
            logger.debug(f"Cache hit: {key}")
            return self.cache[key]

        logger.debug(f"Cache miss: {key}")
        return None

    def set(self, key: str, value: Any) -> None:
        """Set value in cache."""
        self.cache[key] = value
        logger.debug(f"Cache set: {key}")

    def delete(self, key: str) -> None:
        """Delete value from cache."""
        if key in self.cache:
            del self.cache[key]
            logger.debug(f"Cache deleted: {key}")

    def clear(self) -> None:
        """Clear all cache."""
        self.cache.clear()
        logger.info("Cache cleared")

    def get_stats(self) -> dict:
        """Get cache statistics."""
        return {
            "size": len(self.cache),
            "keys": list(self.cache.keys()),
        }
