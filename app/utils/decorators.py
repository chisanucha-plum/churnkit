"""Custom decorators."""

import logging
import time
from functools import wraps
from typing import Callable, Any

logger = logging.getLogger(__name__)


def log_execution_time(func: Callable) -> Callable:
    """Decorator to log function execution time."""

    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        logger.info(f"Starting execution of {func.__name__}")

        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(
                f"Completed {func.__name__} in {execution_time:.3f} seconds"
            )
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(
                f"Error in {func.__name__} after {execution_time:.3f} seconds: {str(e)}"
            )
            raise

    return wrapper


def retry(max_attempts: int = 3, delay: float = 1.0):
    """Decorator to retry function on failure."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        logger.error(f"Failed after {max_attempts} attempts: {str(e)}")
                        raise

                    logger.warning(
                        f"Attempt {attempt + 1} failed, retrying in {delay}s: {str(e)}"
                    )
                    time.sleep(delay)

        return wrapper

    return decorator
