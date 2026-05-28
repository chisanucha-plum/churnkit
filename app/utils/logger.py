"""Logging utilities with PII masking and structured logging.

This module provides logging utilities including:
- JSON formatted logging
- PII (Personally Identifiable Information) masking
- Structured logging with context
- Request ID propagation for distributed tracing
"""

import logging
import json
import re
from datetime import datetime
from typing import Any, Dict, Optional
from functools import wraps

from app.config.settings import get_settings


class PIIMasker:
    """Mask PII in log messages."""

    # PII patterns to mask
    PII_PATTERNS = {
        "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "credit_card": r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
        "ip_address": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
        "customer_id": r"(?i)customer[_-]?id[:\s]*([A-Z0-9_-]+)",
        "api_key": r"(?i)api[_-]?key[:\s]*([A-Za-z0-9_-]+)",
    }

    @staticmethod
    def mask_pii(text: str) -> str:
        """Mask PII in text.
        
        Args:
            text: Text to mask
            
        Returns:
            Text with PII masked
        """
        if not isinstance(text, str):
            return text

        masked_text = text
        for pii_type, pattern in PIIMasker.PII_PATTERNS.items():
            masked_text = re.sub(pattern, f"[{pii_type.upper()}_MASKED]", masked_text)

        return masked_text

    @staticmethod
    def mask_dict(data: Dict[str, Any]) -> Dict[str, Any]:
        """Mask PII in dictionary.
        
        Args:
            data: Dictionary to mask
            
        Returns:
            Dictionary with PII masked
        """
        masked_data = {}
        sensitive_keys = {
            "password", "api_key", "secret", "token", "email",
            "phone", "ssn", "credit_card", "customer_id", "user_id"
        }

        for key, value in data.items():
            if key.lower() in sensitive_keys:
                masked_data[key] = "[MASKED]"
            elif isinstance(value, str):
                masked_data[key] = PIIMasker.mask_pii(value)
            elif isinstance(value, dict):
                masked_data[key] = PIIMasker.mask_dict(value)
            elif isinstance(value, list):
                masked_data[key] = [
                    PIIMasker.mask_dict(item) if isinstance(item, dict)
                    else PIIMasker.mask_pii(item) if isinstance(item, str)
                    else item
                    for item in value
                ]
            else:
                masked_data[key] = value

        return masked_data


class JSONFormatter(logging.Formatter):
    """JSON log formatter with PII masking."""

    def __init__(self, mask_pii: bool = True):
        """Initialize formatter.
        
        Args:
            mask_pii: Whether to mask PII in logs
        """
        super().__init__()
        self.mask_pii = mask_pii

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON.
        
        Args:
            record: Log record to format
            
        Returns:
            JSON formatted log string
        """
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add request ID if available
        if hasattr(record, "request_id"):
            log_data["request_id"] = record.request_id

        # Add extra fields
        if hasattr(record, "extra"):
            log_data.update(record.extra)

        # Add exception info
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Mask PII if enabled
        if self.mask_pii:
            log_data["message"] = PIIMasker.mask_pii(log_data["message"])
            if "extra" in log_data:
                log_data["extra"] = PIIMasker.mask_dict(log_data["extra"])

        return json.dumps(log_data)


class StructuredLogger:
    """Structured logger with context and PII masking."""

    def __init__(self, name: str, mask_pii: bool = True):
        """Initialize logger.
        
        Args:
            name: Logger name
            mask_pii: Whether to mask PII
        """
        self.logger = logging.getLogger(name)
        self.mask_pii = mask_pii
        self.context = {}

    def set_context(self, **kwargs):
        """Set logging context.
        
        Args:
            **kwargs: Context key-value pairs
        """
        self.context.update(kwargs)

    def clear_context(self):
        """Clear logging context."""
        self.context.clear()

    def _log(self, level: int, message: str, extra: Dict[str, Any] = None, **kwargs):
        """Internal logging method.
        
        Args:
            level: Log level
            message: Log message
            extra: Extra fields to include
            **kwargs: Additional arguments
        """
        log_extra = {**self.context}
        if extra:
            log_extra.update(extra)

        # Mask PII if enabled
        if self.mask_pii:
            message = PIIMasker.mask_pii(message)
            log_extra = PIIMasker.mask_dict(log_extra)

        self.logger.log(level, message, extra={"extra": log_extra}, **kwargs)

    def debug(self, message: str, extra: Dict[str, Any] = None, **kwargs):
        """Log debug message."""
        self._log(logging.DEBUG, message, extra, **kwargs)

    def info(self, message: str, extra: Dict[str, Any] = None, **kwargs):
        """Log info message."""
        self._log(logging.INFO, message, extra, **kwargs)

    def warning(self, message: str, extra: Dict[str, Any] = None, **kwargs):
        """Log warning message."""
        self._log(logging.WARNING, message, extra, **kwargs)

    def error(self, message: str, extra: Dict[str, Any] = None, **kwargs):
        """Log error message."""
        self._log(logging.ERROR, message, extra, **kwargs)

    def critical(self, message: str, extra: Dict[str, Any] = None, **kwargs):
        """Log critical message."""
        self._log(logging.CRITICAL, message, extra, **kwargs)


def get_logger(name: str, mask_pii: bool = True) -> StructuredLogger:
    """Get configured logger.
    
    Args:
        name: Logger name
        mask_pii: Whether to mask PII
        
    Returns:
        StructuredLogger instance
    """
    settings = get_settings()
    logger = StructuredLogger(name, mask_pii=mask_pii and settings.mask_pii_in_logs)
    return logger


def log_execution(logger: Optional[StructuredLogger] = None):
    """Decorator to log function execution.
    
    Args:
        logger: Logger instance (optional)
        
    Returns:
        Decorator function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_logger = logger or get_logger(func.__module__)
            func_logger.debug(
                f"Executing {func.__name__}",
                extra={"function": func.__name__, "args_count": len(args), "kwargs_count": len(kwargs)}
            )
            try:
                result = func(*args, **kwargs)
                func_logger.debug(
                    f"Completed {func.__name__}",
                    extra={"function": func.__name__, "status": "success"}
                )
                return result
            except Exception as e:
                func_logger.error(
                    f"Error in {func.__name__}: {str(e)}",
                    extra={"function": func.__name__, "status": "error", "error": str(e)},
                    exc_info=True
                )
                raise
        return wrapper
    return decorator
