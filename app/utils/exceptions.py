"""Custom exceptions for the Customer Churn Prediction System.

This module defines all custom exceptions used throughout the application.
Each exception includes error codes and detailed messages for better debugging.
"""


class ChurnPredictionException(Exception):
    """Base exception for churn prediction system.
    
    All custom exceptions inherit from this base class.
    """

    def __init__(self, message: str, error_code: str = "UNKNOWN_ERROR", details: dict = None):
        """Initialize exception.
        
        Args:
            message: Error message
            error_code: Error code for categorization
            details: Additional error details
        """
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)

    def to_dict(self) -> dict:
        """Convert exception to dictionary for API responses.
        
        Returns:
            Dictionary with error information
        """
        return {
            "error": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
        }


class ModelNotLoadedException(ChurnPredictionException):
    """Raised when model is not loaded or fails to load."""

    def __init__(self, message: str = "Model not loaded", details: dict = None):
        """Initialize exception."""
        super().__init__(message, "MODEL_NOT_LOADED", details)


class InvalidInputException(ChurnPredictionException):
    """Raised when input data is invalid or malformed."""

    def __init__(self, message: str = "Invalid input", details: dict = None):
        """Initialize exception."""
        super().__init__(message, "INVALID_INPUT", details)


class PredictionException(ChurnPredictionException):
    """Raised when prediction fails."""

    def __init__(self, message: str = "Prediction failed", details: dict = None):
        """Initialize exception."""
        super().__init__(message, "PREDICTION_ERROR", details)


class DatabaseException(ChurnPredictionException):
    """Raised when database operation fails."""

    def __init__(self, message: str = "Database operation failed", details: dict = None):
        """Initialize exception."""
        super().__init__(message, "DATABASE_ERROR", details)


class CacheException(ChurnPredictionException):
    """Raised when cache operation fails."""

    def __init__(self, message: str = "Cache operation failed", details: dict = None):
        """Initialize exception."""
        super().__init__(message, "CACHE_ERROR", details)


class DataValidationException(ChurnPredictionException):
    """Raised when data validation fails."""

    def __init__(self, message: str = "Data validation failed", details: dict = None):
        """Initialize exception."""
        super().__init__(message, "DATA_VALIDATION_ERROR", details)


class ConfigurationException(ChurnPredictionException):
    """Raised when configuration is invalid."""

    def __init__(self, message: str = "Configuration error", details: dict = None):
        """Initialize exception."""
        super().__init__(message, "CONFIG_ERROR", details)


class FeatureEngineeringException(ChurnPredictionException):
    """Raised when feature engineering fails."""

    def __init__(self, message: str = "Feature engineering failed", details: dict = None):
        """Initialize exception."""
        super().__init__(message, "FEATURE_ENGINEERING_ERROR", details)


class ModelTrainingException(ChurnPredictionException):
    """Raised when model training fails."""

    def __init__(self, message: str = "Model training failed", details: dict = None):
        """Initialize exception."""
        super().__init__(message, "MODEL_TRAINING_ERROR", details)


class ExplainabilityException(ChurnPredictionException):
    """Raised when explainability computation fails."""

    def __init__(self, message: str = "Explainability computation failed", details: dict = None):
        """Initialize exception."""
        super().__init__(message, "EXPLAINABILITY_ERROR", details)


class LLMException(ChurnPredictionException):
    """Raised when LLM API call fails."""

    def __init__(self, message: str = "LLM API call failed", details: dict = None):
        """Initialize exception."""
        super().__init__(message, "LLM_ERROR", details)


class RateLimitException(ChurnPredictionException):
    """Raised when rate limit is exceeded."""

    def __init__(self, message: str = "Rate limit exceeded", details: dict = None):
        """Initialize exception."""
        super().__init__(message, "RATE_LIMIT_EXCEEDED", details)


class AuthenticationException(ChurnPredictionException):
    """Raised when authentication fails."""

    def __init__(self, message: str = "Authentication failed", details: dict = None):
        """Initialize exception."""
        super().__init__(message, "AUTH_ERROR", details)
