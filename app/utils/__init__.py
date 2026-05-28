"""Utilities package for Customer Churn Prediction System."""

from app.utils.exceptions import (
    ChurnPredictionException,
    ModelNotLoadedException,
    InvalidInputException,
    PredictionException,
    DatabaseException,
    CacheException,
    DataValidationException,
    ConfigurationException,
    FeatureEngineeringException,
    ModelTrainingException,
    ExplainabilityException,
    LLMException,
    RateLimitException,
    AuthenticationException,
)

from app.utils.logger import (
    get_logger,
    StructuredLogger,
    JSONFormatter,
    PIIMasker,
    log_execution,
)

from app.utils.validators import InputValidator

from app.utils.helpers import (
    safe_get,
    safe_get_nested,
    format_percentage,
    format_currency,
    format_number,
    chunk_list,
    flatten_list,
    remove_duplicates,
    merge_dicts,
    invert_dict,
    filter_dict,
    exclude_dict,
    calculate_percentage_change,
    normalize_value,
    denormalize_value,
    get_top_n,
    convert_to_numeric,
    convert_to_bool,
    truncate_string,
    get_dataframe_summary,
)

from app.utils.config_loader import (
    ConfigLoader,
    EnvironmentConfig,
)

__all__ = [
    # Exceptions
    "ChurnPredictionException",
    "ModelNotLoadedException",
    "InvalidInputException",
    "PredictionException",
    "DatabaseException",
    "CacheException",
    "DataValidationException",
    "ConfigurationException",
    "FeatureEngineeringException",
    "ModelTrainingException",
    "ExplainabilityException",
    "LLMException",
    "RateLimitException",
    "AuthenticationException",
    # Logger
    "get_logger",
    "StructuredLogger",
    "JSONFormatter",
    "PIIMasker",
    "log_execution",
    # Validators
    "InputValidator",
    # Helpers
    "safe_get",
    "safe_get_nested",
    "format_percentage",
    "format_currency",
    "format_number",
    "chunk_list",
    "flatten_list",
    "remove_duplicates",
    "merge_dicts",
    "invert_dict",
    "filter_dict",
    "exclude_dict",
    "calculate_percentage_change",
    "normalize_value",
    "denormalize_value",
    "get_top_n",
    "convert_to_numeric",
    "convert_to_bool",
    "truncate_string",
    "get_dataframe_summary",
    # Config
    "ConfigLoader",
    "EnvironmentConfig",
]

