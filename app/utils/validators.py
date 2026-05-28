"""Input validation utilities for the Customer Churn Prediction System.

This module provides comprehensive input validation for:
- Customer data validation
- Batch data validation
- Feature validation
- API request validation
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from app.utils.exceptions import InvalidInputException, DataValidationException

logger = logging.getLogger(__name__)


class InputValidator:
    """Validate input data for predictions and training."""

    # Required fields for customer data
    REQUIRED_CUSTOMER_FIELDS = {
        "tenure": (int, float),
        "monthly_charges": (int, float),
        "total_charges": (int, float),
        "contract_type": str,
        "internet_service": str,
        "online_security": (int, float),
        "online_backup": (int, float),
        "device_protection": (int, float),
        "tech_support": (int, float),
        "streaming_tv": (int, float),
        "streaming_movies": (int, float),
        "payment_method": str,
        "paperless_billing": (int, float),
        "senior_citizen": (int, float),
        "partner": (int, float),
        "dependents": (int, float),
        "phone_service": (int, float),
        "multiple_lines": (int, float),
    }

    # Valid values for categorical fields
    VALID_CONTRACT_TYPES = {"Month-to-month", "One year", "Two year"}
    VALID_INTERNET_SERVICES = {"Fiber optic", "DSL", "No"}
    VALID_PAYMENT_METHODS = {
        "Electronic check", "Mailed check", "Bank transfer", "Credit card"
    }

    @staticmethod
    def validate_customer_data(data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate customer data for prediction.
        
        Args:
            data: Customer data dictionary
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check required fields
            for field, field_type in InputValidator.REQUIRED_CUSTOMER_FIELDS.items():
                if field not in data:
                    return False, f"Missing required field: {field}"

                if not isinstance(data[field], field_type):
                    return False, f"Invalid type for {field}: expected {field_type}, got {type(data[field])}"

            # Validate numeric ranges
            if data["tenure"] < 0:
                return False, "Tenure must be non-negative"

            if data["monthly_charges"] < 0:
                return False, "Monthly charges must be non-negative"

            if data["total_charges"] < 0:
                return False, "Total charges must be non-negative"

            # Validate categorical values
            if data["contract_type"] not in InputValidator.VALID_CONTRACT_TYPES:
                return False, f"Invalid contract_type: {data['contract_type']}"

            if data["internet_service"] not in InputValidator.VALID_INTERNET_SERVICES:
                return False, f"Invalid internet_service: {data['internet_service']}"

            if data["payment_method"] not in InputValidator.VALID_PAYMENT_METHODS:
                return False, f"Invalid payment_method: {data['payment_method']}"

            # Validate customer_id format if provided
            if "customer_id" in data:
                if not isinstance(data["customer_id"], str) or len(data["customer_id"]) == 0:
                    return False, "customer_id must be a non-empty string"

            return True, None

        except Exception as e:
            logger.error(f"Error validating customer data: {str(e)}")
            return False, f"Validation error: {str(e)}"

    @staticmethod
    def validate_batch_data(customers: List[Dict[str, Any]]) -> Tuple[bool, Optional[str]]:
        """Validate batch customer data.
        
        Args:
            customers: List of customer data dictionaries
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            if not customers or len(customers) == 0:
                return False, "Empty customer list"

            if len(customers) > 10000:
                return False, "Batch size exceeds maximum (10000)"

            for idx, customer in enumerate(customers):
                is_valid, error_msg = InputValidator.validate_customer_data(customer)
                if not is_valid:
                    return False, f"Error in customer {idx}: {error_msg}"

            return True, None

        except Exception as e:
            logger.error(f"Error validating batch data: {str(e)}")
            return False, f"Batch validation error: {str(e)}"

    @staticmethod
    def validate_dataframe(df: pd.DataFrame, required_columns: List[str] = None) -> Tuple[bool, Optional[str]]:
        """Validate DataFrame structure and content.
        
        Args:
            df: DataFrame to validate
            required_columns: List of required columns
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            if not isinstance(df, pd.DataFrame):
                return False, "Input must be a pandas DataFrame"

            if df.empty:
                return False, "DataFrame is empty"

            if required_columns:
                missing_cols = set(required_columns) - set(df.columns)
                if missing_cols:
                    return False, f"Missing required columns: {missing_cols}"

            # Check for all NaN columns
            all_nan_cols = df.columns[df.isna().all()].tolist()
            if all_nan_cols:
                return False, f"Columns with all NaN values: {all_nan_cols}"

            return True, None

        except Exception as e:
            logger.error(f"Error validating DataFrame: {str(e)}")
            return False, f"DataFrame validation error: {str(e)}"

    @staticmethod
    def validate_numeric_features(
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        min_samples: int = 10
    ) -> Tuple[bool, Optional[str]]:
        """Validate numeric features for model training.
        
        Args:
            X: Feature matrix
            y: Target variable (optional)
            min_samples: Minimum number of samples required
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check DataFrame
            is_valid, error_msg = InputValidator.validate_dataframe(X)
            if not is_valid:
                return False, error_msg

            # Check minimum samples
            if len(X) < min_samples:
                return False, f"Insufficient samples: {len(X)} < {min_samples}"

            # Check for NaN values
            nan_count = X.isna().sum().sum()
            if nan_count > 0:
                nan_pct = (nan_count / (X.shape[0] * X.shape[1])) * 100
                if nan_pct > 50:
                    return False, f"Too many NaN values: {nan_pct:.1f}%"

            # Check for infinite values
            inf_count = np.isinf(X.select_dtypes(include=[np.number])).sum().sum()
            if inf_count > 0:
                return False, f"Found {inf_count} infinite values"

            # Validate target if provided
            if y is not None:
                if len(y) != len(X):
                    return False, f"Target length {len(y)} != feature length {len(X)}"

                if y.isna().any():
                    return False, f"Target contains {y.isna().sum()} NaN values"

            return True, None

        except Exception as e:
            logger.error(f"Error validating numeric features: {str(e)}")
            return False, f"Feature validation error: {str(e)}"

    @staticmethod
    def validate_prediction_input(data: Dict[str, Any]) -> None:
        """Validate prediction input and raise exception if invalid.
        
        Args:
            data: Input data to validate
            
        Raises:
            InvalidInputException: If validation fails
        """
        is_valid, error_msg = InputValidator.validate_customer_data(data)
        if not is_valid:
            raise InvalidInputException(
                message=error_msg,
                details={"input": data}
            )

    @staticmethod
    def validate_batch_prediction_input(customers: List[Dict[str, Any]]) -> None:
        """Validate batch prediction input and raise exception if invalid.
        
        Args:
            customers: List of customer data
            
        Raises:
            InvalidInputException: If validation fails
        """
        is_valid, error_msg = InputValidator.validate_batch_data(customers)
        if not is_valid:
            raise InvalidInputException(
                message=error_msg,
                details={"batch_size": len(customers) if customers else 0}
            )

    @staticmethod
    def validate_training_data(
        X: pd.DataFrame,
        y: pd.Series,
        min_samples: int = 100
    ) -> None:
        """Validate training data and raise exception if invalid.
        
        Args:
            X: Feature matrix
            y: Target variable
            min_samples: Minimum number of samples
            
        Raises:
            DataValidationException: If validation fails
        """
        is_valid, error_msg = InputValidator.validate_numeric_features(X, y, min_samples)
        if not is_valid:
            raise DataValidationException(
                message=error_msg,
                details={
                    "X_shape": X.shape if isinstance(X, pd.DataFrame) else None,
                    "y_shape": y.shape if isinstance(y, pd.Series) else None
                }
            )
