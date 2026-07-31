"""Unit tests for utility modules.

Tests for:
- Exception handling
- Logging with PII masking
- Input validation
- Helper functions
- Configuration loading
"""

from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

from app.utils.config_loader import ConfigLoader, EnvironmentConfig
from app.utils.exceptions import (
    ConfigurationException,
    DataValidationException,
    InvalidInputException,
)
from app.utils.helpers import (
    chunk_list,
    convert_to_bool,
    convert_to_numeric,
    format_currency,
    format_percentage,
    normalize_value,
    safe_get,
)
from app.utils.logger import PIIMasker, StructuredLogger, get_logger
from app.utils.validators import InputValidator

# ===== EXCEPTION TESTS =====


class TestExceptions:
    """Test custom exceptions."""

    def test_invalid_input_exception(self):
        """Test InvalidInputException."""
        exc = InvalidInputException("Test error", details={"field": "test"})
        assert exc.error_code == "INVALID_INPUT"
        assert exc.message == "Test error"
        assert exc.details == {"field": "test"}

    def test_exception_to_dict(self):
        """Test exception to_dict method."""
        exc = InvalidInputException("Test error")
        exc_dict = exc.to_dict()
        assert exc_dict["error"] == "InvalidInputException"
        assert exc_dict["error_code"] == "INVALID_INPUT"
        assert exc_dict["message"] == "Test error"


# ===== PII MASKING TESTS =====


class TestPIIMasker:
    """Test PII masking functionality."""

    def test_mask_email(self):
        """Test email masking."""
        text = "Contact user@example.com for support"
        masked = PIIMasker.mask_pii(text)
        assert "user@example.com" not in masked
        assert "[EMAIL_MASKED]" in masked

    def test_mask_phone(self):
        """Test phone number masking."""
        text = "Call 555-123-4567 for help"
        masked = PIIMasker.mask_pii(text)
        assert "555-123-4567" not in masked
        assert "[PHONE_MASKED]" in masked

    def test_mask_dict(self):
        """Test dictionary masking."""
        data = {
            "customer_id": "CUST_001",
            "email": "user@example.com",
            "password": "secret123",
            "api_key": "key_12345",
        }
        masked = PIIMasker.mask_dict(data)
        assert masked["email"] == "[MASKED]"
        assert masked["password"] == "[MASKED]"
        assert masked["api_key"] == "[MASKED]"
        assert masked["customer_id"] == "CUST_001"

    def test_mask_nested_dict(self):
        """Test nested dictionary masking."""
        data = {
            "user": {
                "email": "user@example.com",
                "phone": "555-123-4567",
            }
        }
        masked = PIIMasker.mask_dict(data)
        assert masked["user"]["email"] == "[MASKED]"
        assert "[PHONE_MASKED]" in masked["user"]["phone"]


# ===== LOGGER TESTS =====


class TestStructuredLogger:
    """Test structured logger."""

    def test_get_logger(self):
        """Test getting logger."""
        logger = get_logger("test_module")
        assert isinstance(logger, StructuredLogger)
        assert logger.logger.name == "test_module"

    def test_logger_context(self):
        """Test logger context."""
        logger = get_logger("test_module")
        logger.set_context(request_id="123", user_id="456")
        assert logger.context["request_id"] == "123"
        assert logger.context["user_id"] == "456"

    def test_logger_clear_context(self):
        """Test clearing logger context."""
        logger = get_logger("test_module")
        logger.set_context(request_id="123")
        logger.clear_context()
        assert len(logger.context) == 0


# ===== VALIDATOR TESTS =====


class TestInputValidator:
    """Test input validation."""

    def test_validate_valid_customer_data(self):
        """Test validating valid customer data."""
        data = {
            "customer_id": "CUST_001",
            "tenure": 24,
            "monthly_charges": 89.50,
            "total_charges": 2148.00,
            "contract_type": "Month-to-month",
            "internet_service": "Fiber optic",
            "payment_method": "Electronic check",
        }
        is_valid, error = InputValidator.validate_customer_data(data)
        assert is_valid is True
        assert error is None

    def test_validate_missing_field(self):
        """Test validation with missing field."""
        data = {
            "customer_id": "CUST_001",
            "tenure": 24,
            # Missing monthly_charges
            "total_charges": 2148.00,
            "contract_type": "Month-to-month",
            "internet_service": "Fiber optic",
            "payment_method": "Electronic check",
        }
        is_valid, error = InputValidator.validate_customer_data(data)
        assert is_valid is False
        assert "monthly_charges" in error

    def test_validate_invalid_contract_type(self):
        """Test validation with invalid contract type."""
        data = {
            "customer_id": "CUST_001",
            "tenure": 24,
            "monthly_charges": 89.50,
            "total_charges": 2148.00,
            "contract_type": "Invalid",
            "internet_service": "Fiber optic",
            "payment_method": "Electronic check",
        }
        is_valid, error = InputValidator.validate_customer_data(data)
        assert is_valid is False
        assert "contract_type" in error

    def test_validate_negative_tenure(self):
        """Test validation with negative tenure."""
        data = {
            "customer_id": "CUST_001",
            "tenure": -5,
            "monthly_charges": 89.50,
            "total_charges": 2148.00,
            "contract_type": "Month-to-month",
            "internet_service": "Fiber optic",
            "payment_method": "Electronic check",
        }
        is_valid, error = InputValidator.validate_customer_data(data)
        assert is_valid is False
        assert "Tenure" in error

    def test_validate_batch_data(self):
        """Test batch validation."""
        customers = [
            {
                "customer_id": f"CUST_{i:03d}",
                "tenure": 24,
                "monthly_charges": 89.50,
                "total_charges": 2148.00,
                "contract_type": "Month-to-month",
                "internet_service": "Fiber optic",
                "payment_method": "Electronic check",
            }
            for i in range(5)
        ]
        is_valid, error = InputValidator.validate_batch_data(customers)
        assert is_valid is True
        assert error is None

    def test_validate_empty_batch(self):
        """Test validation with empty batch."""
        is_valid, error = InputValidator.validate_batch_data([])
        assert is_valid is False
        assert "Empty" in error

    def test_validate_dataframe(self):
        """Test DataFrame validation."""
        df = pd.DataFrame(
            {
                "col1": [1, 2, 3],
                "col2": [4, 5, 6],
            }
        )
        is_valid, error = InputValidator.validate_dataframe(df)
        assert is_valid is True
        assert error is None

    def test_validate_dataframe_missing_columns(self):
        """Test DataFrame validation with missing columns."""
        df = pd.DataFrame({"col1": [1, 2, 3]})
        is_valid, error = InputValidator.validate_dataframe(
            df, required_columns=["col1", "col2"]
        )
        assert is_valid is False
        assert "col2" in error

    def test_validate_numeric_features(self):
        """Test numeric features validation."""
        X = pd.DataFrame(
            {
                "f1": np.random.rand(100),
                "f2": np.random.rand(100),
            }
        )
        y = pd.Series(np.random.randint(0, 2, 100))
        is_valid, error = InputValidator.validate_numeric_features(X, y)
        assert is_valid is True
        assert error is None


# ===== HELPER TESTS =====


class TestHelpers:
    """Test helper functions."""

    def test_safe_get(self):
        """Test safe_get function."""
        data = {"key": "value"}
        assert safe_get(data, "key") == "value"
        assert safe_get(data, "missing", "default") == "default"

    def test_format_percentage(self):
        """Test format_percentage function."""
        assert format_percentage(0.5) == "50.00%"
        assert format_percentage(0.333, decimals=1) == "33.3%"

    def test_format_currency(self):
        """Test format_currency function."""
        assert format_currency(1234.56) == "$1,234.56"
        assert format_currency(1000, currency="€") == "€1,000.00"

    def test_chunk_list(self):
        """Test chunk_list function."""
        lst = [1, 2, 3, 4, 5, 6, 7]
        chunks = chunk_list(lst, 3)
        assert len(chunks) == 3
        assert chunks[0] == [1, 2, 3]
        assert chunks[1] == [4, 5, 6]
        assert chunks[2] == [7]

    def test_normalize_value(self):
        """Test normalize_value function."""
        normalized = normalize_value(5, 0, 10)
        assert normalized == 0.5
        assert normalize_value(0, 0, 10) == 0.0
        assert normalize_value(10, 0, 10) == 1.0

    def test_convert_to_numeric(self):
        """Test convert_to_numeric function."""
        assert convert_to_numeric(42) == 42.0
        assert convert_to_numeric("3.14") == 3.14
        assert convert_to_numeric("invalid", default=0.0) == 0.0

    def test_convert_to_bool(self):
        """Test convert_to_bool function."""
        assert convert_to_bool(True) is True
        assert convert_to_bool("yes") is True
        assert convert_to_bool("false") is False
        assert convert_to_bool(1) is True
        assert convert_to_bool(0) is False


# ===== CONFIG LOADER TESTS =====


class TestConfigLoader:
    """Test configuration loader."""

    def test_merge_configs(self):
        """Test merging configurations."""
        config1 = {"key1": "value1", "key2": "value2"}
        config2 = {"key2": "updated", "key3": "value3"}
        merged = ConfigLoader.merge_configs(config1, config2)
        assert merged["key1"] == "value1"
        assert merged["key2"] == "updated"
        assert merged["key3"] == "value3"

    def test_get_config_value(self):
        """Test getting config value."""
        config = {"database_url": "postgresql://localhost"}
        value = ConfigLoader.get_config_value("database_url", config=config)
        assert value == "postgresql://localhost"

    def test_validate_required_config(self):
        """Test validating required config."""
        config = {"key1": "value1", "key2": "value2"}
        # Should not raise
        ConfigLoader.validate_required_config(config, ["key1", "key2"])

    def test_validate_required_config_missing(self):
        """Test validating required config with missing keys."""
        config = {"key1": "value1"}
        with pytest.raises(ConfigurationException):
            ConfigLoader.validate_required_config(config, ["key1", "key2"])


# ===== ENVIRONMENT CONFIG TESTS =====


class TestEnvironmentConfig:
    """Test environment configuration."""

    def test_is_development(self, monkeypatch):
        """Test is_development check."""
        monkeypatch.setenv("ENVIRONMENT", "development")
        assert EnvironmentConfig.is_development() is True
        assert EnvironmentConfig.is_production() is False

    def test_is_production(self, monkeypatch):
        """Test is_production check."""
        monkeypatch.setenv("ENVIRONMENT", "production")
        assert EnvironmentConfig.is_production() is True
        assert EnvironmentConfig.is_development() is False

    def test_is_testing(self, monkeypatch):
        """Test is_testing check."""
        monkeypatch.setenv("ENVIRONMENT", "testing")
        assert EnvironmentConfig.is_testing() is True
