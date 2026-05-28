"""
Unit tests for configuration management.

Tests the Settings class and configuration loading from environment variables.
Validates all configuration options and their defaults.

Requirements: 1.2, 33.1, 33.6
"""

import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from app.config.settings import Settings, get_settings


class TestSettingsDefaults:
    """Test default configuration values."""
    
    @pytest.mark.unit
    def test_default_app_name(self):
        """Test default application name."""
        settings = Settings()
        assert settings.app_name == "Customer Churn Prediction System"
    
    @pytest.mark.unit
    def test_default_app_version(self):
        """Test default application version."""
        settings = Settings()
        assert settings.app_version == "1.0.0"
    
    @pytest.mark.unit
    def test_default_environment(self):
        """Test default environment is development."""
        settings = Settings()
        assert settings.environment == "development"
    
    @pytest.mark.unit
    def test_default_debug_mode(self):
        """Test default debug mode is enabled."""
        settings = Settings()
        assert settings.debug is True
    
    @pytest.mark.unit
    def test_default_database_pool_size(self):
        """Test default database pool size."""
        settings = Settings()
        assert settings.database_pool_size == 10
    
    @pytest.mark.unit
    def test_default_api_host(self):
        """Test default API host."""
        settings = Settings()
        assert settings.api_host == "0.0.0.0"
    
    @pytest.mark.unit
    def test_default_api_port(self):
        """Test default API port."""
        settings = Settings()
        assert settings.api_port == 8000
    
    @pytest.mark.unit
    def test_default_log_level(self):
        """Test default log level."""
        settings = Settings()
        assert settings.log_level == "INFO"
    
    @pytest.mark.unit
    def test_default_train_test_split_ratio(self):
        """Test default train/test split ratio."""
        settings = Settings()
        assert settings.train_test_split_ratio == 0.8
    
    @pytest.mark.unit
    def test_default_random_state(self):
        """Test default random state."""
        settings = Settings()
        assert settings.random_state == 42


class TestSettingsValidation:
    """Test configuration validation."""
    
    @pytest.mark.unit
    def test_valid_environment_development(self):
        """Test valid development environment."""
        settings = Settings(environment="development")
        assert settings.environment == "development"
    
    @pytest.mark.unit
    def test_valid_environment_staging(self):
        """Test valid staging environment."""
        settings = Settings(environment="staging")
        assert settings.environment == "staging"
    
    @pytest.mark.unit
    def test_valid_environment_production(self):
        """Test valid production environment."""
        settings = Settings(environment="production")
        assert settings.environment == "production"
    
    @pytest.mark.unit
    def test_invalid_environment_raises_error(self):
        """Test invalid environment raises validation error."""
        with pytest.raises(ValueError, match="Environment must be one of"):
            Settings(environment="invalid")
    
    @pytest.mark.unit
    def test_valid_log_level_debug(self):
        """Test valid DEBUG log level."""
        settings = Settings(log_level="DEBUG")
        assert settings.log_level == "DEBUG"
    
    @pytest.mark.unit
    def test_valid_log_level_info(self):
        """Test valid INFO log level."""
        settings = Settings(log_level="INFO")
        assert settings.log_level == "INFO"
    
    @pytest.mark.unit
    def test_valid_log_level_warning(self):
        """Test valid WARNING log level."""
        settings = Settings(log_level="WARNING")
        assert settings.log_level == "WARNING"
    
    @pytest.mark.unit
    def test_valid_log_level_error(self):
        """Test valid ERROR log level."""
        settings = Settings(log_level="ERROR")
        assert settings.log_level == "ERROR"
    
    @pytest.mark.unit
    def test_invalid_log_level_raises_error(self):
        """Test invalid log level raises validation error."""
        with pytest.raises(ValueError, match="Log level must be one of"):
            Settings(log_level="INVALID")
    
    @pytest.mark.unit
    def test_log_level_case_insensitive(self):
        """Test log level is case-insensitive."""
        settings = Settings(log_level="debug")
        assert settings.log_level == "DEBUG"
    
    @pytest.mark.unit
    def test_train_test_split_ratio_valid(self):
        """Test valid train/test split ratio."""
        settings = Settings(train_test_split_ratio=0.75)
        assert settings.train_test_split_ratio == 0.75
    
    @pytest.mark.unit
    def test_api_port_valid(self):
        """Test valid API port."""
        settings = Settings(api_port=9000)
        assert settings.api_port == 9000
    
    @pytest.mark.unit
    def test_database_pool_size_valid(self):
        """Test valid database pool size."""
        settings = Settings(database_pool_size=20)
        assert settings.database_pool_size == 20


class TestSettingsMethods:
    """Test Settings class methods."""
    
    @pytest.mark.unit
    def test_get_database_url_sqlite_default(self):
        """Test get_database_url returns SQLite URL by default."""
        settings = Settings()
        db_url = settings.get_database_url()
        assert "sqlite:///" in db_url
    
    @pytest.mark.unit
    def test_get_database_url_postgresql(self):
        """Test get_database_url returns PostgreSQL URL when configured."""
        pg_url = "postgresql://user:pass@localhost:5432/churn_db"
        settings = Settings(database_url=pg_url)
        db_url = settings.get_database_url()
        assert db_url == pg_url
    
    @pytest.mark.unit
    def test_get_api_keys_list_single_key(self):
        """Test get_api_keys_list with single key."""
        settings = Settings(api_keys="key123")
        keys = settings.get_api_keys_list()
        assert keys == ["key123"]
    
    @pytest.mark.unit
    def test_get_api_keys_list_multiple_keys(self):
        """Test get_api_keys_list with multiple keys."""
        settings = Settings(api_keys="key1,key2,key3")
        keys = settings.get_api_keys_list()
        assert keys == ["key1", "key2", "key3"]
    
    @pytest.mark.unit
    def test_get_api_keys_list_with_spaces(self):
        """Test get_api_keys_list strips whitespace."""
        settings = Settings(api_keys="key1 , key2 , key3")
        keys = settings.get_api_keys_list()
        assert keys == ["key1", "key2", "key3"]
    
    @pytest.mark.unit
    def test_is_production_true(self):
        """Test is_production returns True for production environment."""
        settings = Settings(environment="production")
        assert settings.is_production() is True
    
    @pytest.mark.unit
    def test_is_production_false(self):
        """Test is_production returns False for non-production environment."""
        settings = Settings(environment="development")
        assert settings.is_production() is False
    
    @pytest.mark.unit
    def test_is_development_true(self):
        """Test is_development returns True for development environment."""
        settings = Settings(environment="development")
        assert settings.is_development() is True
    
    @pytest.mark.unit
    def test_is_development_false(self):
        """Test is_development returns False for non-development environment."""
        settings = Settings(environment="production")
        assert settings.is_development() is False


class TestSettingsEnvironmentVariables:
    """Test loading settings from environment variables."""
    
    @pytest.mark.unit
    def test_load_from_env_app_name(self):
        """Test loading app_name from environment variable."""
        with patch.dict(os.environ, {"APP_NAME": "Test App"}):
            settings = Settings()
            assert settings.app_name == "Test App"
    
    @pytest.mark.unit
    def test_load_from_env_environment(self):
        """Test loading environment from environment variable."""
        with patch.dict(os.environ, {"ENVIRONMENT": "staging"}):
            settings = Settings()
            assert settings.environment == "staging"
    
    @pytest.mark.unit
    def test_load_from_env_debug(self):
        """Test loading debug flag from environment variable."""
        with patch.dict(os.environ, {"DEBUG": "false"}):
            settings = Settings()
            assert settings.debug is False
    
    @pytest.mark.unit
    def test_load_from_env_api_port(self):
        """Test loading API port from environment variable."""
        with patch.dict(os.environ, {"API_PORT": "9000"}):
            settings = Settings()
            assert settings.api_port == 9000
    
    @pytest.mark.unit
    def test_load_from_env_log_level(self):
        """Test loading log level from environment variable."""
        with patch.dict(os.environ, {"LOG_LEVEL": "DEBUG"}):
            settings = Settings()
            assert settings.log_level == "DEBUG"
    
    @pytest.mark.unit
    def test_load_from_env_database_url(self):
        """Test loading database URL from environment variable."""
        db_url = "postgresql://user:pass@localhost:5432/churn_db"
        with patch.dict(os.environ, {"DATABASE_URL": db_url}):
            settings = Settings()
            assert settings.database_url == db_url


class TestGetSettingsCaching:
    """Test get_settings caching behavior."""
    
    @pytest.mark.unit
    def test_get_settings_returns_settings_object(self):
        """Test get_settings returns Settings object."""
        settings = get_settings()
        assert isinstance(settings, Settings)
    
    @pytest.mark.unit
    def test_get_settings_is_cached(self):
        """Test get_settings returns same instance (cached)."""
        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1 is settings2
    
    @pytest.mark.unit
    def test_get_settings_has_required_attributes(self):
        """Test get_settings returns object with required attributes."""
        settings = get_settings()
        
        required_attrs = [
            "app_name", "app_version", "environment", "debug",
            "database_url", "api_host", "api_port", "log_level",
            "model_path", "data_path"
        ]
        
        for attr in required_attrs:
            assert hasattr(settings, attr), f"Missing attribute: {attr}"


class TestSettingsFeatureFlags:
    """Test feature flag configuration."""
    
    @pytest.mark.unit
    def test_default_batch_prediction_enabled(self):
        """Test batch prediction is enabled by default."""
        settings = Settings()
        assert settings.enable_batch_prediction is True
    
    @pytest.mark.unit
    def test_default_shap_explanation_enabled(self):
        """Test SHAP explanation is enabled by default."""
        settings = Settings()
        assert settings.enable_shap_explanation is True
    
    @pytest.mark.unit
    def test_default_llm_insights_enabled(self):
        """Test LLM insights is enabled by default."""
        settings = Settings()
        assert settings.enable_llm_insights is True
    
    @pytest.mark.unit
    def test_default_model_drift_detection_enabled(self):
        """Test model drift detection is enabled by default."""
        settings = Settings()
        assert settings.enable_model_drift_detection is True
    
    @pytest.mark.unit
    def test_disable_batch_prediction(self):
        """Test disabling batch prediction."""
        settings = Settings(enable_batch_prediction=False)
        assert settings.enable_batch_prediction is False
    
    @pytest.mark.unit
    def test_disable_shap_explanation(self):
        """Test disabling SHAP explanation."""
        settings = Settings(enable_shap_explanation=False)
        assert settings.enable_shap_explanation is False


class TestSettingsRateLimiting:
    """Test rate limiting configuration."""
    
    @pytest.mark.unit
    def test_default_rate_limit_per_minute(self):
        """Test default rate limit per minute."""
        settings = Settings()
        assert settings.rate_limit_per_minute == 100
    
    @pytest.mark.unit
    def test_custom_rate_limit_per_minute(self):
        """Test custom rate limit per minute."""
        settings = Settings(rate_limit_per_minute=200)
        assert settings.rate_limit_per_minute == 200


class TestSettingsLLMConfiguration:
    """Test LLM provider configuration."""
    
    @pytest.mark.unit
    def test_default_llm_provider(self):
        """Test default LLM provider."""
        settings = Settings()
        assert settings.llm_provider == "openrouter"
    
    @pytest.mark.unit
    def test_custom_llm_provider(self):
        """Test custom LLM provider."""
        settings = Settings(llm_provider="gemini")
        assert settings.llm_provider == "gemini"
    
    @pytest.mark.unit
    def test_default_openrouter_model(self):
        """Test default OpenRouter model."""
        settings = Settings()
        assert settings.openrouter_model == "meta-llama/llama-2-70b-chat"
    
    @pytest.mark.unit
    def test_custom_openrouter_model(self):
        """Test custom OpenRouter model."""
        model = "meta-llama/llama-2-13b-chat"
        settings = Settings(openrouter_model=model)
        assert settings.openrouter_model == model


class TestSettingsPathConfiguration:
    """Test path configuration."""
    
    @pytest.mark.unit
    def test_default_model_path(self):
        """Test default model path."""
        settings = Settings()
        assert settings.model_path == "./models"
    
    @pytest.mark.unit
    def test_default_data_path(self):
        """Test default data path."""
        settings = Settings()
        assert settings.data_path == "./data"
    
    @pytest.mark.unit
    def test_default_log_file_path(self):
        """Test default log file path."""
        settings = Settings()
        assert settings.log_file_path == "./logs/app.log"
    
    @pytest.mark.unit
    def test_custom_model_path(self):
        """Test custom model path."""
        settings = Settings(model_path="/custom/models")
        assert settings.model_path == "/custom/models"
    
    @pytest.mark.unit
    def test_custom_data_path(self):
        """Test custom data path."""
        settings = Settings(data_path="/custom/data")
        assert settings.data_path == "/custom/data"


class TestSettingsConnectionRetry:
    """Test connection retry configuration."""
    
    @pytest.mark.unit
    def test_default_connection_retry_count(self):
        """Test default connection retry count."""
        settings = Settings()
        assert settings.connection_retry_count == 3
    
    @pytest.mark.unit
    def test_default_connection_retry_delay(self):
        """Test default connection retry delay."""
        settings = Settings()
        assert settings.connection_retry_delay == 1
    
    @pytest.mark.unit
    def test_custom_connection_retry_count(self):
        """Test custom connection retry count."""
        settings = Settings(connection_retry_count=5)
        assert settings.connection_retry_count == 5
    
    @pytest.mark.unit
    def test_custom_connection_retry_delay(self):
        """Test custom connection retry delay."""
        settings = Settings(connection_retry_delay=2)
        assert settings.connection_retry_delay == 2


class TestSettingsPIIMasking:
    """Test PII masking configuration."""
    
    @pytest.mark.unit
    def test_default_mask_pii_in_logs(self):
        """Test PII masking is enabled by default."""
        settings = Settings()
        assert settings.mask_pii_in_logs is True
    
    @pytest.mark.unit
    def test_disable_pii_masking(self):
        """Test disabling PII masking."""
        settings = Settings(mask_pii_in_logs=False)
        assert settings.mask_pii_in_logs is False
