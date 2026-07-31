"""
Configuration management for the Customer Churn Prediction System.

This module loads settings from:
1. Environment variables (.env file)
2. Default values
3. Validates configuration

Usage:
    from app.config.settings import get_settings
    settings = get_settings()
    print(settings.database_url)
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    Attributes:
        app_name: Application name
        app_version: Application version
        environment: Environment (development, staging, production)
        debug: Enable debug mode
        database_url: Database connection URL
        sqlite_db_path: Path to SQLite database
        api_host: API server host
        api_port: API server port
        log_level: Logging level
    """

    # ===== APPLICATION SETTINGS =====
    app_name: str = Field(
        default="Customer Churn Prediction System", description="Application name"
    )
    app_version: str = Field(default="1.0.0", description="Application version")
    environment: str = Field(
        default="development",
        description="Environment (development, staging, production)",
    )
    debug: bool = Field(default=True, description="Enable debug mode")

    # ===== DATABASE SETTINGS =====
    database_url: Optional[str] = Field(
        default=None, description="PostgreSQL connection URL"
    )
    sqlite_db_path: str = Field(
        default="./data/churn.db", description="Path to SQLite database"
    )
    database_pool_size: int = Field(
        default=10, description="Database connection pool size"
    )
    database_max_overflow: int = Field(
        default=20, description="Database connection pool max overflow"
    )
    database_pool_timeout: int = Field(
        default=30, description="Database connection pool timeout"
    )
    connection_retry_count: int = Field(
        default=3, description="Number of retry attempts"
    )
    connection_retry_delay: int = Field(
        default=1, description="Delay between retries (seconds)"
    )

    # ===== API SETTINGS =====
    api_host: str = Field(default="0.0.0.0", description="API server host")
    api_port: int = Field(default=8000, description="API server port")
    api_workers: int = Field(default=4, description="Number of API workers")
    api_timeout: int = Field(default=30, description="API request timeout (seconds)")
    api_keys: str = Field(
        default="dev-key-12345", description="Comma-separated API keys"
    )
    rate_limit_per_minute: int = Field(
        default=100, description="Rate limit per minute per API key"
    )

    # ===== LOGGING SETTINGS =====
    log_level: str = Field(
        default="INFO", description="Logging level (DEBUG, INFO, WARNING, ERROR)"
    )
    log_format: str = Field(default="json", description="Log format (json or text)")
    log_file_path: str = Field(default="./logs/app.log", description="Path to log file")
    mask_pii_in_logs: bool = Field(default=True, description="Mask PII in logs")

    # ===== MODEL SETTINGS =====
    model_path: str = Field(default="./models", description="Path to model storage")
    model_version: str = Field(default="v1.0.0", description="Current model version")
    best_model_name: str = Field(
        default="lightgbm_model.pkl", description="Best model filename"
    )

    # ===== DATA SETTINGS =====
    data_path: str = Field(default="./data", description="Path to data directory")
    train_test_split_ratio: float = Field(
        default=0.8, description="Train/test split ratio"
    )
    random_state: int = Field(
        default=42, description="Random state for reproducibility"
    )

    # ===== LLM SETTINGS =====
    llm_provider: str = Field(
        default="openrouter", description="LLM provider (openrouter, gemini, ollama)"
    )
    openrouter_api_key: Optional[str] = Field(
        default=None, description="OpenRouter API key"
    )
    openrouter_model: str = Field(
        default="meta-llama/llama-2-70b-chat", description="OpenRouter model name"
    )

    # ===== FEATURE FLAGS =====
    enable_batch_prediction: bool = Field(
        default=True, description="Enable batch prediction"
    )
    enable_shap_explanation: bool = Field(
        default=True, description="Enable SHAP explanations"
    )
    enable_llm_insights: bool = Field(default=True, description="Enable LLM insights")
    enable_model_drift_detection: bool = Field(
        default=True, description="Enable model drift detection"
    )

    class Config:
        """Pydantic configuration."""

        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"

    @validator("environment")
    def validate_environment(cls, v: str) -> str:
        """Validate environment is valid."""
        allowed = ["development", "staging", "production"]
        if v not in allowed:
            raise ValueError(f"Environment must be one of {allowed}")
        return v

    @validator("log_level")
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is valid."""
        allowed = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in allowed:
            raise ValueError(f"Log level must be one of {allowed}")
        return v.upper()

    @validator("debug")
    def validate_debug_in_production(cls, v: bool, values) -> bool:
        """Warn if debug is enabled in production."""
        if values.get("environment") == "production" and v:
            print("⚠️  WARNING: Debug mode is enabled in production!")
        return v

    def get_database_url(self) -> str:
        """
        Get the database URL to use.

        Returns:
            PostgreSQL URL if configured, otherwise SQLite URL
        """
        if (
            self.database_url
            and self.database_url
            != "postgresql://user:password@localhost:5432/churn_db"
        ):
            return self.database_url
        else:
            return f"sqlite:///{self.sqlite_db_path}"

    def get_api_keys_list(self) -> list:
        """
        Get list of valid API keys.

        Returns:
            List of API keys
        """
        return [key.strip() for key in self.api_keys.split(",")]

    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment == "production"

    def is_development(self) -> bool:
        """Check if running in development."""
        return self.environment == "development"


@lru_cache()
def get_settings() -> Settings:
    """
    Get application settings (cached).

    Returns:
        Settings object

    Usage:
        settings = get_settings()
        print(settings.app_name)
    """
    return Settings()
