"""Configuration loader utilities.

This module provides utilities for loading and managing configuration
from environment variables, files, and defaults.
"""

import os
import json
import yaml
import logging
from typing import Any, Dict, Optional
from pathlib import Path

from app.utils.exceptions import ConfigurationException

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Load configuration from various sources."""

    @staticmethod
    def load_env_file(env_file: str = ".env") -> Dict[str, str]:
        """Load environment variables from .env file.
        
        Args:
            env_file: Path to .env file
            
        Returns:
            Dictionary of environment variables
        """
        try:
            env_vars = {}
            if os.path.exists(env_file):
                with open(env_file, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            if "=" in line:
                                key, value = line.split("=", 1)
                                env_vars[key.strip()] = value.strip().strip('"\'')
                logger.info(f"Loaded {len(env_vars)} environment variables from {env_file}")
            else:
                logger.warning(f"Environment file not found: {env_file}")
            return env_vars
        except Exception as e:
            logger.error(f"Error loading environment file: {str(e)}")
            raise ConfigurationException(f"Failed to load environment file: {str(e)}")

    @staticmethod
    def load_json_config(config_file: str) -> Dict[str, Any]:
        """Load configuration from JSON file.
        
        Args:
            config_file: Path to JSON config file
            
        Returns:
            Configuration dictionary
        """
        try:
            if not os.path.exists(config_file):
                raise FileNotFoundError(f"Config file not found: {config_file}")

            with open(config_file, "r") as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {config_file}")
            return config
        except Exception as e:
            logger.error(f"Error loading JSON config: {str(e)}")
            raise ConfigurationException(f"Failed to load JSON config: {str(e)}")

    @staticmethod
    def load_yaml_config(config_file: str) -> Dict[str, Any]:
        """Load configuration from YAML file.
        
        Args:
            config_file: Path to YAML config file
            
        Returns:
            Configuration dictionary
        """
        try:
            if not os.path.exists(config_file):
                raise FileNotFoundError(f"Config file not found: {config_file}")

            with open(config_file, "r") as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {config_file}")
            return config or {}
        except Exception as e:
            logger.error(f"Error loading YAML config: {str(e)}")
            raise ConfigurationException(f"Failed to load YAML config: {str(e)}")

    @staticmethod
    def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
        """Merge multiple configuration dictionaries.
        
        Args:
            *configs: Configuration dictionaries to merge
            
        Returns:
            Merged configuration
        """
        merged = {}
        for config in configs:
            if isinstance(config, dict):
                merged.update(config)
        return merged

    @staticmethod
    def get_config_value(
        key: str,
        default: Any = None,
        config: Optional[Dict[str, Any]] = None,
        env_prefix: str = ""
    ) -> Any:
        """Get configuration value from environment or config dict.
        
        Args:
            key: Configuration key
            default: Default value if not found
            config: Configuration dictionary (optional)
            env_prefix: Prefix for environment variable (e.g., "APP_")
            
        Returns:
            Configuration value
        """
        # Try environment variable first
        env_key = f"{env_prefix}{key}".upper() if env_prefix else key.upper()
        env_value = os.getenv(env_key)
        if env_value is not None:
            return env_value

        # Try config dictionary
        if config and key in config:
            return config[key]

        # Return default
        return default

    @staticmethod
    def validate_required_config(
        config: Dict[str, Any],
        required_keys: list
    ) -> None:
        """Validate that required configuration keys are present.
        
        Args:
            config: Configuration dictionary
            required_keys: List of required keys
            
        Raises:
            ConfigurationException: If required keys are missing
        """
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise ConfigurationException(
                f"Missing required configuration keys: {missing_keys}"
            )

    @staticmethod
    def create_config_from_env(env_prefix: str = "APP_") -> Dict[str, str]:
        """Create configuration dictionary from environment variables.
        
        Args:
            env_prefix: Prefix for environment variables
            
        Returns:
            Configuration dictionary
        """
        config = {}
        for key, value in os.environ.items():
            if key.startswith(env_prefix):
                config_key = key[len(env_prefix):].lower()
                config[config_key] = value
        return config

    @staticmethod
    def save_config_to_json(config: Dict[str, Any], output_file: str) -> None:
        """Save configuration to JSON file.
        
        Args:
            config: Configuration dictionary
            output_file: Output file path
        """
        try:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "w") as f:
                json.dump(config, f, indent=2)
            logger.info(f"Saved configuration to {output_file}")
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
            raise ConfigurationException(f"Failed to save configuration: {str(e)}")

    @staticmethod
    def save_config_to_yaml(config: Dict[str, Any], output_file: str) -> None:
        """Save configuration to YAML file.
        
        Args:
            config: Configuration dictionary
            output_file: Output file path
        """
        try:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "w") as f:
                yaml.dump(config, f, default_flow_style=False)
            logger.info(f"Saved configuration to {output_file}")
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
            raise ConfigurationException(f"Failed to save configuration: {str(e)}")


class EnvironmentConfig:
    """Manage environment-specific configuration."""

    ENVIRONMENTS = {"development", "staging", "production", "testing"}

    @staticmethod
    def get_environment() -> str:
        """Get current environment.
        
        Returns:
            Environment name
        """
        env = os.getenv("ENVIRONMENT", "development").lower()
        if env not in EnvironmentConfig.ENVIRONMENTS:
            logger.warning(f"Unknown environment: {env}, defaulting to development")
            return "development"
        return env

    @staticmethod
    def is_production() -> bool:
        """Check if running in production."""
        return EnvironmentConfig.get_environment() == "production"

    @staticmethod
    def is_development() -> bool:
        """Check if running in development."""
        return EnvironmentConfig.get_environment() == "development"

    @staticmethod
    def is_testing() -> bool:
        """Check if running in testing."""
        return EnvironmentConfig.get_environment() == "testing"

    @staticmethod
    def get_config_file(base_path: str = "config") -> str:
        """Get environment-specific config file path.
        
        Args:
            base_path: Base path for config files
            
        Returns:
            Path to config file
        """
        env = EnvironmentConfig.get_environment()
        return os.path.join(base_path, f"{env}.yaml")

    @staticmethod
    def load_environment_config(base_path: str = "config") -> Dict[str, Any]:
        """Load environment-specific configuration.
        
        Args:
            base_path: Base path for config files
            
        Returns:
            Configuration dictionary
        """
        config_file = EnvironmentConfig.get_config_file(base_path)
        if os.path.exists(config_file):
            return ConfigLoader.load_yaml_config(config_file)
        else:
            logger.warning(f"Environment config file not found: {config_file}")
            return {}
