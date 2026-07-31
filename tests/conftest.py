"""Pytest configuration and fixtures for Customer Churn Prediction System.

This module provides:
- Database fixtures for testing
- Sample data fixtures
- Mock data generators
- Test utilities
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator, List

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.models.database_models import Base

# ===== DATABASE FIXTURES =====


@pytest.fixture(scope="session")
def test_db():
    """Create test database for entire test session.

    Yields:
        SQLAlchemy engine for test database
    """
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def db_session(test_db):
    """Create test database session for each test.

    Args:
        test_db: Test database engine

    Yields:
        SQLAlchemy session
    """
    TestingSessionLocal = sessionmaker(bind=test_db)
    session = TestingSessionLocal()
    yield session
    session.close()


@pytest.fixture
def temp_db_file():
    """Create temporary SQLite database file.

    Yields:
        Path to temporary database file
    """
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.remove(path)


# ===== SAMPLE DATA FIXTURES =====


@pytest.fixture
def sample_customer_data() -> Dict[str, Any]:
    """Create sample customer data for prediction.

    Returns:
        Dictionary with customer data
    """
    return {
        "customer_id": "CUST_001",
        "tenure": 24,
        "monthly_charges": 89.50,
        "total_charges": 2148.00,
        "contract_type": "Month-to-month",
        "internet_service": "Fiber optic",
        "payment_method": "Electronic check",
    }


@pytest.fixture
def sample_customer_batch() -> List[Dict[str, Any]]:
    """Create sample batch of customer data.

    Returns:
        List of customer data dictionaries
    """
    return [
        {
            "customer_id": f"CUST_{i:03d}",
            "tenure": 12 + (i * 2),
            "monthly_charges": 50.0 + (i * 5),
            "total_charges": 600.0 + (i * 100),
            "contract_type": ["Month-to-month", "One year", "Two year"][i % 3],
            "internet_service": ["Fiber optic", "DSL", "No"][i % 3],
            "payment_method": [
                "Electronic check",
                "Mailed check",
                "Bank transfer",
                "Credit card",
            ][i % 4],
        }
        for i in range(10)
    ]


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Create sample DataFrame for testing.

    Returns:
        Pandas DataFrame with sample customer data
    """
    return pd.DataFrame(
        {
            "customer_id": ["C001", "C002", "C003", "C004", "C005"],
            "tenure": [12, 24, 36, 48, 60],
            "monthly_charges": [50.0, 75.0, 100.0, 125.0, 150.0],
            "total_charges": [600.0, 1800.0, 3600.0, 6000.0, 9000.0],
            "contract_type": [
                "Month-to-month",
                "One year",
                "Two year",
                "Month-to-month",
                "One year",
            ],
            "internet_service": [
                "Fiber optic",
                "DSL",
                "Fiber optic",
                "DSL",
                "Fiber optic",
            ],
            "payment_method": [
                "Electronic check",
                "Mailed check",
                "Bank transfer",
                "Credit card",
                "Electronic check",
            ],
            "churn": [1, 0, 0, 1, 0],
        }
    )


@pytest.fixture
def sample_numeric_data() -> pd.DataFrame:
    """Create sample numeric data for testing.

    Returns:
        Pandas DataFrame with numeric features
    """
    np.random.seed(42)
    return pd.DataFrame(
        {
            "feature1": np.random.rand(100),
            "feature2": np.random.rand(100),
            "feature3": np.random.rand(100),
            "feature4": np.random.rand(100),
            "feature5": np.random.rand(100),
            "target": np.random.randint(0, 2, 100),
        }
    )


@pytest.fixture
def sample_large_dataframe() -> pd.DataFrame:
    """Create large sample DataFrame for performance testing.

    Returns:
        Pandas DataFrame with 10,000 rows
    """
    np.random.seed(42)
    n_rows = 10000
    return pd.DataFrame(
        {
            "customer_id": [f"C{i:06d}" for i in range(n_rows)],
            "tenure": np.random.randint(0, 72, n_rows),
            "monthly_charges": np.random.uniform(20, 150, n_rows),
            "total_charges": np.random.uniform(100, 10000, n_rows),
            "contract_type": np.random.choice(
                ["Month-to-month", "One year", "Two year"], n_rows
            ),
            "internet_service": np.random.choice(["Fiber optic", "DSL", "No"], n_rows),
            "payment_method": np.random.choice(
                ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
                n_rows,
            ),
            "churn": np.random.randint(0, 2, n_rows),
        }
    )


@pytest.fixture
def sample_dataframe_with_missing() -> pd.DataFrame:
    """Create sample DataFrame with missing values.

    Returns:
        Pandas DataFrame with NaN values
    """
    df = pd.DataFrame(
        {
            "customer_id": ["C001", "C002", "C003", "C004", "C005"],
            "tenure": [12, np.nan, 36, 48, 60],
            "monthly_charges": [50.0, 75.0, np.nan, 125.0, 150.0],
            "total_charges": [600.0, 1800.0, 3600.0, np.nan, 9000.0],
            "contract_type": [
                "Month-to-month",
                "One year",
                None,
                "Month-to-month",
                "One year",
            ],
            "internet_service": [
                "Fiber optic",
                "DSL",
                "Fiber optic",
                "DSL",
                "Fiber optic",
            ],
            "payment_method": [
                "Electronic check",
                "Mailed check",
                "Bank transfer",
                "Credit card",
                "Electronic check",
            ],
            "churn": [1, 0, 0, 1, 0],
        }
    )
    return df


@pytest.fixture
def sample_dataframe_with_outliers() -> pd.DataFrame:
    """Create sample DataFrame with outliers.

    Returns:
        Pandas DataFrame with outlier values
    """
    df = pd.DataFrame(
        {
            "customer_id": ["C001", "C002", "C003", "C004", "C005"],
            "tenure": [12, 24, 36, 48, 9999],  # Outlier
            "monthly_charges": [50.0, 75.0, 100.0, 125.0, 99999.0],  # Outlier
            "total_charges": [600.0, 1800.0, 3600.0, 6000.0, 9000.0],
            "contract_type": [
                "Month-to-month",
                "One year",
                "Two year",
                "Month-to-month",
                "One year",
            ],
            "internet_service": [
                "Fiber optic",
                "DSL",
                "Fiber optic",
                "DSL",
                "Fiber optic",
            ],
            "payment_method": [
                "Electronic check",
                "Mailed check",
                "Bank transfer",
                "Credit card",
                "Electronic check",
            ],
            "churn": [1, 0, 0, 1, 0],
        }
    )
    return df


# ===== MOCK DATA GENERATORS =====


@pytest.fixture
def generate_customer_data():
    """Factory fixture to generate customer data.

    Returns:
        Function to generate customer data
    """

    def _generate(customer_id: str = "CUST_001", **kwargs) -> Dict[str, Any]:
        """Generate customer data with optional overrides.

        Args:
            customer_id: Customer ID
            **kwargs: Field overrides

        Returns:
            Customer data dictionary
        """
        data = {
            "customer_id": customer_id,
            "tenure": 24,
            "monthly_charges": 89.50,
            "total_charges": 2148.00,
            "contract_type": "Month-to-month",
            "internet_service": "Fiber optic",
            "payment_method": "Electronic check",
        }
        data.update(kwargs)
        return data

    return _generate


@pytest.fixture
def generate_dataframe():
    """Factory fixture to generate DataFrames.

    Returns:
        Function to generate DataFrames
    """

    def _generate(n_rows: int = 100, seed: int = 42) -> pd.DataFrame:
        """Generate DataFrame with specified number of rows.

        Args:
            n_rows: Number of rows
            seed: Random seed

        Returns:
            Generated DataFrame
        """
        np.random.seed(seed)
        return pd.DataFrame(
            {
                "customer_id": [f"C{i:06d}" for i in range(n_rows)],
                "tenure": np.random.randint(0, 72, n_rows),
                "monthly_charges": np.random.uniform(20, 150, n_rows),
                "total_charges": np.random.uniform(100, 10000, n_rows),
                "contract_type": np.random.choice(
                    ["Month-to-month", "One year", "Two year"], n_rows
                ),
                "internet_service": np.random.choice(
                    ["Fiber optic", "DSL", "No"], n_rows
                ),
                "payment_method": np.random.choice(
                    [
                        "Electronic check",
                        "Mailed check",
                        "Bank transfer",
                        "Credit card",
                    ],
                    n_rows,
                ),
                "churn": np.random.randint(0, 2, n_rows),
            }
        )

    return _generate


# ===== CONFIGURATION FIXTURES =====


@pytest.fixture
def test_config() -> Dict[str, Any]:
    """Create test configuration.

    Returns:
        Configuration dictionary
    """
    return {
        "environment": "testing",
        "debug": True,
        "database_url": "sqlite:///:memory:",
        "api_host": "127.0.0.1",
        "api_port": 8000,
        "log_level": "DEBUG",
        "model_path": "./models",
        "data_path": "./data",
        "random_state": 42,
    }


@pytest.fixture
def test_env_vars(monkeypatch):
    """Set test environment variables.

    Args:
        monkeypatch: Pytest monkeypatch fixture
    """
    test_vars = {
        "ENVIRONMENT": "testing",
        "DEBUG": "true",
        "DATABASE_URL": "sqlite:///:memory:",
        "API_HOST": "127.0.0.1",
        "API_PORT": "8000",
        "LOG_LEVEL": "DEBUG",
    }
    for key, value in test_vars.items():
        monkeypatch.setenv(key, value)


# ===== MARKERS =====


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "slow: mark test as slow")
    config.addinivalue_line("markers", "smoke: mark test as a smoke test")
    config.addinivalue_line("markers", "performance: mark test as a performance test")
