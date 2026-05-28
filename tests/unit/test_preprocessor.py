"""Unit tests for preprocessor."""

import pytest
import pandas as pd
import numpy as np

from app.services.preprocessor import DataPreprocessor


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return pd.DataFrame(
        {
            "age": [25, 35, 45, 55, 65],
            "tenure": [12, 24, 36, 48, 60],
            "charges": [50.0, 75.0, 100.0, 125.0, 150.0],
            "category": ["A", "B", "A", "B", "A"],
        }
    )


def test_handle_missing_values(sample_data):
    """Test missing value handling."""
    preprocessor = DataPreprocessor()

    # Add missing values
    sample_data.loc[0, "age"] = np.nan

    result = preprocessor.handle_missing_values(sample_data)

    assert result["age"].isna().sum() == 0


def test_remove_duplicates(sample_data):
    """Test duplicate removal."""
    preprocessor = DataPreprocessor()

    # Add duplicate
    duplicate = sample_data.iloc[0:1].copy()
    data_with_dup = pd.concat([sample_data, duplicate], ignore_index=True)

    result = preprocessor.remove_duplicates(data_with_dup)

    assert len(result) == len(sample_data)


def test_encode_categorical(sample_data):
    """Test categorical encoding."""
    preprocessor = DataPreprocessor()

    result = preprocessor.encode_categorical(sample_data, ["category"], fit=True)

    assert result["category"].dtype in [np.int64, np.int32]
