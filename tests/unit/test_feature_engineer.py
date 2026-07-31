"""Unit tests for feature engineer."""

import pandas as pd
import pytest

from app.services.feature_engineer import FeatureEngineer


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [10, 20, 30, 40, 50],
            "feature3": [100, 200, 300, 400, 500],
        }
    )


def test_create_interaction_features(sample_data):
    """Test interaction feature creation."""
    engineer = FeatureEngineer()

    result = engineer.create_interaction_features(sample_data, ["feature1", "feature2"])

    assert "feature1_x_feature2" in result.columns


def test_create_polynomial_features(sample_data):
    """Test polynomial feature creation."""
    engineer = FeatureEngineer()

    result = engineer.create_polynomial_features(sample_data, ["feature1"], degree=2)

    assert "feature1_pow_2" in result.columns


def test_create_binned_features(sample_data):
    """Test binned feature creation."""
    engineer = FeatureEngineer()

    result = engineer.create_binned_features(sample_data, "feature1", bins=3)

    assert "feature1_binned" in result.columns
