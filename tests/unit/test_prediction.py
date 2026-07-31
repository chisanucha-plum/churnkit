"""Unit tests for prediction engine."""

from unittest.mock import Mock

import pytest

from app.services.prediction import PredictionEngine


@pytest.fixture
def mock_model():
    """Create mock model."""
    model = Mock()
    model.predict = Mock(return_value=([1], [0.8]))
    return model


def test_predict_single(mock_model):
    """Test single prediction."""
    engine = PredictionEngine(model=mock_model)

    customer_data = {
        "customer_id": "CUST001",
        "age": 35,
        "tenure_months": 24,
        "monthly_charges": 65.5,
        "total_charges": 1572.0,
        "contract_type": "Month-to-month",
        "internet_service": "Fiber optic",
        "payment_method": "Electronic check",
    }

    result = engine.predict_single(customer_data)

    assert result.customer_id == "CUST001"
    assert result.churn_probability == 0.8
