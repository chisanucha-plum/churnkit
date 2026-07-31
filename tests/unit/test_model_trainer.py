"""Unit tests for model trainer."""

import numpy as np
import pandas as pd
import pytest

from app.services.model_trainer import ModelTrainer


@pytest.fixture
def sample_data():
    """Create sample training data."""
    X = pd.DataFrame(
        {
            "feature1": np.random.rand(100),
            "feature2": np.random.rand(100),
            "feature3": np.random.rand(100),
        }
    )
    y = pd.Series(np.random.randint(0, 2, 100))
    return X, y


def test_model_creation():
    """Test model creation."""
    trainer = ModelTrainer(model_type="random_forest")
    assert trainer.model is not None


def test_model_training(sample_data):
    """Test model training."""
    X, y = sample_data
    trainer = ModelTrainer(model_type="random_forest")

    trainer.train(X, y)

    assert trainer.model is not None


def test_model_prediction(sample_data):
    """Test model prediction."""
    X, y = sample_data
    trainer = ModelTrainer(model_type="random_forest")

    trainer.train(X, y)
    predictions, probabilities = trainer.predict(X)

    assert len(predictions) == len(X)
    assert len(probabilities) == len(X)
