"""ML model definitions and utilities."""

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np


@dataclass
class ModelConfig:
    """ML model configuration."""

    model_type: str
    hyperparameters: Dict[str, Any]
    feature_names: List[str]
    target_name: str = "churn"
    test_size: float = 0.2
    random_state: int = 42


@dataclass
class PredictionResult:
    """Prediction result."""

    customer_id: str
    churn_probability: float
    churn_prediction: bool
    confidence: float
    feature_importance: Dict[str, float]
    shap_values: np.ndarray = None
    explanation: str = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "customer_id": self.customer_id,
            "churn_probability": float(self.churn_probability),
            "churn_prediction": bool(self.churn_prediction),
            "confidence": float(self.confidence),
            "feature_importance": self.feature_importance,
            "explanation": self.explanation,
        }


class ModelMetadata:
    """Model metadata and versioning."""

    def __init__(
        self,
        version: str,
        model_type: str,
        features: List[str],
        metrics: Dict[str, float],
    ):
        """Initialize model metadata."""
        self.version = version
        self.model_type = model_type
        self.features = features
        self.metrics = metrics

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version": self.version,
            "model_type": self.model_type,
            "features": self.features,
            "metrics": self.metrics,
        }
