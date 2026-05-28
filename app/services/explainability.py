"""Model explainability service using SHAP."""

import logging
from typing import Dict, Any

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class ExplainabilityService:
    """Handle model explainability and interpretability."""

    def __init__(self, model: Any, X_background: pd.DataFrame = None):
        """Initialize explainability service."""
        self.model = model
        self.X_background = X_background
        self.explainer = None

    def setup_shap_explainer(self) -> None:
        """Setup SHAP explainer."""
        try:
            import shap

            logger.info("Setting up SHAP explainer")
            self.explainer = shap.TreeExplainer(self.model)
        except ImportError:
            logger.warning("SHAP not installed, skipping SHAP setup")

    def get_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """Get SHAP values for predictions."""
        if self.explainer is None:
            self.setup_shap_explainer()

        if self.explainer is None:
            logger.warning("SHAP explainer not available")
            return None

        logger.info("Computing SHAP values")
        shap_values = self.explainer.shap_values(X)
        return shap_values

    def get_feature_importance(self, X: pd.DataFrame) -> Dict[str, float]:
        """Get feature importance from SHAP values."""
        shap_values = self.get_shap_values(X)

        if shap_values is None:
            return {}

        # For binary classification, take the positive class SHAP values
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        feature_importance = dict(zip(X.columns, mean_abs_shap))

        return dict(
            sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        )

    def generate_explanation(
        self, customer_data: Dict[str, Any], prediction: float, top_features: int = 5
    ) -> str:
        """Generate human-readable explanation."""
        logger.info("Generating explanation")

        explanation = f"Churn probability: {prediction:.2%}. "

        if prediction > 0.7:
            explanation += "High risk customer. "
        elif prediction > 0.4:
            explanation += "Medium risk customer. "
        else:
            explanation += "Low risk customer. "

        return explanation

    def get_local_explanation(
        self, customer_data: Dict[str, Any], prediction: float
    ) -> Dict[str, Any]:
        """Get local explanation for a single prediction."""
        logger.info("Getting local explanation")

        return {
            "prediction": float(prediction),
            "explanation": self.generate_explanation(customer_data, prediction),
            "contributing_factors": {},
        }
