"""Prediction engine service."""

import logging
from typing import Dict, Any, List

import pandas as pd
import numpy as np

from app.models.ml_models import PredictionResult

logger = logging.getLogger(__name__)


class PredictionEngine:
    """Handle predictions using trained model."""

    def __init__(self, model: Any, preprocessor: Any = None, explainer: Any = None):
        """Initialize prediction engine."""
        self.model = model
        self.preprocessor = preprocessor
        self.explainer = explainer

    def predict_single(self, customer_data: Dict[str, Any]) -> PredictionResult:
        """Make prediction for single customer."""
        logger.info(f"Making prediction for customer {customer_data.get('customer_id')}")

        # Convert to DataFrame
        df = pd.DataFrame([customer_data])

        # Preprocess if available
        if self.preprocessor:
            df = self.preprocessor.preprocess(df, fit=False)

        # Make prediction
        prediction, probability = self.model.predict(df)

        # Get explanation if available
        explanation = None
        if self.explainer:
            explanation = self.explainer.generate_explanation(
                customer_data, probability[0]
            )

        result = PredictionResult(
            customer_id=customer_data.get("customer_id"),
            churn_probability=probability[0],
            churn_prediction=bool(prediction[0]),
            confidence=max(probability[0], 1 - probability[0]),
            feature_importance={},
            explanation=explanation,
        )

        return result

    def predict_batch(self, customers: List[Dict[str, Any]]) -> List[PredictionResult]:
        """Make predictions for batch of customers."""
        logger.info(f"Making batch predictions for {len(customers)} customers")

        results = []
        for customer in customers:
            result = self.predict_single(customer)
            results.append(result)

        return results

    def get_high_risk_customers(
        self, predictions: List[PredictionResult], threshold: float = 0.7
    ) -> List[PredictionResult]:
        """Filter high-risk customers."""
        return [p for p in predictions if p.churn_probability >= threshold]
