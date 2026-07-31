"""Prediction service - Business logic layer.

Layered Architecture: Service Layer
- Contains business logic for churn prediction
- Orchestrates model, preprocessor, and explainer
- No direct HTTP framework dependencies
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from app.models.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    CustomerInput,
    DetailedPredictionResponse,
    PredictionResponse,
)

logger = logging.getLogger(__name__)


class PredictionService:
    """Service for handling churn prediction business logic."""

    def __init__(self, model_trainer: Any):
        """Initialize prediction service with model trainer."""
        self.model_trainer = model_trainer

    def predict(self, customer: CustomerInput) -> PredictionResponse:
        """
        Predict churn probability for a customer.

        Args:
            customer: Customer input data

        Returns:
            PredictionResponse with churn probability and recommendations
        """
        logger.info(f"Processing prediction for customer with tenure={customer.tenure}")

        # Transform features
        X = self._preprocess_customer_data(customer)

        # Make prediction
        _, probability = self.model_trainer.predict(X)
        churn_probability = float(probability[0])

        # Determine risk level and recommendation
        risk_level = self._get_risk_level(churn_probability)
        risk_score = int(churn_probability * 100)
        recommendation = self._get_recommendation(
            churn_probability, customer.contract_type
        )

        return PredictionResponse(
            churn_probability=round(churn_probability, 3),
            risk_level=risk_level,
            risk_score=risk_score,
            recommendation=recommendation,
        )

    def _preprocess_customer_data(self, customer: CustomerInput) -> np.ndarray:
        """
        Preprocess customer data for prediction.

        Args:
            customer: Raw customer input

        Returns:
            Preprocessed feature array
        """
        payload = {
            "tenure": customer.tenure,
            "monthly_charges": customer.monthly_charges,
            "total_charges": customer.total_charges,
            "contract_type": customer.contract_type,
            "internet_service": customer.internet_service,
            "online_security": customer.online_security,
            "online_backup": customer.online_backup,
            "device_protection": customer.device_protection,
            "tech_support": customer.tech_support,
            "streaming_tv": customer.streaming_tv,
            "streaming_movies": customer.streaming_movies,
            "payment_method": customer.payment_method,
            "paperless_billing": customer.paperless_billing,
            "senior_citizen": customer.senior_citizen,
            "partner": customer.partner,
            "dependents": customer.dependents,
            "phone_service": customer.phone_service,
            "multiple_lines": customer.multiple_lines,
        }

        df = pd.DataFrame([payload])
        return self.model_trainer.transform_features(df)

    @staticmethod
    def _get_risk_level(probability: float) -> str:
        """
        Determine risk level from probability.

        Args:
            probability: Churn probability (0-1)

        Returns:
            Risk level string: LOW, MEDIUM, or HIGH
        """
        if probability < 0.33:
            return "LOW"
        elif probability < 0.67:
            return "MEDIUM"
        else:
            return "HIGH"

    @staticmethod
    def _get_recommendation(probability: float, contract_type: str) -> str:
        """
        Get retention recommendation based on risk and contract.

        Args:
            probability: Churn probability
            contract_type: Customer's contract type

        Returns:
            Recommendation string
        """
        if probability > 0.7:
            if contract_type == "Month-to-month":
                return "Offer contract upgrade incentive"
            else:
                return "Offer premium support package"
        elif probability > 0.4:
            return "Offer loyalty discount"
        else:
            return "Monitor customer satisfaction"

    def get_metrics(self) -> Optional[Dict[str, float]]:
        """Get model performance metrics."""
        if self.model_trainer and self.model_trainer.metrics:
            return self.model_trainer.metrics
        return None

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance from the model."""
        if self.model_trainer and self.model_trainer.model:
            return self.model_trainer.get_feature_importance()
        return None

    def get_feature_names(self) -> Optional[List[str]]:
        """Get model feature names."""
        if self.model_trainer and self.model_trainer.feature_names:
            return self.model_trainer.feature_names.tolist()
        return None

    def predict_batch(self, request: BatchPredictionRequest) -> BatchPredictionResponse:
        """
        Predict churn probability for multiple customers.

        Args:
            request: Batch prediction request with customer data

        Returns:
            BatchPredictionResponse with predictions for each customer
        """
        logger.info(
            f"Processing batch predictions for {len(request.customers)} customers"
        )

        import time

        start_time = time.time()

        predictions = []
        high_risk_count = 0

        for customer_data in request.customers:
            result = self.predict(customer_data)

            detailed = DetailedPredictionResponse(
                customer_id=customer_data.customer_id,
                churn_probability=result.churn_probability,
                churn_prediction=result.churn_probability >= 0.5,
                risk_level=result.risk_level.lower(),
                confidence=max(result.churn_probability, 1 - result.churn_probability),
                explanation=None,
            )
            predictions.append(detailed)

            if result.risk_level == "HIGH":
                high_risk_count += 1

        processing_time = (time.time() - start_time) * 1000

        return BatchPredictionResponse(
            predictions=predictions,
            total_count=len(predictions),
            high_risk_count=high_risk_count,
            processing_time_ms=round(processing_time, 2),
        )

    def is_model_loaded(self) -> bool:
        """Check if model is loaded and ready."""
        return self.model_trainer is not None and self.model_trainer.model is not None
