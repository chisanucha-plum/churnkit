"""Prediction controller."""

import logging
from typing import List

from app.models.schemas import (
    CustomerData,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
)

logger = logging.getLogger(__name__)


class PredictionController:
    """Handle prediction requests."""

    def __init__(self, prediction_engine):
        """Initialize controller."""
        self.prediction_engine = prediction_engine

    def predict_single(self, customer: CustomerData) -> PredictionResponse:
        """Handle single prediction."""
        logger.info(f"Processing prediction for {customer.customer_id}")

        result = self.prediction_engine.predict_single(customer.dict())

        return PredictionResponse(
            customer_id=result.customer_id,
            churn_probability=result.churn_probability,
            churn_prediction=result.churn_prediction,
            risk_level=self._get_risk_level(result.churn_probability),
            confidence=result.confidence,
            explanation=result.explanation,
        )

    def predict_batch(
        self, request: BatchPredictionRequest
    ) -> BatchPredictionResponse:
        """Handle batch predictions."""
        logger.info(f"Processing batch predictions for {len(request.customers)} customers")

        customers_data = [c.dict() for c in request.customers]
        results = self.prediction_engine.predict_batch(customers_data)

        predictions = [
            PredictionResponse(
                customer_id=r.customer_id,
                churn_probability=r.churn_probability,
                churn_prediction=r.churn_prediction,
                risk_level=self._get_risk_level(r.churn_probability),
                confidence=r.confidence,
                explanation=r.explanation if request.include_explanation else None,
            )
            for r in results
        ]

        high_risk = sum(1 for p in predictions if p.risk_level == "high")

        return BatchPredictionResponse(
            predictions=predictions,
            total_count=len(predictions),
            high_risk_count=high_risk,
            processing_time_ms=0.0,
        )

    @staticmethod
    def _get_risk_level(probability: float) -> str:
        """Get risk level from probability."""
        if probability >= 0.7:
            return "high"
        elif probability >= 0.4:
            return "medium"
        else:
            return "low"
