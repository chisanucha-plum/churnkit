"""Metrics controller."""

import logging

from app.models.schemas import ModelMetrics

logger = logging.getLogger(__name__)


class MetricsController:
    """Handle metrics requests."""

    def __init__(self, model_trainer=None):
        """Initialize controller."""
        self.model_trainer = model_trainer

    def get_model_metrics(self) -> ModelMetrics:
        """Get model performance metrics."""
        logger.info("Retrieving model metrics")

        if self.model_trainer and hasattr(self.model_trainer, "metrics"):
            metrics = self.model_trainer.metrics
        else:
            metrics = self._get_default_metrics()

        return ModelMetrics(
            accuracy=metrics.get("accuracy", 0.0),
            precision=metrics.get("precision", 0.0),
            recall=metrics.get("recall", 0.0),
            f1_score=metrics.get("f1_score", 0.0),
            auc_roc=metrics.get("auc_roc", 0.0),
            confusion_matrix=metrics.get("confusion_matrix", {}),
            feature_importance=metrics.get("feature_importance", {}),
        )

    @staticmethod
    def _get_default_metrics() -> dict:
        """Get default metrics."""
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "auc_roc": 0.0,
            "confusion_matrix": {},
            "feature_importance": {},
        }
