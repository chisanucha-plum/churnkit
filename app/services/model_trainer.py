"""Model training service."""

import logging
from typing import Dict, Any, Tuple

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Handle model training and evaluation."""

    def __init__(self, model_type: str = "random_forest", **kwargs):
        """Initialize trainer with model type."""
        self.model_type = model_type
        self.model = self._create_model(**kwargs)
        self.metrics = {}

    def _create_model(self, **kwargs):
        """Create model based on type."""
        logger.info(f"Creating {self.model_type} model")

        if self.model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=kwargs.get("n_estimators", 100),
                max_depth=kwargs.get("max_depth", 10),
                random_state=42,
            )
        elif self.model_type == "gradient_boosting":
            return GradientBoostingClassifier(
                n_estimators=kwargs.get("n_estimators", 100),
                learning_rate=kwargs.get("learning_rate", 0.1),
                random_state=42,
            )
        elif self.model_type == "logistic_regression":
            return LogisticRegression(random_state=42, max_iter=1000)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train the model."""
        logger.info(f"Training {self.model_type} model")
        self.model.fit(X_train, y_train)
        logger.info("Model training completed")

    def evaluate(
        self, X_test: pd.DataFrame, y_test: pd.Series
    ) -> Dict[str, float]:
        """Evaluate model performance."""
        logger.info("Evaluating model")

        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        self.metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "auc_roc": roc_auc_score(y_test, y_pred_proba),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }

        logger.info(f"Model metrics: {self.metrics}")
        return self.metrics

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions."""
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]
        return predictions, probabilities

    def get_feature_importance(self, feature_names: list) -> Dict[str, float]:
        """Get feature importance."""
        if not hasattr(self.model, "feature_importances_"):
            logger.warning("Model does not have feature_importances_")
            return {}

        importances = self.model.feature_importances_
        feature_importance = dict(zip(feature_names, importances))
        return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
