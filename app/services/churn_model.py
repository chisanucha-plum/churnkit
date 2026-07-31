"""
Production-ready Random Forest model trainer for customer churn prediction.

This module provides a complete ML pipeline including:
- Data preprocessing (encoding, scaling)
- Model training with balanced class weights
- Model evaluation and metrics
- Model persistence (save/load)
- Feature importance analysis
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

logger = logging.getLogger(__name__)


class ChurnModelTrainer:
    """
    Random Forest model trainer with balanced class weights for churn prediction.

    Features:
    - Automatic categorical encoding
    - Numerical feature scaling
    - Balanced class weights for imbalanced datasets
    - Comprehensive metrics tracking
    - Model serialization support
    """

    def __init__(self, model_path: str = "./models"):
        """
        Initialize model trainer.

        Args:
            model_path: Directory path for saving/loading models
        """
        self.model_path = Path(model_path)
        self.model_path.mkdir(parents=True, exist_ok=True)

        self.model: Optional[RandomForestClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.feature_names: Optional[list] = None
        self.metrics: Dict[str, float] = {}

        logger.info(f"ChurnModelTrainer initialized with model path: {self.model_path}")

    def preprocess_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess data: encode categorical variables and scale numerical features.

        Args:
            df: DataFrame with features and 'churn' target column

        Returns:
            Tuple of (X_scaled, y) where X_scaled is preprocessed features
        """
        df = df.copy()

        # Separate features and target
        if "churn" not in df.columns:
            raise ValueError("DataFrame must contain 'churn' column")

        y = df["churn"].values
        X = df.drop(["customer_id", "churn"], axis=1, errors="ignore")

        # Encode categorical variables
        categorical_cols = X.select_dtypes(include=["object"]).columns
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                X[col] = self.label_encoders[col].fit_transform(X[col])
            else:
                X[col] = self.label_encoders[col].transform(X[col])

        # Scale numerical features
        if self.scaler is None:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)

        self.feature_names = X.columns.tolist()

        logger.info(
            f"Data preprocessed: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features"
        )
        return X_scaled, y

    def train(
        self, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
    ) -> Dict[str, float]:
        """
        Train Random Forest model with balanced class weights.

        Args:
            df: Training DataFrame with features and target
            test_size: Proportion of data for testing (default: 0.2)
            random_state: Random seed for reproducibility (default: 42)

        Returns:
            Dictionary containing model performance metrics
        """
        logger.info("Starting model training pipeline...")

        # Preprocess data
        logger.info("Preprocessing training data...")
        X, y = self.preprocess_data(df)

        # Split data
        logger.info(
            f"Splitting data (train: {1-test_size:.0%}, test: {test_size:.0%})..."
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Build and train model
        logger.info("Training Random Forest model with balanced class weights...")
        self.model = self._build_model(y_train)
        self.model.fit(X_train, y_train)

        # Evaluate model
        logger.info("Evaluating model performance...")
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        self.metrics = self._compute_metrics(y_test, y_pred, y_pred_proba)
        self._log_metrics()

        return self.metrics

    def _build_model(self, y_train: np.ndarray) -> RandomForestClassifier:
        """
        Build Random Forest classifier with balanced class weights.

        Args:
            y_train: Training labels for computing class weights

        Returns:
            Configured RandomForestClassifier instance
        """
        # Compute balanced class weights
        class_weights = compute_class_weight(
            "balanced", classes=np.unique(y_train), y=y_train
        )
        class_weight_dict = {i: w for i, w in enumerate(class_weights)}

        logger.info(f"Computed class weights: {class_weight_dict}")

        return RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight=class_weight_dict,
            random_state=42,
            n_jobs=-1,
            verbose=0,
        )

    def _compute_metrics(
        self, y_test: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute comprehensive evaluation metrics.

        Args:
            y_test: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities

        Returns:
            Dictionary containing all metrics
        """
        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_pred_proba),
        }

    def _log_metrics(self) -> None:
        """Log model performance metrics."""
        logger.info("=" * 50)
        logger.info("Model Training Completed Successfully")
        logger.info("=" * 50)
        for metric, value in self.metrics.items():
            logger.info(f"  {metric.upper():12s}: {value:.4f}")
        logger.info("=" * 50)

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on preprocessed features.

        Args:
            X: Preprocessed feature array

        Returns:
            Tuple of (predictions, probabilities)

        Raises:
            RuntimeError: If model hasn't been trained
        """
        if self.model is None:
            raise RuntimeError("Model not trained yet. Call train() first.")

        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]

        return predictions, probabilities

    def transform_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform raw features to model-ready array.

        Args:
            df: DataFrame with raw features

        Returns:
            Preprocessed feature array ready for prediction

        Raises:
            RuntimeError: If preprocessing artifacts not loaded
        """
        if not all([self.scaler, self.label_encoders, self.feature_names]):
            raise RuntimeError(
                "Model preprocessing artifacts not loaded. "
                "Train a model or load an existing one first."
            )

        df = df.copy()

        # Encode categorical variables
        for col, encoder in self.label_encoders.items():
            if col not in df.columns:
                continue

            df[col] = df[col].astype(str)

            # Handle unseen categories by using most common class
            unseen_mask = ~df[col].isin(encoder.classes_)
            if unseen_mask.any():
                logger.warning(
                    f"Unseen categories in column '{col}'. "
                    f"Using default value: {encoder.classes_[0]}"
                )
                df.loc[unseen_mask, col] = encoder.classes_[0]

            df[col] = encoder.transform(df[col])

        # Ensure feature order matches training
        return self.scaler.transform(df[self.feature_names])

    def save(self, filename: str = "churn_model.pkl") -> None:
        """
        Save model and preprocessing artifacts to disk.

        Args:
            filename: Name of file to save model

        Raises:
            RuntimeError: If model hasn't been trained
        """
        if self.model is None:
            raise RuntimeError("No model to save. Train a model first.")

        filepath = self.model_path / filename

        try:
            model_data = {
                "model": self.model,
                "scaler": self.scaler,
                "label_encoders": self.label_encoders,
                "feature_names": self.feature_names,
                "metrics": self.metrics,
                "version": "1.0.0",
            }

            with open(filepath, "wb") as f:
                pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)

            logger.info(f"Model saved successfully to: {filepath}")
            logger.info(f"Model size: {filepath.stat().st_size / 1024:.2f} KB")
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            raise

    def load(self, filename: str = "churn_model.pkl") -> None:
        """
        Load model and preprocessing artifacts from disk.

        Args:
            filename: Name of file to load model from

        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If model file is corrupted
        """
        filepath = self.model_path / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        try:
            with open(filepath, "rb") as f:
                model_data = pickle.load(f)

            # Validate loaded data
            required_keys = [
                "model",
                "scaler",
                "label_encoders",
                "feature_names",
                "metrics",
            ]
            missing_keys = [key for key in required_keys if key not in model_data]

            if missing_keys:
                raise RuntimeError(
                    f"Corrupted model file. Missing keys: {missing_keys}"
                )

            self.model = model_data["model"]
            self.scaler = model_data["scaler"]
            self.label_encoders = model_data["label_encoders"]
            self.feature_names = model_data["feature_names"]
            self.metrics = model_data["metrics"]

            version = model_data.get("version", "unknown")

            logger.info(f"Model loaded successfully from: {filepath}")
            logger.info(f"Model version: {version}")
            logger.info(f"Features: {len(self.feature_names)}")

        except pickle.UnpicklingError as e:
            logger.error(f"Failed to unpickle model file: {str(e)}")
            raise RuntimeError(f"Corrupted model file: {filepath}")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

    def get_feature_importance(self, top_n: Optional[int] = None) -> Dict[str, float]:
        """
        Get feature importance scores sorted by value.

        Args:
            top_n: If specified, return only top N features

        Returns:
            Dictionary mapping feature names to importance scores

        Raises:
            RuntimeError: If model hasn't been trained
        """
        if self.model is None:
            raise RuntimeError("Model not trained yet. Call train() first.")

        importance = dict(zip(self.feature_names, self.model.feature_importances_))
        sorted_importance = dict(
            sorted(importance.items(), key=lambda x: x[1], reverse=True)
        )

        if top_n is not None:
            sorted_importance = dict(list(sorted_importance.items())[:top_n])

        return sorted_importance

    def get_model_info(self) -> Dict[str, any]:
        """
        Get comprehensive model information.

        Returns:
            Dictionary containing model configuration and performance
        """
        if self.model is None:
            return {"status": "not_trained"}

        return {
            "status": "trained",
            "model_type": "RandomForestClassifier",
            "n_estimators": self.model.n_estimators,
            "max_depth": self.model.max_depth,
            "n_features": len(self.feature_names) if self.feature_names else 0,
            "feature_names": self.feature_names,
            "metrics": self.metrics,
            "class_weights": (
                dict(self.model.class_weight)
                if hasattr(self.model, "class_weight")
                else None
            ),
        }
