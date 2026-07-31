"""Feature engineering service."""

import logging
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Handle feature engineering and creation."""

    @staticmethod
    def create_interaction_features(
        df: pd.DataFrame, features: List[str]
    ) -> pd.DataFrame:
        """Create interaction features."""
        logger.info(f"Creating interaction features from {len(features)} features")

        for i, feat1 in enumerate(features):
            for feat2 in features[i + 1 :]:
                df[f"{feat1}_x_{feat2}"] = df[feat1] * df[feat2]

        return df

    @staticmethod
    def create_polynomial_features(
        df: pd.DataFrame, features: List[str], degree: int = 2
    ) -> pd.DataFrame:
        """Create polynomial features."""
        logger.info(f"Creating polynomial features with degree {degree}")

        for feat in features:
            for d in range(2, degree + 1):
                df[f"{feat}_pow_{d}"] = df[feat] ** d

        return df

    @staticmethod
    def create_ratio_features(
        df: pd.DataFrame, numerator: str, denominator: str, name: str
    ) -> pd.DataFrame:
        """Create ratio features."""
        logger.info(f"Creating ratio feature: {name}")

        df[name] = df[numerator] / (df[denominator] + 1e-8)
        return df

    @staticmethod
    def create_binned_features(
        df: pd.DataFrame, column: str, bins: int = 5, name: str = None
    ) -> pd.DataFrame:
        """Create binned features."""
        if name is None:
            name = f"{column}_binned"

        logger.info(f"Creating binned feature: {name} with {bins} bins")
        df[name] = pd.cut(df[column], bins=bins, labels=False)
        return df

    @staticmethod
    def create_temporal_features(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
        """Create temporal features from date column."""
        logger.info(f"Creating temporal features from {date_column}")

        df[date_column] = pd.to_datetime(df[date_column])
        df[f"{date_column}_year"] = df[date_column].dt.year
        df[f"{date_column}_month"] = df[date_column].dt.month
        df[f"{date_column}_day"] = df[date_column].dt.day
        df[f"{date_column}_dayofweek"] = df[date_column].dt.dayofweek

        return df

    @staticmethod
    def select_features(
        df: pd.DataFrame, method: str = "variance", threshold: float = 0.01
    ) -> pd.DataFrame:
        """Select features based on variance or other criteria."""
        logger.info(f"Selecting features using {method} method")

        if method == "variance":
            variances = df.var()
            selected_features = variances[variances > threshold].index.tolist()
            df = df[selected_features]

        return df

    @staticmethod
    def handle_class_imbalance(
        df: pd.DataFrame, target_col: str, method: str = "oversample"
    ) -> pd.DataFrame:
        """Handle class imbalance."""
        logger.info(f"Handling class imbalance using {method} method")

        if method == "oversample":
            from sklearn.utils import resample

            minority_class = df[df[target_col] == 1]
            majority_class = df[df[target_col] == 0]

            minority_upsampled = resample(
                minority_class,
                replace=True,
                n_samples=len(majority_class),
                random_state=42,
            )

            df = pd.concat([majority_class, minority_upsampled])

        return df
