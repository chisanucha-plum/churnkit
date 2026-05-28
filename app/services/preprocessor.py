"""Data preprocessing service."""

import logging
from typing import Optional

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handle data preprocessing and cleaning."""

    def __init__(self):
        """Initialize preprocessor."""
        self.scaler = StandardScaler()
        self.label_encoders = {}

    def handle_missing_values(
        self, df: pd.DataFrame, strategy: str = "mean"
    ) -> pd.DataFrame:
        """Handle missing values."""
        logger.info(f"Handling missing values with strategy: {strategy}")

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=["object"]).columns

        if strategy == "mean":
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        elif strategy == "median":
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        elif strategy == "drop":
            df = df.dropna()

        df[categorical_cols] = df[categorical_cols].fillna("Unknown")

        return df

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows."""
        initial_rows = len(df)
        df = df.drop_duplicates()
        removed = initial_rows - len(df)
        logger.info(f"Removed {removed} duplicate rows")
        return df

    def remove_outliers(
        self, df: pd.DataFrame, columns: list, method: str = "iqr"
    ) -> pd.DataFrame:
        """Remove outliers from numeric columns."""
        logger.info(f"Removing outliers using {method} method")

        if method == "iqr":
            for col in columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

        return df

    def encode_categorical(
        self, df: pd.DataFrame, categorical_cols: list, fit: bool = True
    ) -> pd.DataFrame:
        """Encode categorical variables."""
        logger.info(f"Encoding {len(categorical_cols)} categorical columns")

        for col in categorical_cols:
            if fit:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                if col in self.label_encoders:
                    df[col] = self.label_encoders[col].transform(df[col].astype(str))

        return df

    def scale_features(
        self, df: pd.DataFrame, numeric_cols: list, fit: bool = True
    ) -> pd.DataFrame:
        """Scale numeric features."""
        logger.info(f"Scaling {len(numeric_cols)} numeric columns")

        if fit:
            df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        else:
            df[numeric_cols] = self.scaler.transform(df[numeric_cols])

        return df

    def preprocess(
        self,
        df: pd.DataFrame,
        numeric_cols: list,
        categorical_cols: list,
        fit: bool = True,
    ) -> pd.DataFrame:
        """Complete preprocessing pipeline."""
        logger.info("Starting preprocessing pipeline")

        df = self.handle_missing_values(df)
        df = self.remove_duplicates(df)
        df = self.remove_outliers(df, numeric_cols)
        df = self.encode_categorical(df, categorical_cols, fit=fit)
        df = self.scale_features(df, numeric_cols, fit=fit)

        logger.info("Preprocessing completed")
        return df
