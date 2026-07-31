"""Data loading service."""

import logging
from typing import Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class DataLoader:
    """Load data from various sources."""

    @staticmethod
    def load_csv(file_path: str) -> pd.DataFrame:
        """Load data from CSV file."""
        try:
            logger.info(f"Loading data from {file_path}")
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
            return df
        except Exception as e:
            logger.error(f"Error loading CSV: {str(e)}")
            raise

    @staticmethod
    def load_parquet(file_path: str) -> pd.DataFrame:
        """Load data from Parquet file."""
        try:
            logger.info(f"Loading data from {file_path}")
            df = pd.read_parquet(file_path)
            logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
            return df
        except Exception as e:
            logger.error(f"Error loading Parquet: {str(e)}")
            raise

    @staticmethod
    def split_data(
        df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into train and test sets."""
        from sklearn.model_selection import train_test_split

        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state
        )
        logger.info(f"Split data: {len(train_df)} train, {len(test_df)} test samples")
        return train_df, test_df

    @staticmethod
    def validate_data(df: pd.DataFrame) -> bool:
        """Validate data integrity."""
        if df.empty:
            logger.error("DataFrame is empty")
            return False

        if df.isnull().sum().sum() > 0:
            logger.warning(f"Found {df.isnull().sum().sum()} null values")

        return True
