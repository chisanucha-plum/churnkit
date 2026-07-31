"""
Sample customer data generator for testing and demonstration.

This module creates realistic synthetic customer data that mimics
real customer churn patterns for development and testing purposes.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def generate_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    Generate sample customer data.

    Args:
        n_samples: Number of samples to generate

    Returns:
        DataFrame with customer data
    """
    np.random.seed(42)

    data = {
        "customer_id": [f"CUST_{i:05d}" for i in range(n_samples)],
        "tenure": np.random.randint(1, 73, n_samples),
        "monthly_charges": np.random.uniform(20, 120, n_samples),
        "total_charges": np.random.uniform(100, 8000, n_samples),
        "contract_type": np.random.choice(
            ["Month-to-month", "One year", "Two year"], n_samples, p=[0.55, 0.25, 0.20]
        ),
        "internet_service": np.random.choice(
            ["Fiber optic", "DSL", "No"], n_samples, p=[0.40, 0.35, 0.25]
        ),
        "online_security": np.random.choice([0, 1], n_samples, p=[0.70, 0.30]),
        "online_backup": np.random.choice([0, 1], n_samples, p=[0.75, 0.25]),
        "device_protection": np.random.choice([0, 1], n_samples, p=[0.80, 0.20]),
        "tech_support": np.random.choice([0, 1], n_samples, p=[0.75, 0.25]),
        "streaming_tv": np.random.choice([0, 1], n_samples, p=[0.60, 0.40]),
        "streaming_movies": np.random.choice([0, 1], n_samples, p=[0.60, 0.40]),
        "payment_method": np.random.choice(
            ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
            n_samples,
        ),
        "paperless_billing": np.random.choice([0, 1], n_samples, p=[0.40, 0.60]),
        "senior_citizen": np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
        "partner": np.random.choice([0, 1], n_samples, p=[0.50, 0.50]),
        "dependents": np.random.choice([0, 1], n_samples, p=[0.70, 0.30]),
        "phone_service": np.random.choice([0, 1], n_samples, p=[0.10, 0.90]),
        "multiple_lines": np.random.choice([0, 1], n_samples, p=[0.60, 0.40]),
    }

    df = pd.DataFrame(data)

    # Create churn target (with realistic patterns)
    churn_prob = np.zeros(n_samples)

    # Month-to-month contracts have higher churn
    month_to_month_mask = df["contract_type"] == "Month-to-month"
    churn_prob[month_to_month_mask] += 0.40

    # Longer tenure = lower churn
    churn_prob -= (df["tenure"] / 100) * 0.30

    # Higher charges = higher churn
    churn_prob += (df["monthly_charges"] / 120) * 0.20

    # Electronic check = higher churn
    electronic_check_mask = df["payment_method"] == "Electronic check"
    churn_prob[electronic_check_mask] += 0.15

    # Clamp probabilities
    churn_prob = np.clip(churn_prob, 0, 1)

    # Generate churn based on probabilities
    df["churn"] = (np.random.random(n_samples) < churn_prob).astype(int)

    return df


def save_sample_data(
    df: pd.DataFrame, filepath: str = "./data/sample_customers.csv"
) -> None:
    """
    Save sample data to CSV file.

    Args:
        df: DataFrame to save
        filepath: Path to save file
    """
    try:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath, index=False)
        logger.info(f"Sample data saved successfully to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save sample data: {str(e)}")
        raise


def load_sample_data(filepath: str = "./data/sample_customers.csv") -> pd.DataFrame:
    """
    Load sample data from CSV file.

    Args:
        filepath: Path to data file

    Returns:
        DataFrame with customer data
    """
    try:
        if not Path(filepath).exists():
            logger.warning(f"File not found: {filepath}. Generating new sample data...")
            df = generate_sample_data()
            save_sample_data(df, filepath)
            return df

        df = pd.read_csv(filepath)
        logger.info(f"Sample data loaded successfully from {filepath}")
        return df
    except Exception as e:
        logger.error(f"Failed to load sample data: {str(e)}")
        raise


if __name__ == "__main__":
    # Configure logging for standalone execution
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Generate and save sample data
    df = generate_sample_data(n_samples=1000)
    save_sample_data(df)

    logger.info(f"Generated {len(df)} sample records")
    logger.info(f"Churn rate: {df['churn'].mean():.1%}")
    logger.info(f"\nFirst 5 records:\n{df.head()}")
