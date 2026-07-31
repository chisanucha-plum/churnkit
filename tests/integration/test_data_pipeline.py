"""Integration tests for data pipeline."""

import pandas as pd
import pytest

from app.services.data_loader import DataLoader
from app.services.preprocessor import DataPreprocessor


@pytest.fixture
def sample_csv(tmp_path):
    """Create sample CSV file."""
    df = pd.DataFrame(
        {
            "customer_id": ["C001", "C002", "C003"],
            "age": [25, 35, 45],
            "tenure": [12, 24, 36],
            "charges": [50.0, 75.0, 100.0],
        }
    )

    csv_file = tmp_path / "test_data.csv"
    df.to_csv(csv_file, index=False)
    return str(csv_file)


def test_data_loading_and_preprocessing(sample_csv):
    """Test data loading and preprocessing pipeline."""
    loader = DataLoader()
    df = loader.load_csv(sample_csv)

    assert len(df) == 3
    assert "customer_id" in df.columns

    preprocessor = DataPreprocessor()
    df = preprocessor.handle_missing_values(df)

    assert df.isnull().sum().sum() == 0
