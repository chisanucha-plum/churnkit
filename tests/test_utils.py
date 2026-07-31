"""Test utilities and helpers for Customer Churn Prediction System tests."""

import json
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


class TestDataGenerator:
    """Generate test data for various scenarios."""

    @staticmethod
    def create_valid_customer_data(
        customer_id: str = "CUST_001",
        tenure: int = 24,
        monthly_charges: float = 89.50,
        total_charges: float = 2148.00,
        contract_type: str = "Month-to-month",
        internet_service: str = "Fiber optic",
        payment_method: str = "Electronic check",
    ) -> Dict[str, Any]:
        """Create valid customer data for testing.

        Args:
            customer_id: Customer ID
            tenure: Tenure in months
            monthly_charges: Monthly charges
            total_charges: Total charges
            contract_type: Contract type
            internet_service: Internet service type
            payment_method: Payment method

        Returns:
            Valid customer data dictionary
        """
        return {
            "customer_id": customer_id,
            "tenure": tenure,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "contract_type": contract_type,
            "internet_service": internet_service,
            "payment_method": payment_method,
        }

    @staticmethod
    def create_invalid_customer_data(
        missing_field: str = None,
        invalid_field: str = None,
        invalid_value: Any = None,
    ) -> Dict[str, Any]:
        """Create invalid customer data for testing.

        Args:
            missing_field: Field to omit
            invalid_field: Field to make invalid
            invalid_value: Invalid value for the field

        Returns:
            Invalid customer data dictionary
        """
        data = TestDataGenerator.create_valid_customer_data()

        if missing_field:
            del data[missing_field]

        if invalid_field:
            data[invalid_field] = invalid_value

        return data

    @staticmethod
    def create_customer_batch(
        n_customers: int = 10,
        seed: int = 42,
    ) -> List[Dict[str, Any]]:
        """Create batch of customer data.

        Args:
            n_customers: Number of customers
            seed: Random seed

        Returns:
            List of customer data dictionaries
        """
        np.random.seed(seed)
        customers = []

        contract_types = ["Month-to-month", "One year", "Two year"]
        internet_services = ["Fiber optic", "DSL", "No"]
        payment_methods = [
            "Electronic check",
            "Mailed check",
            "Bank transfer",
            "Credit card",
        ]

        for i in range(n_customers):
            customers.append(
                {
                    "customer_id": f"CUST_{i:06d}",
                    "tenure": np.random.randint(0, 72),
                    "monthly_charges": np.random.uniform(20, 150),
                    "total_charges": np.random.uniform(100, 10000),
                    "contract_type": np.random.choice(contract_types),
                    "internet_service": np.random.choice(internet_services),
                    "payment_method": np.random.choice(payment_methods),
                }
            )

        return customers

    @staticmethod
    def create_dataframe(
        n_rows: int = 100,
        seed: int = 42,
        include_target: bool = True,
    ) -> pd.DataFrame:
        """Create test DataFrame.

        Args:
            n_rows: Number of rows
            seed: Random seed
            include_target: Whether to include target column

        Returns:
            Test DataFrame
        """
        np.random.seed(seed)

        contract_types = ["Month-to-month", "One year", "Two year"]
        internet_services = ["Fiber optic", "DSL", "No"]
        payment_methods = [
            "Electronic check",
            "Mailed check",
            "Bank transfer",
            "Credit card",
        ]

        data = {
            "customer_id": [f"C{i:06d}" for i in range(n_rows)],
            "tenure": np.random.randint(0, 72, n_rows),
            "monthly_charges": np.random.uniform(20, 150, n_rows),
            "total_charges": np.random.uniform(100, 10000, n_rows),
            "contract_type": np.random.choice(contract_types, n_rows),
            "internet_service": np.random.choice(internet_services, n_rows),
            "payment_method": np.random.choice(payment_methods, n_rows),
        }

        if include_target:
            data["churn"] = np.random.randint(0, 2, n_rows)

        return pd.DataFrame(data)


class TestAssertions:
    """Custom assertions for testing."""

    @staticmethod
    def assert_valid_prediction_response(response: Dict[str, Any]) -> None:
        """Assert prediction response has required fields.

        Args:
            response: Prediction response dictionary

        Raises:
            AssertionError: If response is invalid
        """
        required_fields = [
            "customer_id",
            "churn_probability",
            "risk_level",
            "risk_score",
            "top_churn_factors",
            "recommendation",
            "confidence",
            "model_version",
            "timestamp",
        ]

        for field in required_fields:
            assert field in response, f"Missing field: {field}"

        # Validate types
        assert isinstance(response["customer_id"], str)
        assert isinstance(response["churn_probability"], (int, float))
        assert 0 <= response["churn_probability"] <= 1
        assert response["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
        assert isinstance(response["risk_score"], (int, float))
        assert 0 <= response["risk_score"] <= 100
        assert isinstance(response["top_churn_factors"], list)
        assert isinstance(response["recommendation"], str)
        assert isinstance(response["confidence"], (int, float))
        assert 0 <= response["confidence"] <= 1

    @staticmethod
    def assert_valid_batch_prediction_response(response: Dict[str, Any]) -> None:
        """Assert batch prediction response has required fields.

        Args:
            response: Batch prediction response dictionary

        Raises:
            AssertionError: If response is invalid
        """
        required_fields = ["predictions", "processing_time_ms", "batch_size"]

        for field in required_fields:
            assert field in response, f"Missing field: {field}"

        assert isinstance(response["predictions"], list)
        assert isinstance(response["processing_time_ms"], (int, float))
        assert isinstance(response["batch_size"], int)
        assert response["batch_size"] == len(response["predictions"])

    @staticmethod
    def assert_valid_metrics_response(response: Dict[str, Any]) -> None:
        """Assert metrics response has required fields.

        Args:
            response: Metrics response dictionary

        Raises:
            AssertionError: If response is invalid
        """
        required_sections = ["model_performance", "dataset_stats", "model_info"]

        for section in required_sections:
            assert section in response, f"Missing section: {section}"

        # Validate model_performance
        perf = response["model_performance"]
        perf_fields = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
        for field in perf_fields:
            assert field in perf, f"Missing performance field: {field}"
            assert 0 <= perf[field] <= 1

    @staticmethod
    def assert_dataframe_valid(
        df: pd.DataFrame, expected_shape: Tuple[int, int] = None
    ) -> None:
        """Assert DataFrame is valid.

        Args:
            df: DataFrame to validate
            expected_shape: Expected shape (rows, cols)

        Raises:
            AssertionError: If DataFrame is invalid
        """
        assert isinstance(df, pd.DataFrame)
        assert not df.empty

        if expected_shape:
            assert (
                df.shape == expected_shape
            ), f"Expected shape {expected_shape}, got {df.shape}"

    @staticmethod
    def assert_no_nan_values(df: pd.DataFrame, columns: List[str] = None) -> None:
        """Assert DataFrame has no NaN values.

        Args:
            df: DataFrame to check
            columns: Specific columns to check (None = all)

        Raises:
            AssertionError: If NaN values found
        """
        if columns:
            df_to_check = df[columns]
        else:
            df_to_check = df

        nan_count = df_to_check.isna().sum().sum()
        assert nan_count == 0, f"Found {nan_count} NaN values"

    @staticmethod
    def assert_numeric_range(
        values: np.ndarray,
        min_val: float = None,
        max_val: float = None,
    ) -> None:
        """Assert numeric values are within range.

        Args:
            values: Values to check
            min_val: Minimum value
            max_val: Maximum value

        Raises:
            AssertionError: If values out of range
        """
        if min_val is not None:
            assert np.all(values >= min_val), f"Values below minimum {min_val}"

        if max_val is not None:
            assert np.all(values <= max_val), f"Values above maximum {max_val}"


class TestComparison:
    """Utilities for comparing test results."""

    @staticmethod
    def compare_predictions(
        pred1: Dict[str, Any],
        pred2: Dict[str, Any],
        tolerance: float = 0.01,
    ) -> bool:
        """Compare two predictions for similarity.

        Args:
            pred1: First prediction
            pred2: Second prediction
            tolerance: Tolerance for numeric comparison

        Returns:
            True if predictions are similar
        """
        # Compare customer_id
        if pred1.get("customer_id") != pred2.get("customer_id"):
            return False

        # Compare churn_probability with tolerance
        prob_diff = abs(
            pred1.get("churn_probability", 0) - pred2.get("churn_probability", 0)
        )
        if prob_diff > tolerance:
            return False

        # Compare risk_level
        if pred1.get("risk_level") != pred2.get("risk_level"):
            return False

        return True

    @staticmethod
    def compare_dataframes(
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        tolerance: float = 0.01,
    ) -> Tuple[bool, str]:
        """Compare two DataFrames for similarity.

        Args:
            df1: First DataFrame
            df2: Second DataFrame
            tolerance: Tolerance for numeric comparison

        Returns:
            Tuple of (is_similar, message)
        """
        # Check shape
        if df1.shape != df2.shape:
            return False, f"Shape mismatch: {df1.shape} vs {df2.shape}"

        # Check columns
        if not df1.columns.equals(df2.columns):
            return False, f"Columns mismatch: {df1.columns} vs {df2.columns}"

        # Check numeric columns
        numeric_cols = df1.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            max_diff = np.max(np.abs(df1[col] - df2[col]))
            if max_diff > tolerance:
                return False, f"Column {col} differs by {max_diff}"

        # Check non-numeric columns
        non_numeric_cols = df1.select_dtypes(exclude=[np.number]).columns
        for col in non_numeric_cols:
            if not df1[col].equals(df2[col]):
                return False, f"Column {col} values differ"

        return True, "DataFrames are similar"


class MockResponse:
    """Mock HTTP response for testing."""

    def __init__(self, json_data: Dict[str, Any], status_code: int = 200):
        """Initialize mock response.

        Args:
            json_data: Response JSON data
            status_code: HTTP status code
        """
        self.json_data = json_data
        self.status_code = status_code

    def json(self) -> Dict[str, Any]:
        """Get JSON data."""
        return self.json_data

    def raise_for_status(self):
        """Raise exception for error status codes."""
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")
