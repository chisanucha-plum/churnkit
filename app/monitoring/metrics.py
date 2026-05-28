"""Prometheus metrics."""

import logging
from fastapi import FastAPI

logger = logging.getLogger(__name__)


def setup_prometheus_metrics(app: FastAPI) -> None:
    """Setup Prometheus metrics."""
    try:
        from prometheus_client import Counter, Histogram, Gauge, generate_latest
        from prometheus_client import CONTENT_TYPE_LATEST

        # Define metrics
        request_count = Counter(
            "churn_api_requests_total",
            "Total API requests",
            ["method", "endpoint", "status"],
        )

        request_duration = Histogram(
            "churn_api_request_duration_seconds",
            "API request duration",
            ["method", "endpoint"],
        )

        predictions_total = Counter(
            "churn_predictions_total",
            "Total predictions made",
            ["risk_level"],
        )

        model_accuracy = Gauge(
            "churn_model_accuracy",
            "Current model accuracy",
        )

        @app.get("/metrics")
        async def metrics():
            """Prometheus metrics endpoint."""
            return generate_latest()

        logger.info("Prometheus metrics setup completed")

    except ImportError:
        logger.warning("prometheus_client not installed, skipping metrics setup")
