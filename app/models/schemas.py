"""Pydantic schemas for API requests and responses.

Layered Architecture: Model Layer (DTOs)
- Input/Output DTOs separate from business logic
- Validation handled at the boundary
"""

from typing import List, Optional
import uuid

from pydantic import BaseModel, Field


# ===== Input DTOs =====

class CustomerInput(BaseModel):
    """Customer input for prediction (DTO)."""

    tenure: int = Field(..., ge=0, description="Months as customer")
    monthly_charges: float = Field(..., ge=0, description="Monthly charges")
    total_charges: float = Field(..., ge=0, description="Total charges")
    contract_type: str = Field(..., description="Contract type")
    internet_service: str = Field(..., description="Internet service type")
    online_security: int = Field(..., ge=0, le=1, description="Has online security")
    online_backup: int = Field(..., ge=0, le=1, description="Has online backup")
    device_protection: int = Field(..., ge=0, le=1, description="Has device protection")
    tech_support: int = Field(..., ge=0, le=1, description="Has tech support")
    streaming_tv: int = Field(..., ge=0, le=1, description="Has streaming TV")
    streaming_movies: int = Field(..., ge=0, le=1, description="Has streaming movies")
    payment_method: str = Field(..., description="Payment method")
    paperless_billing: int = Field(..., ge=0, le=1, description="Uses paperless billing")
    senior_citizen: int = Field(..., ge=0, le=1, description="Is senior citizen")
    partner: int = Field(..., ge=0, le=1, description="Has partner")
    dependents: int = Field(..., ge=0, le=1, description="Has dependents")
    phone_service: int = Field(..., ge=0, le=1, description="Has phone service")
    multiple_lines: int = Field(..., ge=0, le=1, description="Has multiple lines")

    model_config = {
        "json_schema_extra": {
            "example": {
                "tenure": 24,
                "monthly_charges": 89.5,
                "total_charges": 2148.0,
                "contract_type": "Month-to-month",
                "internet_service": "Fiber optic",
                "online_security": 1,
                "online_backup": 0,
                "device_protection": 1,
                "tech_support": 0,
                "streaming_tv": 1,
                "streaming_movies": 1,
                "payment_method": "Electronic check",
                "paperless_billing": 1,
                "senior_citizen": 0,
                "partner": 1,
                "dependents": 0,
                "phone_service": 1,
                "multiple_lines": 0,
            }
        }
    }


class CustomerData(CustomerInput):
    """Customer data with optional ID (for batch operations)."""

    customer_id: str = Field(
        default_factory=lambda: f"REQ-{uuid.uuid4().hex[:8]}",
        description="Unique customer identifier",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "customer_id": "CUST001",
                "tenure": 24,
                "monthly_charges": 65.5,
                "total_charges": 1572.0,
                "contract_type": "Month-to-month",
                "internet_service": "Fiber optic",
                "online_security": 1,
                "online_backup": 0,
                "device_protection": 0,
                "tech_support": 0,
                "streaming_tv": 1,
                "streaming_movies": 1,
                "payment_method": "Electronic check",
                "paperless_billing": 1,
                "senior_citizen": 0,
                "partner": 1,
                "dependents": 0,
                "phone_service": 1,
                "multiple_lines": 0,
            }
        }
    }


# ===== Output DTOs =====

class PredictionResponse(BaseModel):
    """Prediction output response (DTO)."""

    churn_probability: float = Field(..., ge=0, le=1)
    risk_level: str  # "LOW", "MEDIUM", "HIGH"
    risk_score: int = Field(..., ge=0, le=100)
    recommendation: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "churn_probability": 0.75,
                "risk_level": "HIGH",
                "risk_score": 75,
                "recommendation": "Offer contract upgrade incentive",
            }
        }
    }


class DetailedPredictionResponse(BaseModel):
    """Detailed prediction response with customer ID and explanation."""

    customer_id: str
    churn_probability: float = Field(..., ge=0, le=1)
    churn_prediction: bool
    risk_level: str  # "low", "medium", "high"
    confidence: float = Field(..., ge=0, le=1)
    explanation: Optional[str] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "customer_id": "CUST001",
                "churn_probability": 0.75,
                "churn_prediction": True,
                "risk_level": "high",
                "confidence": 0.92,
                "explanation": "High monthly charges and short tenure indicate churn risk",
            }
        }
    }


# ===== Batch Operation DTOs =====

class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""

    customers: List[CustomerData]
    include_explanation: bool = False


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""

    predictions: List[DetailedPredictionResponse]
    total_count: int
    high_risk_count: int
    processing_time_ms: float


# ===== System DTOs =====

class ModelMetrics(BaseModel):
    """Model performance metrics."""

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    database_healthy: bool
    version: str


class FeatureImportanceResponse(BaseModel):
    """Feature importance response."""

    importance: dict
    top_10: dict


class ErrorResponse(BaseModel):
    """Error response."""

    error: str
    detail: str
    status_code: int
