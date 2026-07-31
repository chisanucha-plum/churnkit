"""Data access layer repositories."""

import logging
from typing import List, Optional

from sqlalchemy.orm import Session

from app.models.database_models import Customer, ModelVersion, Prediction

logger = logging.getLogger(__name__)


class CustomerRepository:
    """Customer data access."""

    def __init__(self, db: Session):
        """Initialize repository."""
        self.db = db

    def create(self, customer_data: dict) -> Customer:
        """Create new customer."""
        customer = Customer(**customer_data)
        self.db.add(customer)
        self.db.commit()
        self.db.refresh(customer)
        logger.info(f"Created customer: {customer.customer_id}")
        return customer

    def get_by_id(self, customer_id: str) -> Optional[Customer]:
        """Get customer by ID."""
        return (
            self.db.query(Customer).filter(Customer.customer_id == customer_id).first()
        )

    def get_all(self, skip: int = 0, limit: int = 100) -> List[Customer]:
        """Get all customers."""
        return self.db.query(Customer).offset(skip).limit(limit).all()

    def update(self, customer_id: str, customer_data: dict) -> Optional[Customer]:
        """Update customer."""
        customer = self.get_by_id(customer_id)
        if customer:
            for key, value in customer_data.items():
                setattr(customer, key, value)
            self.db.commit()
            self.db.refresh(customer)
            logger.info(f"Updated customer: {customer_id}")
        return customer

    def delete(self, customer_id: str) -> bool:
        """Delete customer."""
        customer = self.get_by_id(customer_id)
        if customer:
            self.db.delete(customer)
            self.db.commit()
            logger.info(f"Deleted customer: {customer_id}")
            return True
        return False


class PredictionRepository:
    """Prediction data access."""

    def __init__(self, db: Session):
        """Initialize repository."""
        self.db = db

    def create(self, prediction_data: dict) -> Prediction:
        """Create new prediction."""
        prediction = Prediction(**prediction_data)
        self.db.add(prediction)
        self.db.commit()
        self.db.refresh(prediction)
        logger.info(f"Created prediction for customer: {prediction.customer_id}")
        return prediction

    def get_by_customer_id(self, customer_id: str) -> List[Prediction]:
        """Get predictions for customer."""
        return (
            self.db.query(Prediction)
            .filter(Prediction.customer_id == customer_id)
            .all()
        )

    def get_high_risk(self, threshold: float = 0.7) -> List[Prediction]:
        """Get high-risk predictions."""
        return (
            self.db.query(Prediction)
            .filter(Prediction.churn_probability >= threshold)
            .all()
        )


class ModelVersionRepository:
    """Model version data access."""

    def __init__(self, db: Session):
        """Initialize repository."""
        self.db = db

    def create(self, version_data: dict) -> ModelVersion:
        """Create new model version."""
        version = ModelVersion(**version_data)
        self.db.add(version)
        self.db.commit()
        self.db.refresh(version)
        logger.info(f"Created model version: {version.version}")
        return version

    def get_active(self) -> Optional[ModelVersion]:
        """Get active model version."""
        return (
            self.db.query(ModelVersion).filter(ModelVersion.is_active == True).first()
        )

    def get_all(self) -> List[ModelVersion]:
        """Get all model versions."""
        return self.db.query(ModelVersion).all()
