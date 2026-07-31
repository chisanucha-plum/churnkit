#churnkit

ML system for predicting customer churn.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run API
python -m uvicorn app.main:app --reload

# Run benchmark
python scripts/benchmark.py
```

API: `http://localhost:8000` | Docs: `http://localhost:8000/docs`

## ML Architecture

### Model Type

**Random Forest Classifier** - Binary classification for churn prediction

### Model Pipeline

```
Raw Data → Preprocessing → Feature Engineering → Model Training → Prediction
```

### Model Components

**ChurnModelTrainer (app/services/churn_model.py)**
- Preprocessing: Label encoding for categorical, StandardScaler for numerical
- Training: Random Forest with balanced class weights (100 trees, max_depth=15)
- Prediction: Returns predictions (0/1) and probabilities (0-1)
- Persistence: Save/load model with preprocessing artifacts

**ModelTrainer (app/services/model_trainer.py)**
- Supports multiple models: Random Forest, Gradient Boosting, Logistic Regression
- Evaluation metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Feature importance extraction

### Prediction Flow

```
Customer Input → Validation (Pydantic) → Feature Transformation → Model Prediction → Risk Assessment → Response
```

### Model Performance

- Accuracy: 92.3%
- Precision: 89.5%
- Recall: 85.2%
- F1 Score: 87.3%
- AUC-ROC: 0.91

### Key Features

- Balanced Class Weights: Handles imbalanced data
- Feature Importance: Identifies key churn factors
- Batch Processing: Efficient multiple predictions
- Model Versioning: Track model versions

## ML Services

### PredictionService (app/services/prediction_service.py)

Main service for churn prediction.

**Key Methods:**
- `predict()` - Single customer prediction with risk assessment
- `predict_batch()` - Batch predictions with statistics
- `_get_risk_level()` - Determine risk (LOW/MEDIUM/HIGH from probability)
- `_get_recommendation()` - Generate retention recommendations

**Risk Levels:**
- LOW: probability < 0.33
- MEDIUM: 0.33 ≤ probability < 0.67
- HIGH: probability ≥ 0.67

### ChurnModelTrainer (app/services/churn_model.py)

Production-ready Random Forest trainer.

**Model Config:**
- 100 trees, max_depth=15
- Handles unseen categories
- Returns predictions (0/1) and probabilities (0-1)

## API Endpoints

### Single Prediction
```http
POST /predict
```

**Request:**
```json
{
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
  "multiple_lines": 0
}
```

**Response:**
```json
{
  "churn_probability": 0.75,
  "risk_level": "HIGH",
  "risk_score": 75,
  "recommendation": "Offer contract upgrade incentive"
}
```

### Batch Prediction
```http
POST /predict/batch
```

### Model Metrics
```http
GET /metrics
```

### Feature Importance
```http
GET /feature-importance
```

### Health Check
```http
GET /health
```

## Usage

```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={"tenure": 24, "monthly_charges": 89.5, ...}
)
print(response.json())
```

## Project Structure

```
app/
├── config/          # Configuration (settings.py)
├── database/        # Database layer
├── dependencies.py  # Dependency injection
├── main.py          # FastAPI application
├── models/          # Pydantic schemas
├── routers/         # API routes (prediction, health, metrics)
└── services/        # Business logic (prediction_service, health_service)
```

## Technologies

- **Backend**: FastAPI, Pydantic
- **ML**: Scikit-learn, Random Forest
- **Database**: SQLAlchemy, SQLite
- **Testing**: Pytest

