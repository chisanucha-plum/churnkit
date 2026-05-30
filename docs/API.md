# API Documentation

## Customer Churn Prediction API

### Base URL
```
http://localhost:8000/api/v1
```

### Authentication
All endpoints require an API key in the header:
```
X-API-Key: your-api-key
```

## Endpoints

### Health Check
```
GET /health
```

Returns application health status.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "database": "connected",
  "cache": "connected",
  "model_loaded": true
}
```

### Single Prediction
```
POST /predict
```

Make a prediction for a single customer.

**Request Body:**
```json
{
  "customer_id": "CUST001",
  "age": 35,
  "tenure_months": 24,
  "monthly_charges": 65.5,
  "total_charges": 1572.0,
  "contract_type": "Month-to-month",
  "internet_service": "Fiber optic",
  "payment_method": "Electronic check"
}
```

**Response:**
```json
{
  "customer_id": "CUST001",
  "churn_probability": 0.75,
  "churn_prediction": true,
  "risk_level": "high",
  "confidence": 0.92,
  "explanation": "High monthly charges and short tenure indicate churn risk"
}
```

### Batch Predictions
```
POST /predict/batch
```

Make predictions for multiple customers.

**Request Body:**
```json
{
  "customers": [
    {
      "customer_id": "CUST001",
      "age": 35,
      "tenure_months": 24,
      "monthly_charges": 65.5,
      "total_charges": 1572.0,
      "contract_type": "Month-to-month",
      "internet_service": "Fiber optic",
      "payment_method": "Electronic check"
    }
  ],
  "include_explanation": true
}
```

**Response:**
```json
{
  "predictions": [...],
  "total_count": 1,
  "high_risk_count": 1,
  "processing_time_ms": 125.5
}
```

### Model Metrics
```
GET /metrics/model
```

Get model performance metrics.

**Response:**
```json
{
  "accuracy": 0.923,
  "precision": 0.895,
  "recall": 0.852,
  "f1_score": 0.873,
  "auc_roc": 0.91,
  "confusion_matrix": {...},
  "feature_importance": {...}
}
```

### Model Information
```
GET /model/info
```

Get model information and metadata.

**Response:**
```json
{
  "version": "1.0.0",
  "type": "random_forest",
  "features": [...],
  "created_at": "2024-01-01T00:00:00Z",
  "deployed_at": "2024-01-01T00:00:00Z",
  "status": "active"
}
```

## Error Responses

All errors follow this format:

```json
{
  "error": "Error Type",
  "detail": "Detailed error message",
  "status_code": 400
}
```

## Rate Limiting

API requests are rate limited to 60 requests per minute per API key.

## Pagination

List endpoints support pagination:
- `skip`: Number of items to skip (default: 0)
- `limit`: Number of items to return (default: 100, max: 1000)
