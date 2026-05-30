# Architecture Documentation

## System Architecture

### Overview
The Customer Churn Prediction System is built with a layered architecture:

```
┌─────────────────────────────────────────┐
│         API Layer (FastAPI)             │
│  ├─ Routers (predict, metrics, health)  │
│  └─ Controllers (business logic)        │
├─────────────────────────────────────────┤
│         Service Layer                   │
│  ├─ Prediction Engine                   │
│  ├─ Model Trainer                       │
│  ├─ Data Preprocessor                   │
│  ├─ Feature Engineer                    │
│  ├─ Explainability Service              │
│  └─ LLM Integration                     │
├─────────────────────────────────────────┤
│         Data Layer                      │
│  ├─ Database (SQLAlchemy)               │
│  ├─ Cache (Redis/In-Memory)             │
│  └─ Repositories (Data Access)          │
├─────────────────────────────────────────┤
│         ML Layer                        │
│  ├─ Model Training                      │
│  ├─ Model Evaluation                    │
│  └─ Feature Engineering                 │
├─────────────────────────────────────────┤
│         Monitoring & Observability      │
│  ├─ Prometheus Metrics                  │
│  ├─ Distributed Tracing                 │
│  └─ Alert Management                    │
└─────────────────────────────────────────┘
```

## Key Components

### 1. API Layer
- **FastAPI**: Modern web framework for building APIs
- **Routers**: Organize endpoints by functionality
- **Controllers**: Handle request/response logic

### 2. Service Layer
- **PredictionEngine**: Makes predictions using trained models
- **ModelTrainer**: Trains and evaluates ML models
- **DataPreprocessor**: Cleans and prepares data
- **FeatureEngineer**: Creates and transforms features
- **ExplainabilityService**: Provides model interpretability (SHAP)
- **LLMInsightsService**: Generates AI-powered insights

### 3. Data Layer
- **Database**: SQLAlchemy ORM for data persistence
- **Cache**: Redis or in-memory caching for performance
- **Repositories**: Data access objects for clean separation

### 4. ML Layer
- **Model Training**: Scikit-learn based model training
- **Model Evaluation**: Comprehensive metrics and validation
- **Feature Engineering**: Advanced feature creation and selection

### 5. Monitoring & Observability
- **Prometheus Metrics**: Application metrics collection
- **Distributed Tracing**: Request tracing across services
- **Alert Management**: System alerts and notifications

## Data Flow

### Prediction Flow
```
Customer Data → Validation → Preprocessing → Feature Engineering → 
Model Prediction → Risk Assessment → Explanation Generation → Response
```

### Training Flow
```
Raw Data → Loading → Preprocessing → Feature Engineering → 
Train/Test Split → Model Training → Evaluation → Model Versioning
```

## Database Schema

### Customers Table
- customer_id (PK)
- age, tenure_months, monthly_charges, total_charges
- contract_type, internet_service, payment_method
- created_at, updated_at

### Predictions Table
- id (PK)
- customer_id (FK)
- churn_probability, churn_prediction
- risk_level, confidence
- model_version, created_at

### ModelVersions Table
- id (PK)
- version (unique)
- accuracy, precision, recall, f1_score, auc_roc
- is_active, created_at, deployed_at

## Deployment Architecture

### Development
- Local SQLite database
- In-memory caching
- Single process

### Production
- PostgreSQL database
- Redis caching
- Load-balanced API servers
- Separate model serving
- Monitoring and logging infrastructure

## Security Considerations

1. **API Authentication**: API key validation on all endpoints
2. **Input Validation**: Comprehensive input validation
3. **Error Handling**: Secure error messages without sensitive data
4. **Database**: Connection pooling and prepared statements
5. **Logging**: Sensitive data masking in logs

## Scalability

1. **Horizontal Scaling**: Stateless API servers
2. **Caching**: Redis for distributed caching
3. **Database**: Connection pooling and read replicas
4. **Batch Processing**: Async batch prediction jobs
5. **Model Serving**: Separate model serving infrastructure
