# Customer Churn Prediction System

A production-ready ML system for predicting customer churn with FastAPI backend, Streamlit dashboard, and comprehensive monitoring.

## Features

- **ML Predictions**: Random Forest, Gradient Boosting, and Logistic Regression models
- **REST API**: FastAPI with comprehensive endpoints for predictions and metrics
- **Dashboard**: Streamlit-based interactive dashboard for visualization
- **Explainability**: SHAP-based model interpretability
- **LLM Integration**: AI-powered insights and recommendations
- **Monitoring**: Prometheus metrics, distributed tracing, and alerts
- **Database**: SQLAlchemy ORM with PostgreSQL support
- **Caching**: Redis caching for performance optimization
- **Testing**: Comprehensive unit and integration tests
- **Docker**: Production-ready Docker and Docker Compose setup

## Quick Start

### Prerequisites
- Python 3.9+
- Docker and Docker Compose (optional)

### Local Development

1. **Clone repository**
```bash
git clone <repository-url>
cd customer-churn-prediction
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Setup environment**
```bash
cp .env.example .env.local
```

5. **Setup database**
```bash
python scripts/setup_db.py
```

6. **Run API**
```bash
python main.py
```

API will be available at `http://localhost:8000`

7. **Run Dashboard** (in another terminal)
```bash
streamlit run dashboard/app.py
```

Dashboard will be available at `http://localhost:8501`

### Docker Deployment

```bash
docker-compose up -d
```

## Project Structure

```
customer-churn-prediction/
├── app/                    # Main application
│   ├── config/            # Configuration
│   ├── models/            # Data models and schemas
│   ├── services/          # Business logic
│   ├── controllers/       # Request handlers
│   ├── routers/           # API routes
│   ├── middleware/        # Custom middleware
│   ├── utils/             # Utilities
│   ├── database/          # Database layer
│   ├── cache/             # Caching
│   └── monitoring/        # Monitoring
├── dashboard/             # Streamlit dashboard
├── scripts/               # Utility scripts
├── tests/                 # Test suite
├── docs/                  # Documentation
└── data/                  # Data directory
```

## API Endpoints

### Health Check
```
GET /api/v1/health
```

### Single Prediction
```
POST /api/v1/predict
```

### Batch Predictions
```
POST /api/v1/predict/batch
```

### Model Metrics
```
GET /api/v1/metrics/model
```

### Model Information
```
GET /api/v1/model/info
```

See [API Documentation](docs/API.md) for detailed endpoint documentation.

## Configuration

Environment variables in `.env`:

```
DEBUG=False
HOST=0.0.0.0
PORT=8000
DATABASE_URL=sqlite:///./churn_prediction.db
REDIS_URL=redis://localhost:6379/0
API_KEY=your-api-key
LOG_LEVEL=INFO
LLM_ENABLED=False
LLM_API_KEY=your-openai-key
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app tests/

# Run specific test file
pytest tests/unit/test_prediction.py -v
```

## Model Training

```bash
# Train models
python scripts/train_models.py

# Evaluate models
python scripts/evaluate_models.py

# Generate predictions
python scripts/generate_predictions.py
```

## Documentation

- [Setup Guide](docs/SETUP.md)
- [API Documentation](docs/API.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Deployment Guide](docs/DEPLOYMENT.md)

## Performance Metrics

- **Accuracy**: 92.3%
- **Precision**: 89.5%
- **Recall**: 85.2%
- **F1 Score**: 87.3%
- **AUC-ROC**: 0.91

## Technologies

- **Backend**: FastAPI, SQLAlchemy, Pydantic
- **ML**: Scikit-learn, XGBoost, LightGBM
- **Frontend**: Streamlit, Plotly
- **Database**: PostgreSQL, SQLite
- **Cache**: Redis
- **Monitoring**: Prometheus, OpenTelemetry
- **Containerization**: Docker, Docker Compose
- **Testing**: Pytest, Pytest-cov

## Contributing

1. Create a feature branch
2. Make your changes
3. Run tests
4. Submit a pull request

## License

MIT License - see LICENSE file for details

## Support

For issues and questions, please open an issue on GitHub.

## Roadmap

- [ ] Advanced feature engineering
- [ ] Ensemble model improvements
- [ ] Real-time prediction streaming
- [ ] Mobile app integration
- [ ] Advanced visualization
- [ ] Multi-language support
