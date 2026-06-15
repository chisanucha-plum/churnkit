# Setup Guide

## Prerequisites

- Python 3.9+
- pip or conda
- PostgreSQL (for production)
- Redis (optional, for caching)

## Local Development Setup

### 1. Clone Repository
```bash
git clone <repository-url>
cd customer-churn-prediction
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 4. Setup Environment Variables
```bash
cp .env.example .env.local
# Edit .env.local with your configuration
```

### 5. Setup Database
```bash
python scripts/setup_db.py
```

### 6. Run Application
```bash
python -m uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`

### 7. Run Dashboard
```bash
streamlit run dashboard/app.py
```

The dashboard will be available at `http://localhost:8501`

## Docker Setup

### Build Docker Image
```bash
docker build -t churn-prediction:latest .
```

### Run with Docker Compose
```bash
docker-compose up -d
```

## Testing

### Run Unit Tests
```bash
pytest tests/unit/ -v
```

### Run Integration Tests
```bash
pytest tests/integration/ -v
```

### Run All Tests with Coverage
```bash
pytest --cov=app tests/
```

## Model Training

### Train Models
```bash
python scripts/train_models.py
```

### Evaluate Models
```bash
python scripts/evaluate_models.py
```

### Generate Predictions
```bash
python scripts/generate_predictions.py
```

## Configuration

### Environment Variables

Key environment variables:

- `DEBUG`: Enable debug mode (default: False)
- `HOST`: API host (default: 0.0.0.0)
- `PORT`: API port (default: 8000)
- `DATABASE_URL`: Database connection string
- `REDIS_URL`: Redis connection string
- `API_KEY`: API authentication key
- `LOG_LEVEL`: Logging level (default: INFO)
- `LLM_ENABLED`: Enable LLM integration (default: False)
- `LLM_API_KEY`: LLM API key
- `LLM_MODEL`: LLM model name (default: gpt-4)

### Database Configuration

For PostgreSQL:
```
DATABASE_URL=postgresql://user:password@localhost:5432/churn_db
```

For SQLite (development):
```
DATABASE_URL=sqlite:///./churn_prediction.db
```

## Troubleshooting

### Port Already in Use
```bash
# Find process using port 8000
lsof -i :8000

# Kill process
kill -9 <PID>
```

### Database Connection Issues
- Verify DATABASE_URL is correct
- Check database server is running
- Verify credentials

### Missing Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Next Steps

1. Review API documentation: `docs/API.md`
2. Check architecture: `docs/ARCHITECTURE.md`
3. Review deployment guide: `docs/DEPLOYMENT.md`
