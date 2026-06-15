# Customer Churn Prediction System - Setup Guide

## Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- PostgreSQL (optional, for database integration)
- Git (for version control)

## Project Structure

```
Customer Churn Prediction/
├── src/                          # Source code
│   ├── data_loader.py           # Data loading module
│   ├── preprocessor.py          # Data preprocessing module
│   ├── feature_engineer.py      # Feature engineering module
│   ├── model_trainer.py         # Model training module
│   ├── explainability.py        # SHAP and feature importance
│   ├── api.py                   # FastAPI application
│   ├── dashboard.py             # Streamlit dashboard
│   └── utils.py                 # Utility functions
├── tests/                        # Test files
│   ├── test_data_loader.py
│   ├── test_preprocessor.py
│   ├── test_feature_engineer.py
│   ├── test_model_trainer.py
│   ├── test_api.py
│   └── test_integration.py
├── data/                         # Data directory
│   ├── raw/                     # Raw data files
│   ├── processed/               # Processed data
│   └── sample/                  # Sample data for testing
├── models/                       # Trained models
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   └── lightgbm.pkl
├── config/                       # Configuration files
│   ├── preprocessing_config.json
│   ├── feature_engineering_config.json
│   └── model_config.json
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment variables template
├── SETUP.md                      # This file
└── README.md                     # Project documentation
```

## Installation Steps

### 1. Clone or Navigate to Project Directory

```bash
cd "c:\Customer Churn Prediction"
```

### 2. Create Virtual Environment

#### On Windows (PowerShell):
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

#### On Windows (Command Prompt):
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

#### On macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Upgrade pip

```bash
python -m pip install --upgrade pip
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure Environment Variables

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your configuration
# Update database credentials, API keys, LLM settings, etc.
```

### 6. Verify Installation

```bash
# Check Python version
python --version

# Check installed packages
pip list

# Run a quick test
python -c "import pandas; import sklearn; import xgboost; import lightgbm; print('All dependencies installed successfully!')"
```

## Running the Application

### 1. Data Processing Pipeline

```bash
python src/data_loader.py
python src/preprocessor.py
python src/feature_engineer.py
```

### 2. Model Training

```bash
python src/model_trainer.py
```

### 3. Start API Server

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at: `http://localhost:8000`
API documentation: `http://localhost:8000/docs`

### 4. Start Dashboard

```bash
streamlit run src/dashboard.py
```

The dashboard will be available at: `http://localhost:8501`

### 5. Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_data_loader.py -v
```

## Database Setup (Optional)

### PostgreSQL Setup

1. Install PostgreSQL from https://www.postgresql.org/download/

2. Create database:
```sql
CREATE DATABASE churn_db;
CREATE USER churn_user WITH PASSWORD 'your_password';
ALTER ROLE churn_user SET client_encoding TO 'utf8';
ALTER ROLE churn_user SET default_transaction_isolation TO 'read committed';
ALTER ROLE churn_user SET default_transaction_deferrable TO on;
ALTER ROLE churn_user SET default_transaction_read_only TO off;
GRANT ALL PRIVILEGES ON DATABASE churn_db TO churn_user;
```

3. Update `.env` with connection string:
```
DATABASE_URL=postgresql://churn_user:your_password@localhost:5432/churn_db
```

## Development Workflow

### Code Style

```bash
# Format code with Black
black src/ tests/

# Check code style with Flake8
flake8 src/ tests/

# Type checking with mypy
mypy src/
```

### Pre-commit Hooks (Optional)

```bash
pip install pre-commit
pre-commit install
```

## Troubleshooting

### Virtual Environment Issues

**Problem**: Virtual environment not activating
**Solution**: 
- Ensure you're in the project directory
- Try using the full path: `.\venv\Scripts\Activate.ps1`
- On PowerShell, you may need to set execution policy: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### Dependency Installation Issues

**Problem**: `pip install` fails with permission error
**Solution**: 
- Ensure virtual environment is activated
- Try: `pip install --user -r requirements.txt`

**Problem**: PostgreSQL connection fails
**Solution**:
- Verify PostgreSQL is running
- Check connection string in `.env`
- Verify database and user exist
- Check firewall settings

### API Port Already in Use

**Problem**: Port 8000 is already in use
**Solution**: 
```bash
# Use a different port
uvicorn src.api:app --host 0.0.0.0 --port 8001
```

### Dashboard Port Already in Use

**Problem**: Port 8501 is already in use
**Solution**:
```bash
# Use a different port
streamlit run src/dashboard.py --server.port 8502
```

## Environment Variables Reference

| Variable | Description | Default |
|----------|-------------|---------|
| `ENVIRONMENT` | Environment (development/production) | development |
| `API_PORT` | API server port | 8000 |
| `DATABASE_URL` | PostgreSQL connection string | - |
| `LLM_PROVIDER` | LLM provider (openrouter/gemini/ollama) | openrouter |
| `LLM_API_KEY` | LLM API key | - |
| `LOG_LEVEL` | Logging level (DEBUG/INFO/WARNING/ERROR) | INFO |
| `MASK_PII_IN_LOGS` | Mask PII in logs | True |

## Next Steps

1. Load sample data into the `data/raw/` directory
2. Run the data processing pipeline
3. Train models using the model trainer
4. Start the API and dashboard
5. Make predictions and monitor performance

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the design.md and requirements.md documents
3. Check API documentation at `/docs` endpoint
4. Review logs in the `logs/` directory

## Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [scikit-learn Documentation](https://scikit-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [SHAP Documentation](https://shap.readthedocs.io/)
