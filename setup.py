"""Setup configuration for the package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="customer-churn-prediction",
    version="1.0.0",
    author="Development Team",
    description="ML-powered system for predicting customer churn",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=[
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "sqlalchemy>=2.0.0",
        "pydantic>=2.0.0",
        "pydantic-settings>=2.0.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "streamlit>=1.28.0",
        "plotly>=5.17.0",
        "redis>=5.0.0",
        "psycopg2-binary>=2.9.0",
        "python-dotenv>=1.0.0",
        "prometheus-client>=0.18.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.10.0",
            "flake8>=6.1.0",
            "mypy>=1.6.0",
            "isort>=5.12.0",
        ],
        "ml": [
            "shap>=0.43.0",
            "xgboost>=2.0.0",
            "lightgbm>=4.0.0",
        ],
        "llm": [
            "openai>=1.3.0",
        ],
    },
)
