import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config.settings import get_settings
from app.database.connection import close_db, get_db_manager
from app.dependencies import set_model_trainer
from app.routers import health, metrics, prediction
from app.services.churn_model import ChurnModelTrainer
from app.services.sample_data import generate_sample_data, save_sample_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application startup and shutdown.

    Initializes database connection and loads/trains the ML model.
    """
    # Initialize database
    try:
        db_manager = get_db_manager()
        if db_manager.health_check():
            logger.info("Database connection established successfully")
        else:
            logger.warning("Database connection unhealthy")
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")

    # Load or train model
    try:
        model_trainer = ChurnModelTrainer()

        # Try to load existing model
        try:
            model_trainer.load("churn_model.pkl")
            logger.info("Model loaded successfully from disk")
        except FileNotFoundError:
            logger.info(
                "Model not found. Generating sample data and training new model..."
            )

            # Generate sample data
            df = generate_sample_data(n_samples=1000)
            save_sample_data(df)

            # Train model
            model_trainer.train(df)
            model_trainer.save("churn_model.pkl")
            logger.info("Model trained and saved successfully")

        # Set model trainer for dependency injection
        set_model_trainer(model_trainer)

    except Exception as e:
        logger.error(f"Model initialization failed: {str(e)}")

    logger.info("Application startup completed successfully")

    yield

    # Shutdown
    logger.info("Shutting down application...")
    close_db()
    logger.info("Application shutdown completed")


# ===== Create FastAPI App =====
settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Customer Churn Prediction System - Layered Architecture",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== Register Routers =====
app.include_router(health.router)
app.include_router(prediction.router)
app.include_router(metrics.router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
    )
