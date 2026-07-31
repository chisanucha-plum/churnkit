import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.data_loader import DataLoader
from app.services.feature_engineer import FeatureEngineer
# ModelTrainer import removed because training steps are currently commented out
from app.services.preprocessor import DataPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Train models."""
    logger.info("Starting model training")

    # Load data
    loader = DataLoader()
    df = loader.load_csv("data/raw/churn_data.csv")

    # Preprocess
    preprocessor = DataPreprocessor()
    df = preprocessor.preprocess(df, numeric_cols=[], categorical_cols=[])

    # Feature engineering
    engineer = FeatureEngineer()
    df = engineer.create_interaction_features(df, [])

    # Train model (training steps commented out until data split is available)
    # trainer = ModelTrainer(model_type="random_forest")
    # trainer.train(X_train, y_train)
    # metrics = trainer.evaluate(X_test, y_test)

    logger.info("Model training completed")


if __name__ == "__main__":
    main()
