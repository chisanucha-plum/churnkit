"""Database setup script."""

import logging
import sys
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.database.connection import create_db_engine
from app.models.database_models import Base

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Setup database."""
    logger.info("Setting up database")

    engine = create_db_engine()
    Base.metadata.create_all(bind=engine)

    logger.info("Database setup completed")


if __name__ == "__main__":
    main()
