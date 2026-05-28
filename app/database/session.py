"""
Database session management.

This module provides dependency injection for database sessions
in FastAPI endpoints.

Usage:
    from app.database.session import get_db
    
    @app.get("/data")
    def get_data(db: Session = Depends(get_db)):
        result = db.execute(query)
        return result
"""

import logging
from typing import Generator

from sqlalchemy.orm import Session

from app.database.connection import get_db_manager


logger = logging.getLogger(__name__)


def get_db() -> Generator[Session, None, None]:
    """
    Dependency injection for database sessions in FastAPI.
    
    Yields:
        Database session
        
    Usage:
        @app.get("/data")
        def get_data(db: Session = Depends(get_db)):
            result = db.execute(query)
            return result
    """
    db_manager = get_db_manager()
    
    try:
        with db_manager.get_session() as session:
            yield session
    except Exception as e:
        logger.error(f"Database session error: {str(e)}")
        raise
    finally:
        logger.debug("Database session closed")
