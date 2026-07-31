"""
Database connection management.

This module provides:
1. Connection pooling for PostgreSQL and SQLite
2. Connection retry logic
3. Health checks
4. Connection pool status monitoring

Usage:
    from app.database.connection import get_db_engine
    engine = get_db_engine()
"""

import logging
import time
from typing import Optional
from contextlib import contextmanager

from sqlalchemy import create_engine, text, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError, OperationalError
from sqlalchemy.pool import QueuePool, StaticPool

from app.config.settings import get_settings

logger = logging.getLogger(__name__)


class DatabaseConnectionManager:
    """
    Manages database connections with pooling and retry logic.

    Features:
    - Connection pooling for PostgreSQL
    - SQLite support
    - Automatic retry on connection failures
    - Health checks
    - Connection pool monitoring
    """

    def __init__(self):
        """Initialize database connection manager."""
        self.engine = None
        self.session_factory = None
        self.settings = get_settings()
        self._initialize()

    def _initialize(self) -> None:
        """Initialize database connection."""
        try:
            logger.info("Initializing database connection...")

            database_url = self.settings.get_database_url()
            logger.info(f"Database URL: {database_url[:50]}...")  # Hide sensitive info

            # Determine if PostgreSQL or SQLite
            if "postgresql" in database_url:
                self._initialize_postgresql(database_url)
            else:
                self._initialize_sqlite(database_url)

            # Test connection
            self._test_connection()

            logger.info("Database connection initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database: {str(e)}")
            raise

    def _initialize_postgresql(self, database_url: str) -> None:
        """Initialize PostgreSQL connection with pooling."""
        logger.info("Setting up PostgreSQL connection pool...")

        self.engine = create_engine(
            database_url,
            poolclass=QueuePool,
            pool_size=self.settings.database_pool_size,
            max_overflow=self.settings.database_max_overflow,
            pool_timeout=self.settings.database_pool_timeout,
            pool_recycle=3600,  # Recycle connections after 1 hour
            echo=self.settings.debug,
            connect_args={
                "connect_timeout": 10,
                "application_name": "churn_prediction",
            },
        )

        # Set up event listeners
        @event.listens_for(self.engine, "connect")
        def receive_connect(dbapi_conn, connection_record):
            logger.debug("PostgreSQL connection established")

        @event.listens_for(self.engine, "close")
        def receive_close(dbapi_conn, connection_record):
            logger.debug("PostgreSQL connection closed")

        self.session_factory = sessionmaker(bind=self.engine)
        logger.info("PostgreSQL connection pool configured successfully")

    def _initialize_sqlite(self, database_url: str) -> None:
        """Initialize SQLite connection."""
        logger.info("Setting up SQLite connection...")

        self.engine = create_engine(
            database_url,
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
            echo=self.settings.debug,
        )

        self.session_factory = sessionmaker(bind=self.engine)
        logger.info("SQLite connection configured successfully")

    def _test_connection(self) -> None:
        """Test database connection with retry logic."""
        retry_count = 0
        last_error = None

        while retry_count < self.settings.connection_retry_count:
            try:
                with self.engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                    conn.commit()

                logger.info("Database connection test successful")
                return
            except Exception as e:
                last_error = e
                retry_count += 1

                if retry_count < self.settings.connection_retry_count:
                    wait_time = self.settings.connection_retry_delay * retry_count
                    logger.warning(
                        f"Connection test failed (attempt {retry_count}/"
                        f"{self.settings.connection_retry_count}). "
                        f"Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)

        raise ConnectionError(
            f"Failed to connect to database after {self.settings.connection_retry_count} attempts. "
            f"Error: {str(last_error)}"
        )

    @contextmanager
    def get_session(self):
        """
        Get a database session (context manager).

        Usage:
            with db_manager.get_session() as session:
                result = session.execute(query)
        """
        if self.session_factory is None:
            raise RuntimeError("Database not initialized")

        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {str(e)}")
            raise
        finally:
            session.close()

    def health_check(self) -> bool:
        """
        Check if database connection is healthy.

        Returns:
            True if healthy, False otherwise
        """
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.debug("Database health check passed")
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {str(e)}")
            return False

    def get_pool_status(self) -> dict:
        """
        Get connection pool status.

        Returns:
            Dictionary with pool statistics
        """
        try:
            pool = self.engine.pool

            if isinstance(pool, QueuePool):
                return {
                    "type": "QueuePool",
                    "size": pool.size(),
                    "checked_out": pool.checkedout(),
                    "overflow": pool.overflow(),
                    "total": pool.size() + pool.overflow(),
                }
            elif isinstance(pool, StaticPool):
                return {"type": "StaticPool", "status": "active"}
            else:
                return {"type": str(type(pool).__name__)}
        except Exception as e:
            logger.error(f"Failed to get pool status: {str(e)}")
            return {"error": str(e)}

    def close(self) -> None:
        """Close all database connections."""
        try:
            if self.engine:
                self.engine.dispose()
                logger.info("Database connections closed successfully")
        except Exception as e:
            logger.error(f"Error closing database connections: {str(e)}")


# Global database manager instance
_db_manager: Optional[DatabaseConnectionManager] = None


def get_db_manager() -> DatabaseConnectionManager:
    """
    Get or create the global database manager.

    Returns:
        DatabaseConnectionManager instance
    """
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseConnectionManager()
    return _db_manager


def get_db_engine():
    """
    Get the SQLAlchemy engine.

    Returns:
        SQLAlchemy engine
    """
    return get_db_manager().engine


def get_db_session():
    """
    Get a database session.

    Returns:
        Database session context manager
    """
    return get_db_manager().get_session()


def close_db() -> None:
    """Close database connections."""
    global _db_manager
    if _db_manager:
        _db_manager.close()
        _db_manager = None
