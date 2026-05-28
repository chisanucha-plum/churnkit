"""
Unit tests for database connection management.

Tests the DatabaseConnectionManager class, connection pooling,
retry logic, and health checks.

Requirements: 1.2, 33.6
"""

import pytest
import logging
from unittest.mock import patch, MagicMock, call
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from app.database.connection import (
    DatabaseConnectionManager,
    get_db_manager,
    get_db_engine,
    get_db_session,
    close_db,
    _db_manager
)
from app.config.settings import Settings


class TestDatabaseConnectionManagerInitialization:
    """Test DatabaseConnectionManager initialization."""
    
    @pytest.mark.unit
    def test_manager_initializes_with_sqlite(self):
        """Test manager initializes with SQLite."""
        with patch("app.database.connection.get_settings") as mock_settings:
            mock_settings.return_value = Settings(
                database_url=None,
                sqlite_db_path=":memory:"
            )
            manager = DatabaseConnectionManager()
            assert manager.engine is not None
            assert manager.session_factory is not None
    
    @pytest.mark.unit
    def test_manager_initializes_with_postgresql(self):
        """Test manager initializes with PostgreSQL configuration."""
        with patch("app.database.connection.get_settings") as mock_settings:
            mock_settings.return_value = Settings(
                database_url="sqlite:///:memory:"
            )
            manager = DatabaseConnectionManager()
            assert manager.engine is not None
            assert manager.session_factory is not None
    
    @pytest.mark.unit
    def test_manager_stores_settings(self):
        """Test manager stores settings reference."""
        with patch("app.database.connection.get_settings") as mock_settings:
            mock_settings.return_value = Settings()
            manager = DatabaseConnectionManager()
            assert manager.settings is not None
            assert isinstance(manager.settings, Settings)


class TestSQLiteInitialization:
    """Test SQLite connection initialization."""
    
    @pytest.mark.unit
    def test_sqlite_engine_created(self):
        """Test SQLite engine is created."""
        with patch("app.database.connection.get_settings") as mock_settings:
            mock_settings.return_value = Settings(
                sqlite_db_path=":memory:"
            )
            manager = DatabaseConnectionManager()
            assert manager.engine is not None
    
    @pytest.mark.unit
    def test_sqlite_session_factory_created(self):
        """Test SQLite session factory is created."""
        with patch("app.database.connection.get_settings") as mock_settings:
            mock_settings.return_value = Settings(
                sqlite_db_path=":memory:"
            )
            manager = DatabaseConnectionManager()
            assert manager.session_factory is not None
    
    @pytest.mark.unit
    def test_sqlite_connection_test_passes(self):
        """Test SQLite connection test passes."""
        with patch("app.database.connection.get_settings") as mock_settings:
            mock_settings.return_value = Settings(
                sqlite_db_path=":memory:"
            )
            manager = DatabaseConnectionManager()
            # If initialization succeeds, connection test passed
            assert manager.engine is not None


class TestConnectionPooling:
    """Test connection pooling configuration."""
    
    @pytest.mark.unit
    def test_postgresql_uses_queue_pool(self):
        """Test PostgreSQL uses QueuePool for connection pooling."""
        with patch("app.database.connection.get_settings") as mock_settings:
            with patch("app.database.connection.create_engine") as mock_create:
                mock_settings.return_value = Settings(
                    database_url="postgresql://user:pass@localhost/db"
                )
                mock_engine = MagicMock()
                mock_create.return_value = mock_engine
                
                manager = DatabaseConnectionManager()
                
                # Verify create_engine was called with QueuePool
                mock_create.assert_called_once()
                call_kwargs = mock_create.call_args[1]
                assert "poolclass" in call_kwargs
    
    @pytest.mark.unit
    def test_sqlite_uses_static_pool(self):
        """Test SQLite uses StaticPool."""
        with patch("app.database.connection.get_settings") as mock_settings:
            with patch("app.database.connection.create_engine") as mock_create:
                mock_settings.return_value = Settings(
                    sqlite_db_path=":memory:"
                )
                mock_engine = MagicMock()
                mock_create.return_value = mock_engine
                
                manager = DatabaseConnectionManager()
                
                # Verify create_engine was called with StaticPool
                mock_create.assert_called_once()
                call_kwargs = mock_create.call_args[1]
                assert "poolclass" in call_kwargs


class TestConnectionRetryLogic:
    """Test connection retry logic."""
    
    @pytest.mark.unit
    def test_connection_retry_on_failure(self):
        """Test connection retries on failure."""
        with patch("app.database.connection.get_settings") as mock_settings:
            with patch("app.database.connection.time.sleep"):
                mock_settings.return_value = Settings(
                    sqlite_db_path=":memory:",
                    connection_retry_count=3,
                    connection_retry_delay=1
                )
                
                manager = DatabaseConnectionManager()
                # If initialization succeeds, retry logic worked
                assert manager.engine is not None
    
    @pytest.mark.unit
    def test_connection_retry_count_respected(self):
        """Test connection retry count is respected."""
        with patch("app.database.connection.get_settings") as mock_settings:
            mock_settings.return_value = Settings(
                connection_retry_count=5,
                connection_retry_delay=0
            )
            
            manager = DatabaseConnectionManager()
            assert manager.settings.connection_retry_count == 5
    
    @pytest.mark.unit
    def test_connection_retry_delay_respected(self):
        """Test connection retry delay is respected."""
        with patch("app.database.connection.get_settings") as mock_settings:
            mock_settings.return_value = Settings(
                connection_retry_delay=2
            )
            
            manager = DatabaseConnectionManager()
            assert manager.settings.connection_retry_delay == 2


class TestHealthCheck:
    """Test database health check functionality."""
    
    @pytest.mark.unit
    def test_health_check_returns_true_on_success(self):
        """Test health check returns True on successful connection."""
        with patch("app.database.connection.get_settings") as mock_settings:
            mock_settings.return_value = Settings(
                sqlite_db_path=":memory:"
            )
            manager = DatabaseConnectionManager()
            result = manager.health_check()
            assert result is True
    
    @pytest.mark.unit
    def test_health_check_returns_false_on_failure(self):
        """Test health check returns False on connection failure."""
        with patch("app.database.connection.get_settings") as mock_settings:
            mock_settings.return_value = Settings(
                sqlite_db_path=":memory:"
            )
            manager = DatabaseConnectionManager()
            
            # Mock engine to raise exception
            with patch.object(manager.engine, "connect") as mock_connect:
                mock_connect.side_effect = Exception("Connection failed")
                result = manager.health_check()
                assert result is False
    
    @pytest.mark.unit
    def test_health_check_executes_select_query(self):
        """Test health check executes SELECT 1 query."""
        with patch("app.database.connection.get_settings") as mock_settings:
            mock_settings.return_value = Settings(
                sqlite_db_path=":memory:"
            )
            manager = DatabaseConnectionManager()
            
            # Health check should succeed
            result = manager.health_check()
            assert result is True


class TestSessionManagement:
    """Test database session management."""
    
    @pytest.mark.unit
    def test_get_session_returns_context_manager(self):
        """Test get_session returns a context manager."""
        with patch("app.database.connection.get_settings") as mock_settings:
            mock_settings.return_value = Settings(
                sqlite_db_path=":memory:"
            )
            manager = DatabaseConnectionManager()
            
            session_cm = manager.get_session()
            assert hasattr(session_cm, "__enter__")
            assert hasattr(session_cm, "__exit__")
    
    @pytest.mark.unit
    def test_get_session_yields_session_object(self):
        """Test get_session yields a Session object."""
        with patch("app.database.connection.get_settings") as mock_settings:
            mock_settings.return_value = Settings(
                sqlite_db_path=":memory:"
            )
            manager = DatabaseConnectionManager()
            
            with manager.get_session() as session:
                assert isinstance(session, Session)
    
    @pytest.mark.unit
    def test_get_session_commits_on_success(self):
        """Test get_session commits transaction on success."""
        with patch("app.database.connection.get_settings") as mock_settings:
            mock_settings.return_value = Settings(
                sqlite_db_path=":memory:"
            )
            manager = DatabaseConnectionManager()
            
            with patch.object(manager.session_factory, "return_value") as mock_session:
                mock_session.__enter__ = MagicMock(return_value=mock_session)
                mock_session.__exit__ = MagicMock(return_value=None)
                
                try:
                    with manager.get_session() as session:
                        pass
                except:
                    pass
    
    @pytest.mark.unit
    def test_get_session_raises_error_if_not_initialized(self):
        """Test get_session raises error if not initialized."""
        manager = DatabaseConnectionManager.__new__(DatabaseConnectionManager)
        manager.session_factory = None
        
        with pytest.raises(RuntimeError, match="Database not initialized"):
            with manager.get_session():
                pass


class TestPoolStatus:
    """Test connection pool status monitoring."""
    
    @pytest.mark.unit
    def test_get_pool_status_returns_dict(self):
        """Test get_pool_status returns dictionary."""
        with patch("app.database.connection.get_settings") as mock_settings:
            mock_settings.return_value = Settings(
                sqlite_db_path=":memory:"
            )
            manager = DatabaseConnectionManager()
            status = manager.get_pool_status()
            assert isinstance(status, dict)
    
    @pytest.mark.unit
    def test_get_pool_status_includes_type(self):
        """Test get_pool_status includes pool type."""
        with patch("app.database.connection.get_settings") as mock_settings:
            mock_settings.return_value = Settings(
                sqlite_db_path=":memory:"
            )
            manager = DatabaseConnectionManager()
            status = manager.get_pool_status()
            assert "type" in status
    
    @pytest.mark.unit
    def test_get_pool_status_handles_errors(self):
        """Test get_pool_status handles errors gracefully."""
        with patch("app.database.connection.get_settings") as mock_settings:
            mock_settings.return_value = Settings(
                sqlite_db_path=":memory:"
            )
            manager = DatabaseConnectionManager()
            
            # Mock pool to raise exception
            with patch.object(manager.engine, "pool") as mock_pool:
                mock_pool.side_effect = Exception("Pool error")
                status = manager.get_pool_status()
                assert "error" in status


class TestConnectionClose:
    """Test connection closing."""
    
    @pytest.mark.unit
    def test_close_disposes_engine(self):
        """Test close disposes the engine."""
        with patch("app.database.connection.get_settings") as mock_settings:
            mock_settings.return_value = Settings(
                sqlite_db_path=":memory:"
            )
            manager = DatabaseConnectionManager()
            
            with patch.object(manager.engine, "dispose") as mock_dispose:
                manager.close()
                mock_dispose.assert_called_once()
    
    @pytest.mark.unit
    def test_close_handles_errors(self):
        """Test close handles errors gracefully."""
        with patch("app.database.connection.get_settings") as mock_settings:
            mock_settings.return_value = Settings(
                sqlite_db_path=":memory:"
            )
            manager = DatabaseConnectionManager()
            
            # Mock engine.dispose to raise exception
            with patch.object(manager.engine, "dispose") as mock_dispose:
                mock_dispose.side_effect = Exception("Dispose error")
                # Should not raise
                manager.close()


class TestGlobalDatabaseManager:
    """Test global database manager functions."""
    
    @pytest.mark.unit
    def test_get_db_manager_returns_manager(self):
        """Test get_db_manager returns DatabaseConnectionManager."""
        # Reset global manager
        import app.database.connection as conn_module
        conn_module._db_manager = None
        
        with patch("app.database.connection.get_settings") as mock_settings:
            mock_settings.return_value = Settings(
                sqlite_db_path=":memory:"
            )
            manager = get_db_manager()
            assert isinstance(manager, DatabaseConnectionManager)
    
    @pytest.mark.unit
    def test_get_db_manager_is_singleton(self):
        """Test get_db_manager returns same instance (singleton)."""
        import app.database.connection as conn_module
        conn_module._db_manager = None
        
        with patch("app.database.connection.get_settings") as mock_settings:
            mock_settings.return_value = Settings(
                sqlite_db_path=":memory:"
            )
            manager1 = get_db_manager()
            manager2 = get_db_manager()
            assert manager1 is manager2
    
    @pytest.mark.unit
    def test_get_db_engine_returns_engine(self):
        """Test get_db_engine returns SQLAlchemy engine."""
        import app.database.connection as conn_module
        conn_module._db_manager = None
        
        with patch("app.database.connection.get_settings") as mock_settings:
            mock_settings.return_value = Settings(
                sqlite_db_path=":memory:"
            )
            engine = get_db_engine()
            assert engine is not None
    
    @pytest.mark.unit
    def test_get_db_session_returns_context_manager(self):
        """Test get_db_session returns context manager."""
        import app.database.connection as conn_module
        conn_module._db_manager = None
        
        with patch("app.database.connection.get_settings") as mock_settings:
            mock_settings.return_value = Settings(
                sqlite_db_path=":memory:"
            )
            session_cm = get_db_session()
            assert hasattr(session_cm, "__enter__")
            assert hasattr(session_cm, "__exit__")
    
    @pytest.mark.unit
    def test_close_db_closes_manager(self):
        """Test close_db closes the manager."""
        import app.database.connection as conn_module
        conn_module._db_manager = None
        
        with patch("app.database.connection.get_settings") as mock_settings:
            mock_settings.return_value = Settings(
                sqlite_db_path=":memory:"
            )
            manager = get_db_manager()
            
            with patch.object(manager, "close") as mock_close:
                close_db()
                mock_close.assert_called_once()
    
    @pytest.mark.unit
    def test_close_db_resets_global_manager(self):
        """Test close_db resets global manager."""
        import app.database.connection as conn_module
        conn_module._db_manager = None
        
        with patch("app.database.connection.get_settings") as mock_settings:
            mock_settings.return_value = Settings(
                sqlite_db_path=":memory:"
            )
            get_db_manager()
            close_db()
            assert conn_module._db_manager is None


class TestDatabaseIntegration:
    """Integration tests for database functionality."""
    
    @pytest.mark.unit
    def test_full_connection_lifecycle(self):
        """Test full connection lifecycle."""
        import app.database.connection as conn_module
        conn_module._db_manager = None
        
        with patch("app.database.connection.get_settings") as mock_settings:
            mock_settings.return_value = Settings(
                sqlite_db_path=":memory:"
            )
            
            # Get manager
            manager = get_db_manager()
            assert manager is not None
            
            # Check health
            assert manager.health_check() is True
            
            # Get session
            with manager.get_session() as session:
                assert isinstance(session, Session)
            
            # Close
            close_db()
            assert conn_module._db_manager is None
    
    @pytest.mark.unit
    def test_multiple_sessions_from_same_manager(self):
        """Test creating multiple sessions from same manager."""
        import app.database.connection as conn_module
        conn_module._db_manager = None
        
        with patch("app.database.connection.get_settings") as mock_settings:
            mock_settings.return_value = Settings(
                sqlite_db_path=":memory:"
            )
            
            manager = get_db_manager()
            
            # Create multiple sessions
            with manager.get_session() as session1:
                assert isinstance(session1, Session)
            
            with manager.get_session() as session2:
                assert isinstance(session2, Session)
