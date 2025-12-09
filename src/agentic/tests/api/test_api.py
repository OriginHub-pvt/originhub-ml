"""
Integration tests for the FastAPI application.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import os
os.environ['MODEL_PRELOAD'] = 'false'

from src.agentic.api.app import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.mark.skip(reason="Requires model files and Weaviate - run manually")
def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()


@pytest.mark.skip(reason="Requires model files and Weaviate - run manually")
def test_create_session(client):
    """Test session creation."""
    response = client.post("/sessions")
    assert response.status_code == 200
    data = response.json()
    assert "session_id" in data
    assert "created_at" in data


@pytest.mark.skip(reason="Requires model files and Weaviate - run manually")
def test_api_info(client):
    """Test API info endpoint."""
    response = client.get("/info")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "endpoints" in data


def test_models_import():
    """Test that API models can be imported."""
    # This is just to ensure models.py doesn't have syntax errors
    try:
        from src.agentic.api.models import MessageRequest, MessageResponse
        assert MessageRequest is not None
        assert MessageResponse is not None
    except ImportError:
        pytest.skip("pydantic not installed")


def test_session_manager():
    """Test SessionManager functionality."""
    from src.agentic.api.session_manager import SessionManager
    from src.agentic.pipeline.interactive_pipeline_runner import InteractivePipelineRunner
    
    manager = SessionManager()
    
    # Create mock runner
    mock_runner = MagicMock(spec=InteractivePipelineRunner)
    
    # Test session creation
    session_id = manager.create_session(mock_runner)
    assert session_id is not None
    assert len(session_id) > 0
    
    # Test get session
    retrieved = manager.get_session(session_id)
    assert retrieved is not None
    
    # Test message increment
    manager.increment_message(session_id)
    info = manager.get_session_info(session_id)
    assert info["message_count"] == 1
    
    # Test session deletion
    assert manager.delete_session(session_id) is True
    assert manager.get_session(session_id) is None
    
    # Test non-existent session
    assert manager.get_session("nonexistent") is None
    assert manager.delete_session("nonexistent") is False
