"""
Tests for /api/start endpoint.
"""

from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from src.agentic.api.main import app


def test_api_start_initializes_pipeline():
    client = TestClient(app)

    resp = client.post("/api/start", json={"text": "AI for founders"})

    assert resp.status_code == 200
    assert "session_id" in resp.json()
