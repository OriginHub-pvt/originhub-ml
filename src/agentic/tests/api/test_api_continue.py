"""
Tests for /api/continue endpoint.
"""

from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from src.agentic.api.main import app


def test_api_continue_accepts_user_input():
    client = TestClient(app)

    resp = client.post("/api/continue", json={
        "session_id": "123",
        "answers": {"problem_statement": "Founders need help"}
    })

    assert resp.status_code == 200
    assert "status" in resp.json()
