"""
Tests for RAGAgent.

RAGAgent responsibilities:
- Calls vector store search with interpreted JSON fields
- Stores retrieved results into state.rag_results
- Sets state.is_new_idea = True if no results
- Leaves is_new_idea = False if matches exist
- Does NOT call inference engine (no LLM needed)
- Stores raw payload (retrieved JSON) under agent_outputs["RAGAgent"]
- Should gracefully handle exceptions (connection issues, etc.)
"""

import pytest
from unittest.mock import MagicMock

from src.agentic.core.state import State
from src.agentic.agents.rag_agent import RAGAgent


# -------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------

def test_rag_agent_calls_vector_store():
    """RAGAgent must call vector_db.search() with interpreted fields."""
    mock_db = MagicMock()
    mock_db.search.return_value = [{"obj": 1}]

    agent = RAGAgent(vector_db=mock_db)

    s = State()
    s.interpreted = {
        "title": "AI tutor",
        "problem": "students struggle",
        "domain": "education"
    }

    agent.run(s)

    # Should call vector store search exactly once
    mock_db.search.assert_called_once()

    args, kwargs = mock_db.search.call_args
    assert "AI tutor" in kwargs["query_text"]


def test_rag_agent_stores_results_into_state():
    mock_db = MagicMock()
    mock_db.search.return_value = [{"id": "123", "distance": 0.1}]

    agent = RAGAgent(vector_db=mock_db)

    s = State()
    s.interpreted = {"title": "AI mentor"}

    updated = agent.run(s)

    assert len(updated.rag_results) == 1
    assert updated.rag_results[0]["id"] == "123"
    assert updated.rag_results[0]["score"] == 0.92
    assert updated.is_new_idea is False


def test_rag_agent_marks_new_idea_if_no_results():
    mock_db = MagicMock()
    mock_db.search.return_value = []   # No matches found

    agent = RAGAgent(vector_db=mock_db)

    s = State()
    s.interpreted = {"title": "Quantum blockchain toaster"}

    updated = agent.run(s)

    assert updated.rag_results == []
    assert updated.is_new_idea is True


def test_rag_agent_treats_far_matches_as_new_idea(monkeypatch):
    """
    If retrieved matches have high distance values above the threshold, they
    should be considered not similar -> treat as new idea.
    """
    mock_db = MagicMock()
    mock_db.search.return_value = [{"id": "123", "distance": 0.8}]

    # Use default threshold 0.35 -> 0.8 should be considered new
    agent = RAGAgent(vector_db=mock_db)

    s = State()
    s.interpreted = {"title": "Novel hardware approach"}

    updated = agent.run(s)

    assert len(updated.rag_results) == 1
    assert updated.rag_results[0]["id"] == "123"
    assert updated.rag_results[0]["distance"] == 0.8
    assert updated.is_new_idea is True


def test_rag_agent_treats_close_matches_as_not_new(monkeypatch):
    """
    Low distance values should be considered similar and mark is_new_idea False.
    """
    mock_db = MagicMock()
    mock_db.search.return_value = [{"id": "123", "distance": 0.05}]

    agent = RAGAgent(vector_db=mock_db)

    s = State()
    s.interpreted = {"title": "Incremental feature"}

    updated = agent.run(s)

    assert len(updated.rag_results) == 1
    assert updated.rag_results[0]["id"] == "123"
    assert updated.rag_results[0]["distance"] == 0.05
    assert updated.is_new_idea is False


def test_rag_agent_stores_raw_payload_in_agent_outputs():
    mock_db = MagicMock()
    mock_db.search.return_value = [{"x": 1}]

    agent = RAGAgent(vector_db=mock_db)

    s = State()
    s.interpreted = {"title": "Test idea"}

    agent.run(s)

    # Raw stored as str(...) for debugging and UI rendering
    assert s.agent_outputs["RAGAgent"] == str([{"x": 1}])


def test_rag_agent_handles_search_errors_gracefully():
    mock_db = MagicMock()
    mock_db.search.side_effect = RuntimeError("connection failed")

    agent = RAGAgent(vector_db=mock_db)

    s = State()
    s.interpreted = {"title": "Error test"}

    updated = agent.run(s)

    # On exception: no crash
    assert updated.rag_results == []
    assert updated.is_new_idea is False

    # Error recorded in agent outputs
    assert "error" in updated.agent_outputs["RAGAgent"].lower()