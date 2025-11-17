"""
Tests for RAGAgent.
"""

from unittest.mock import MagicMock
from src.agentic.core.state import State
from src.agentic.agents.rag_agent import RAGAgent


def test_rag_agent_calls_search_with_correct_query():
    fake_rag = MagicMock()
    fake_rag.search.return_value = [{"text": "existing idea"}]

    state = State()
    state.set("idea_title", "AI for founders")

    agent = RAGAgent(fake_rag)

    results = agent.run(state)

    assert results[0]["text"] == "existing idea"
    fake_rag.search.assert_called_once()


def test_rag_agent_updates_state():
    fake_rag = MagicMock()
    fake_rag.search.return_value = [{"text": "match"}]

    state = State()
    agent = RAGAgent(fake_rag)

    agent.run(state)

    assert state.get("rag_results")[0]["text"] == "match"
