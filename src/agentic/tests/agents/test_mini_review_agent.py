"""
Tests for MiniReviewAgent (detects missing JSON fields).
"""

from src.agentic.core.state import State
from src.agentic.agents.mini_review_agent import MiniReviewAgent


def test_mini_review_agent_detects_missing_fields():
    agent = MiniReviewAgent()

    state = State()
    state.update({
        "idea_title": "AI App",
        "problem_statement": None,
        "target_users": "",
        "solution_features": ["x", "y"]
    })

    missing = agent.run(state)
    assert "problem_statement" in missing
    assert "target_users" in missing
    assert "idea_title" not in missing


def test_mini_review_agent_returns_empty_if_none_missing():
    agent = MiniReviewAgent()

    state = State()
    state.update({
        "idea_title": "X",
        "problem_statement": "Y",
        "target_users": "Z"
    })

    missing = agent.run(state)
    assert missing == []
