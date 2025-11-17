"""
Tests for ReviewerAgent.
"""

from unittest.mock import MagicMock
from src.agentic.core.state import State
from src.agentic.agents.reviewer_agent import ReviewerAgent


def test_reviewer_agent_builds_prompt():
    fake_engine = MagicMock()
    agent = ReviewerAgent(fake_engine)

    state = State()
    state.set("raw_text", "AI to summarize lectures")

    prompt = agent.build_prompt(state)

    assert "Review this idea" in prompt


def test_reviewer_agent_saves_review_to_state():
    fake_engine = MagicMock()
    fake_engine.generate.return_value = "Score: 8/10"

    state = State()
    agent = ReviewerAgent(fake_engine)
    agent.run(state)

    assert state.get("review") == "Score: 8/10"
