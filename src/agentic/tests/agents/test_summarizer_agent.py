"""
Tests for SummarizerAgent.
"""

from unittest.mock import MagicMock
from src.agentic.core.state import State
from src.agentic.agents.summarizer_agent import SummarizerAgent


def test_summarizer_agent_builds_prompt():
    fake_engine = MagicMock()
    agent = SummarizerAgent(fake_engine)

    state = State()
    state.set("raw_text", "AI for mental health assistants.")

    prompt = agent.build_prompt(state)

    assert "Summarize this idea" in prompt
    assert "mental health assistants" in prompt


def test_summarizer_agent_saves_summary_to_state():
    fake_engine = MagicMock()
    fake_engine.generate.return_value = "- Bullet 1\n- Bullet 2"

    state = State()

    agent = SummarizerAgent(fake_engine)
    agent.run(state)

    assert state.get("summary") == "- Bullet 1\n- Bullet 2"
