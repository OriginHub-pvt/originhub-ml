"""
Tests for StrategistAgent.
"""

from unittest.mock import MagicMock
from src.agentic.core.state import State
from src.agentic.agents.strategist_agent import StrategistAgent


def test_strategist_agent_builds_prompt():
    fake_engine = MagicMock()
    agent = StrategistAgent(fake_engine)

    state = State()
    state.set("raw_text", "AI for sales forecasting")

    prompt = agent.build_prompt(state)

    assert "SWOT" in prompt
    assert "sales forecasting" in prompt


def test_strategist_agent_saves_strategy_to_state():
    fake_engine = MagicMock()
    fake_engine.generate.return_value = "Strengths: ..."

    state = State()
    agent = StrategistAgent(fake_engine)

    agent.run(state)

    assert state.get("strategy") == "Strengths: ..."
