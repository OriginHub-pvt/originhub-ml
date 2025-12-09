"""
Tests for StrategistAgent.

StrategistAgent responsibilities:
- Uses HEAVY model (heavy=True)
- Builds prompt using prompt_builder.strategy_prompt(interpreted)
- Calls inference engine with the heavy model
- Stores model output into state.strategy
- Stores raw output into state.agent_outputs["StrategistAgent"]
- Supports both plain text output and JSON output
- Gracefully handles LLM errors (no crash)
"""

import pytest
from unittest.mock import MagicMock

from src.agentic.core.state import State
from src.agentic.agents.strategist_agent import StrategistAgent


class MockPromptBuilder:
    def strategy_prompt(self, interpreted):
        return f"STRATEGY: {interpreted}"
        

# -------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------

def test_strategist_uses_heavy_model_by_default():
    mock_engine = MagicMock()
    mock_engine.generate.return_value = "SWOT text"

    agent = StrategistAgent(
        inference_engine=mock_engine,
        prompt_builder=MockPromptBuilder(),
    )

    s = State()
    s.interpreted = {"title": "New AI startup idea"}
    s.is_new_idea = True

    agent.run(s)

    _, kwargs = mock_engine.generate.call_args
    assert kwargs["heavy"] is False  # MUST use heavy model


def test_strategist_calls_prompt_builder_correctly():
    mock_engine = MagicMock()
    mock_engine.generate.return_value = "Some strategy here."

    pb = MockPromptBuilder()
    agent = StrategistAgent(mock_engine, pb)

    s = State()
    s.interpreted = {"domain": "finance"}
    s.is_new_idea = True

    agent.run(s)

    assert agent.last_prompt == "STRATEGY: {'domain': 'finance'}"


def test_strategist_updates_state_strategy_with_plain_text():
    mock_engine = MagicMock()
    mock_engine.generate.return_value = "Strengths, weaknesses, opportunities, threats."

    agent = StrategistAgent(
        inference_engine=mock_engine,
        prompt_builder=MockPromptBuilder()
    )

    s = State()
    s.interpreted = {"title": "New idea"}
    s.is_new_idea = True

    updated = agent.run(s)

    assert updated.strategy == "Strengths, weaknesses, opportunities, threats."


def test_strategist_updates_state_strategy_with_json_output():
    mock_engine = MagicMock()
    mock_engine.generate.return_value = '''
    {
        "strengths": ["unique angle", "low competition"],
        "weaknesses": ["needs data"],
        "opportunities": ["market gap"],
        "threats": ["big tech"]
    }
    '''

    agent = StrategistAgent(mock_engine, MockPromptBuilder())

    s = State()
    s.interpreted = {"title": "Idea"}
    s.is_new_idea = True

    updated = agent.run(s)

    assert isinstance(updated.strategy, dict)
    assert "strengths" in updated.strategy
    assert updated.strategy["opportunities"] == ["market gap"]


def test_strategist_stores_raw_output():
    mock_engine = MagicMock()
    mock_engine.generate.return_value = "RAW STRATEGY TEXT"

    agent = StrategistAgent(mock_engine, MockPromptBuilder())

    s = State()
    s.interpreted = {"x": 1}
    s.is_new_idea = True

    agent.run(s)

    assert s.agent_outputs["StrategistAgent"] == "RAW STRATEGY TEXT"


def test_strategist_handles_errors_gracefully():
    mock_engine = MagicMock()
    mock_engine.generate.side_effect = RuntimeError("model failed")

    agent = StrategistAgent(mock_engine, MockPromptBuilder())

    s = State()
    s.interpreted = {"x": 1}
    s.is_new_idea = True

    updated = agent.run(s)

    # Should not crash and strategy remains None
    assert updated.strategy is None

    # Error is recorded
    out = updated.agent_outputs["StrategistAgent"]
    assert "error" in out.lower()
