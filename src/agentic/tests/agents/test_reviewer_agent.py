"""
Tests for ReviewerAgent.

ReviewerAgent responsibilities:
- Uses LIGHT model (heavy=False)
- Builds prompt using prompt_builder.review_prompt(interpreted, rag_results)
- Calls inference engine with that prompt
- Stores parsed model output into state.analysis
- Stores raw output into state.agent_outputs["ReviewerAgent"]
- Gracefully handles LLM errors (no crash)
"""

from unittest.mock import MagicMock

from src.agentic.core.state import State
from src.agentic.agents.reviewer_agent import ReviewerAgent


class MockPromptBuilder:
    def review_prompt(self, interpreted, rag_results):
        return f"REVIEW: {interpreted} || {rag_results}"


# -------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------

def test_reviewer_uses_light_model_by_default():
    mock_engine = MagicMock()
    mock_engine.generate.return_value = "ok"

    agent = ReviewerAgent(
        inference_engine=mock_engine,
        prompt_builder=MockPromptBuilder(),
    )

    s = State()
    s.interpreted = {"title": "AI assistant"}
    s.rag_results = [{"id": "1"}]

    agent.run(s)

    _, kwargs = mock_engine.generate.call_args
    assert kwargs["heavy"] is False  # must use light model


def test_reviewer_calls_prompt_builder_correctly():
    mock_engine = MagicMock()
    mock_engine.generate.return_value = "analysis abc"

    pb = MockPromptBuilder()
    agent = ReviewerAgent(mock_engine, pb)

    s = State()
    s.interpreted = {"title": "Idea"}
    s.rag_results = [{"id": "123"}]

    agent.run(s)

    assert agent.last_prompt == "REVIEW: {'title': 'Idea'} || [{'id': '123'}]"


def test_reviewer_updates_state_analysis():
    mock_engine = MagicMock()
    mock_engine.generate.return_value = "This is your market analysis."

    agent = ReviewerAgent(mock_engine, MockPromptBuilder())

    s = State()
    s.interpreted = {"title": "My Startup"}
    s.rag_results = [{"id": "existing"}]

    updated = agent.run(s)

    assert updated.analysis == "This is your market analysis."


def test_reviewer_stores_raw_output():
    mock_engine = MagicMock()
    mock_engine.generate.return_value = "RAW TEXT"

    agent = ReviewerAgent(mock_engine, MockPromptBuilder())

    s = State()
    s.interpreted = {"x": 1}
    s.rag_results = [{"id": "abc"}]

    agent.run(s)

    assert s.agent_outputs["ReviewerAgent"] == "RAW TEXT"


def test_reviewer_handles_errors_gracefully():
    mock_engine = MagicMock()
    mock_engine.generate.side_effect = RuntimeError("llm failed")

    agent = ReviewerAgent(mock_engine, MockPromptBuilder())

    s = State()
    s.interpreted = {"x": 1}
    s.rag_results = [{"id": "abc"}]

    updated = agent.run(s)

    # No crash → analysis stays None
    assert updated.analysis is None

    # Error must be recorded
    out = updated.agent_outputs["ReviewerAgent"]
    assert "error" in out.lower()