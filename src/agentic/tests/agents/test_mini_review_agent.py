"""
Tests for MiniReviewAgent.

MiniReviewAgent responsibilities:
- Uses LIGHT model (heavy=False)
- Builds prompt using prompt_builder.mini_review_prompt(interpreted, rag_results, analysis)
- Calls inference engine with the prompt
- Stores clean output into state.mini_review
- Stores raw output into state.agent_outputs["MiniReviewAgent"]
- Supports plain text or JSON output
- Gracefully handles malformed outputs (no crash)
"""

from unittest.mock import MagicMock
from src.agentic.core.state import State
from src.agentic.agents.mini_review_agent import MiniReviewAgent


class MockPromptBuilder:
    def mini_review_prompt(self, interpreted, rag_results, analysis):
        return f"MINI_REVIEW: {interpreted} | {rag_results} | {analysis}"


# -------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------

def test_mini_review_uses_light_model():
    mock_engine = MagicMock()
    mock_engine.generate.return_value = "ok"

    agent = MiniReviewAgent(
        inference_engine=mock_engine,
        prompt_builder=MockPromptBuilder(),
    )

    s = State()
    s.interpreted = {"title": "AI Tutor"}
    s.rag_results = [{"id": "1"}]
    s.analysis = "Market review here"

    agent.run(s)

    _, kwargs = mock_engine.generate.call_args
    assert kwargs["heavy"] is False


def test_mini_review_calls_prompt_builder_correctly():
    mock_engine = MagicMock()
    mock_engine.generate.return_value = "output text"

    pb = MockPromptBuilder()
    agent = MiniReviewAgent(mock_engine, pb)

    s = State()
    s.interpreted = {"x": 1}
    s.rag_results = [{"id": "123"}]
    s.analysis = "analysis-text"

    agent.run(s)

    assert agent.last_prompt == "MINI_REVIEW: {'x': 1} | [{'id': '123'}] | analysis-text"


def test_mini_review_updates_state_with_plain_text():
    mock_engine = MagicMock()
    mock_engine.generate.return_value = "Your idea differs by X and Y."

    agent = MiniReviewAgent(mock_engine, MockPromptBuilder())

    s = State()
    s.interpreted = {"i": 1}
    s.rag_results = [{"id": "1"}]
    s.analysis = "analysis"

    updated = agent.run(s)

    assert updated.mini_review == "Your idea differs by X and Y."


def test_mini_review_updates_state_with_json_output():
    mock_engine = MagicMock()
    mock_engine.generate.return_value = '''
    {
        "unique_points": ["better UX", "cheaper"],
        "risks": ["crowded market"]
    }
    '''

    agent = MiniReviewAgent(mock_engine, MockPromptBuilder())

    s = State()
    s.interpreted = {"t": 1}
    s.rag_results = [{"id": "a"}]
    s.analysis = "review"

    updated = agent.run(s)

    assert isinstance(updated.mini_review, dict)
    assert "unique_points" in updated.mini_review
    assert updated.mini_review["risks"] == ["crowded market"]


def test_mini_review_stores_raw_output():
    mock_engine = MagicMock()
    mock_engine.generate.return_value = "RAW OUTPUT"

    agent = MiniReviewAgent(mock_engine, MockPromptBuilder())

    s = State()
    s.interpreted = {"x": 1}
    s.rag_results = []
    s.analysis = "r"

    agent.run(s)

    assert s.agent_outputs["MiniReviewAgent"] == "RAW OUTPUT"


def test_mini_review_handles_errors_gracefully():
    mock_engine = MagicMock()
    mock_engine.generate.side_effect = RuntimeError("LLM failed")

    agent = MiniReviewAgent(mock_engine, MockPromptBuilder())

    s = State()
    s.interpreted = {"e": 1}
    s.rag_results = []
    s.analysis = "a"

    updated = agent.run(s)

    # Should not crash
    assert updated.mini_review is None

    # Error recorded
    assert "error" in updated.agent_outputs["MiniReviewAgent"].lower()
