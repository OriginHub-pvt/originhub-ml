"""
Tests for SummarizerAgent.

SummarizerAgent responsibilities:
- Uses light model (heavy=False)
- Builds prompt using prompt_builder.summarizer_prompt()
- Calls inference engine with that prompt
- Produces final user-facing summary under state.summary
- Stores raw output under state.agent_outputs["SummarizerAgent"]
- Supports both text and JSON output
- Handles malformed output gracefully
- Never crashes
"""

from unittest.mock import MagicMock

from src.agentic.core.state import State
from src.agentic.agents.summarizer_agent import SummarizerAgent


class MockPromptBuilder:
    def summarizer_prompt(self, interpreted, analysis, mini_review, strategy):
        return f"SUMMARY: {interpreted} | {analysis} | {mini_review} | {strategy}"


# -------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------

def test_summarizer_uses_light_model():
    mock_engine = MagicMock()
    mock_engine.generate.return_value = "ok"

    agent = SummarizerAgent(
        inference_engine=mock_engine,
        prompt_builder=MockPromptBuilder(),
    )

    s = State()
    s.interpreted = {"title": "Idea"}
    s.analysis = "Market review"
    s.mini_review = "Mini compare"
    s.strategy = "Strategy text"

    agent.run(s)

    _, kwargs = mock_engine.generate.call_args
    assert kwargs["heavy"] is False


def test_summarizer_calls_prompt_builder_correctly():
    mock_engine = MagicMock()
    mock_engine.generate.return_value = "summary"

    pb = MockPromptBuilder()
    agent = SummarizerAgent(mock_engine, pb)

    s = State()
    s.interpreted = {"t": 1}
    s.analysis = "A"
    s.mini_review = "B"
    s.strategy = "C"

    agent.run(s)

    assert agent.last_prompt == "SUMMARY: {'t': 1} | A | B | C"


def test_summarizer_updates_state_with_plain_text():
    mock_engine = MagicMock()
    mock_engine.generate.return_value = "Final summary."

    agent = SummarizerAgent(mock_engine, MockPromptBuilder())

    s = State()
    s.interpreted = {"i": 1}
    s.analysis = "review1"
    s.mini_review = "review2"
    s.strategy = "plan"

    updated = agent.run(s)

    assert updated.summary == "Final summary."


def test_summarizer_updates_state_with_json_output():
    mock_engine = MagicMock()
    mock_engine.generate.return_value = '''
    {
        "final_summary": "You should build this.",
        "next_steps": ["validate with users", "create MVP"]
    }
    '''

    agent = SummarizerAgent(mock_engine, MockPromptBuilder())

    s = State()
    s.interpreted = {"a": 1}
    s.analysis = "R"
    s.mini_review = "M"
    s.strategy = None

    updated = agent.run(s)

    assert isinstance(updated.summary, dict)
    assert updated.summary["final_summary"] == "You should build this."


def test_summarizer_stores_raw_output():
    mock_engine = MagicMock()
    mock_engine.generate.return_value = "RAW FINAL SUMMARY"

    agent = SummarizerAgent(mock_engine, MockPromptBuilder())

    s = State()
    s.interpreted = {"x": 1}

    agent.run(s)

    assert s.agent_outputs["SummarizerAgent"] == "RAW FINAL SUMMARY"


def test_summarizer_handles_llm_errors_gracefully():
    mock_engine = MagicMock()
    mock_engine.generate.side_effect = RuntimeError("LLM failed")

    agent = SummarizerAgent(mock_engine, MockPromptBuilder())

    s = State()
    s.interpreted = {"err": True}

    updated = agent.run(s)

    assert updated.summary is None
    assert "error" in updated.agent_outputs["SummarizerAgent"].lower()
