"""
Tests for ClarifierAgent.

ClarifierAgent behavior:
- Uses the light model by default (heavy=False)
- Builds prompt using prompt_builder.clarifier_prompt()
- Calls inference engine with that prompt
- Parses LLM output into a list of clarifying questions
- Appends questions to state.clarifications
- Sets state.need_more_clarification = True if any questions exist
- Handles malformed / non-list LLM outputs gracefully
- Raw output must be stored in agent_outputs["ClarifierAgent"]
"""

from unittest.mock import MagicMock

from src.agentic.core.state import State
from src.agentic.agents.clarifier_agent import ClarifierAgent


# Helper mock prompt builder
class MockPromptBuilder:
    def clarifier_prompt(self, interpreted):
        return f"CLARIFY: {interpreted}"


# -------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------

def test_clarifier_uses_light_model_by_default():
    mock_engine = MagicMock()
    mock_engine.generate.return_value = '["Q1", "Q2"]'

    agent = ClarifierAgent(
        inference_engine=mock_engine,
        prompt_builder=MockPromptBuilder(),
    )

    s = State()
    s.interpreted = {"title": "AI tool"}

    agent.run(s)

    _, kwargs = mock_engine.generate.call_args
    assert kwargs["heavy"] is False   # uses light model


def test_clarifier_calls_prompt_builder_correctly():
    mock_engine = MagicMock()
    mock_engine.generate.return_value = '["Q"]'

    pb = MockPromptBuilder()
    agent = ClarifierAgent(mock_engine, pb)

    s = State()
    s.interpreted = {"domain": "health"}

    agent.run(s)

    assert agent.last_prompt == "CLARIFY: {'domain': 'health'}"


def test_clarifier_parses_list_of_questions():
    mock_engine = MagicMock()
    mock_engine.generate.return_value = '["What is the target audience?","What problem is this solving?"]'

    agent = ClarifierAgent(
        inference_engine=mock_engine,
        prompt_builder=MockPromptBuilder(),
    )

    s = State(input_text="hello")
    s.interpreted = {"title": "Something"}

    updated = agent.run(s)

    assert len(updated.clarifications) == 2
    assert updated.clarifications[0].startswith("What is")
    assert updated.need_more_clarification is True


def test_clarifier_handles_empty_list():
    mock_engine = MagicMock()
    mock_engine.generate.return_value = '[]'

    agent = ClarifierAgent(mock_engine, MockPromptBuilder())

    s = State()
    s.interpreted = {"title": "Something"}

    updated = agent.run(s)

    assert updated.clarifications == []
    assert updated.need_more_clarification is False


def test_clarifier_handles_malformed_output_gracefully():
    mock_engine = MagicMock()
    mock_engine.generate.return_value = "not a list"

    agent = ClarifierAgent(mock_engine, MockPromptBuilder())

    s = State()
    s.interpreted = {"x": 1}

    updated = agent.run(s)

    # No clarifications added
    assert updated.clarifications == []
    assert updated.need_more_clarification is False

    # Error stored in agent_outputs
    assert "error" in updated.agent_outputs["ClarifierAgent"].lower()


def test_clarifier_stores_raw_output():
    mock_engine = MagicMock()
    mock_engine.generate.return_value = '["Clarify description"]'

    agent = ClarifierAgent(
        inference_engine=mock_engine,
        prompt_builder=MockPromptBuilder(),
    )

    s = State()
    s.interpreted = {"title": "Something"}

    agent.run(s)

    assert s.agent_outputs["ClarifierAgent"] == '["Clarify description"]'