"""
Tests for InterpreterAgent.

InterpreterAgent:
- Uses heavy model by default
- Builds prompt using prompt_builder.interpreter_prompt()
- Calls inference engine with the prompt
- Parses structured JSON-like output into state.interpreted
- Stores raw output into state.agent_outputs[name]
- Handles JSON parsing errors without crashing
"""

from unittest.mock import MagicMock

from src.agentic.core.state import State
from src.agentic.agents.interpreter_agent import InterpreterAgent


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

class MockPromptBuilder:
    def interpreter_prompt(self, text):
        return f"INTERPRET: {text}"


# -------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------

def test_interpreter_agent_uses_heavy_model():
    """InterpreterAgent should use heavy model by default."""
    mock_engine = MagicMock()
    mock_engine.generate.return_value = '{"title": "AI Startup"}'

    agent = InterpreterAgent(
        inference_engine=mock_engine,
        prompt_builder=MockPromptBuilder(),
    )

    s = State(input_text="AI tool for founders")
    agent.run(s)

    # Expect heavy=True because interpreter is a "big brain" agent
    _, kwargs = mock_engine.generate.call_args
    assert kwargs["heavy"] is True


def test_interpreter_agent_calls_prompt_builder():
    """InterpreterAgent must call interpreter_prompt() to build prompt."""
    mock_engine = MagicMock()
    mock_engine.generate.return_value = '{"idea": "something"}'

    pb = MockPromptBuilder()

    agent = InterpreterAgent(
        inference_engine=mock_engine,
        prompt_builder=pb,
    )

    s = State(input_text="hello")
    agent.run(s)

    # Check last_prompt property inherited from AgentBase
    assert agent.last_prompt == "INTERPRET: hello"


def test_interpreter_agent_updates_state_interpreted():
    """InterpreterAgent should parse and store structured data."""
    mock_engine = MagicMock()
    mock_engine.generate.return_value = '''
    {
        "title": "My Idea",
        "domain": "finance"
    }
    '''

    agent = InterpreterAgent(
        inference_engine=mock_engine,
        prompt_builder=MockPromptBuilder(),
    )

    s = State(input_text="something")
    s = agent.run(s)

    assert s.interpreted["title"] == "My Idea"
    assert s.interpreted["domain"] == "finance"


def test_interpreter_agent_stores_raw_output():
    """Raw output should appear under agent_outputs['InterpreterAgent']."""
    mock_engine = MagicMock()
    mock_engine.generate.return_value = '{"title": "Raw Stuff"}'

    agent = InterpreterAgent(
        inference_engine=mock_engine,
        prompt_builder=MockPromptBuilder(),
    )

    s = State(input_text="test")
    agent.run(s)

    assert s.agent_outputs["InterpreterAgent"] == '{"title": "Raw Stuff"}'


def test_interpreter_agent_handles_bad_json_gracefully():
    """Agent should not crash on malformed JSON."""
    mock_engine = MagicMock()
    mock_engine.generate.return_value = 'not valid json'

    agent = InterpreterAgent(
        inference_engine=mock_engine,
        prompt_builder=MockPromptBuilder(),
    )

    s = State(input_text="broken output")
    s = agent.run(s)

    # Should not crash → state.interpreted should stay None
    assert s.interpreted is None

    # Should store error in agent_outputs
    out = s.agent_outputs["InterpreterAgent"]
    assert "error" in out.lower()
