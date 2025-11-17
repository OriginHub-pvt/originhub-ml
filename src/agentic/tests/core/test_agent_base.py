"""
Tests for AgentBase.

Defines required behavior for all agents:
- Requires name, inference_engine, and prompt_builder
- Must implement build_prompt()
- run() must call build_prompt and generate()
- run() must return the same State instance (mutated inline)
- Writes to state.agent_outputs[name]
- Properly stores last_prompt and last_output
- Errors should not crash; instead stored as error
"""

from unittest.mock import MagicMock

from src.agentic.core.state import State
from src.agentic.core.agent_base import AgentBase


# Dummy subclasses for testing
class DummyAgent(AgentBase):
    def build_prompt(self, state: State) -> str:
        return f"PROMPT: {state.input_text}"


class ErrorAgent(AgentBase):
    def build_prompt(self, state: State) -> str:
        raise RuntimeError("boom")


# --------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------

def test_agent_base_initializes_properly():
    mock_engine = MagicMock()
    mock_pb = MagicMock()

    agent = DummyAgent(
        name="Interpreter",
        inference_engine=mock_engine,
        prompt_builder=mock_pb,
    )

    assert agent.name == "Interpreter"
    assert agent.engine is mock_engine
    assert agent.prompt_builder is mock_pb


def test_agent_base_calls_build_prompt():
    mock_engine = MagicMock()
    mock_pb = MagicMock()

    agent = DummyAgent(
        name="Interpreter",
        inference_engine=mock_engine,
        prompt_builder=mock_pb,
    )

    state = State(input_text="hello")
    agent.run(state)

    assert agent.last_prompt == "PROMPT: hello"


def test_agent_base_calls_inference_engine():
    mock_engine = MagicMock()
    mock_engine.generate.return_value = "agent output"

    agent = DummyAgent(
        name="TestAgent",
        inference_engine=mock_engine,
        prompt_builder=MagicMock(),
        heavy=False,
    )

    state = State(input_text="testing")
    agent.run(state)

    mock_engine.generate.assert_called_once()
    _, kwargs = mock_engine.generate.call_args

    assert kwargs["prompt"] == "PROMPT: testing"
    assert kwargs["heavy"] is False


def test_agent_run_updates_state_outputs():
    mock_engine = MagicMock()
    mock_engine.generate.return_value = "out"

    agent = DummyAgent(
        name="MyAgent",
        inference_engine=mock_engine,
        prompt_builder=MagicMock(),
    )

    s = State(input_text="x")
    agent.run(s)

    assert s.agent_outputs["MyAgent"] == "out"


def test_agent_run_returns_same_state():
    mock_engine = MagicMock()
    mock_engine.generate.return_value = "hi"

    agent = DummyAgent(
        name="A",
        inference_engine=mock_engine,
        prompt_builder=MagicMock(),
    )

    s = State(input_text="q")
    returned = agent.run(s)
    assert returned is s


def test_agent_tracks_last_prompt_and_last_output():
    mock_engine = MagicMock()
    mock_engine.generate.return_value = "ok"

    agent = DummyAgent(
        name="TrackAgent",
        inference_engine=mock_engine,
        prompt_builder=MagicMock(),
    )

    s = State(input_text="hello")
    agent.run(s)

    assert agent.last_prompt == "PROMPT: hello"
    assert agent.last_output == "ok"


def test_agent_handles_errors_without_crashing():
    mock_engine = MagicMock()

    agent = ErrorAgent(
        name="BadAgent",
        inference_engine=mock_engine,
        prompt_builder=MagicMock(),
    )

    s = State(input_text="hello")
    out = agent.run(s)

    assert isinstance(out, State)
    assert "error" in out.agent_outputs["BadAgent"]
