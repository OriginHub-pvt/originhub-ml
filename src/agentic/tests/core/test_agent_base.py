"""
Tests for AgentBase class.
"""

import pytest
from unittest.mock import MagicMock
from src.agentic.core.agent_base import AgentBase
from src.agentic.core.state import State


class DummyAgent(AgentBase):
    """Minimal concrete agent to test AgentBase."""

    def build_prompt(self, state: State) -> str:
        return "PROMPT"

    def process_output(self, output: str, state: State) -> None:
        state.set("processed", output)


def test_agent_base_runs_full_flow():
    """AgentBase.run() should build prompt → generate → process output."""

    fake_engine = MagicMock()
    fake_engine.generate.return_value = "MODEL_OUT"

    agent = DummyAgent(fake_engine)
    state = State()

    agent.run(state)

    # verify call sequence
    fake_engine.generate.assert_called_once_with(
        "PROMPT",
        heavy=False,
        max_tokens=512,
        temperature=0.2,
        top_p=0.95,
        stop=None,
    )

    assert state.get("processed") == "MODEL_OUT"


def test_agent_base_can_use_heavy_model():
    """AgentBase should respect heavy=True flag."""

    fake_engine = MagicMock()
    fake_engine.generate.return_value = "OUT"

    agent = DummyAgent(fake_engine, heavy=True)
    state = State()

    agent.run(state)

    fake_engine.generate.assert_called_once()
    _, kwargs = fake_engine.generate.call_args

    assert kwargs["heavy"] is True
