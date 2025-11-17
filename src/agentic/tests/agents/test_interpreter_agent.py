"""
Tests for InterpreterAgent.
"""

import pytest
from unittest.mock import MagicMock
from src.agentic.core.state import State
from src.agentic.agents.interpreter_agent import InterpreterAgent


def test_interpreter_agent_builds_correct_prompt():
    """InterpreterAgent should build JSON-extraction prompt."""

    fake_engine = MagicMock()
    agent = InterpreterAgent(fake_engine)

    state = State()
    state.set("raw_text", "AI that helps founders test ideas")

    prompt = agent.build_prompt(state)

    assert "Extract structured JSON" in prompt
    assert "AI that helps founders test ideas" in prompt
    assert "{\n  \"idea_title\"" in prompt


def test_interpreter_agent_processes_valid_json_output():
    """InterpreterAgent should parse output JSON and write into state."""

    fake_engine = MagicMock()
    agent = InterpreterAgent(fake_engine)

    state = State()

    json_output = """
    {
      "idea_title": "AI Founder Assistant",
      "problem_statement": "Founders can't validate ideas quickly."
    }
    """

    agent.process_output(json_output, state)

    assert state.get("idea_title") == "AI Founder Assistant"
    assert state.get("problem_statement") == "Founders can't validate ideas quickly."


def test_interpreter_agent_handles_invalid_json_without_crashing():
    """InterpreterAgent should not crash on invalid output; stores errors."""

    fake_engine = MagicMock()
    agent = InterpreterAgent(fake_engine)

    state = State()

    bad_output = "NOT JSON AT ALL"

    agent.process_output(bad_output, state)

    assert state.get("parse_error") is True
    assert state.get("idea_title") is None
