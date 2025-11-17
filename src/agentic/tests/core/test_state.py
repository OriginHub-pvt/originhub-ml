"""
Tests for the State object (shared agent state).
"""

import pytest
from src.agentic.core.state import State


def test_state_initializes_empty():
    """State should initialize with empty internal data dict."""
    s = State()
    assert isinstance(s.data, dict)
    assert len(s.data) == 0


def test_state_get_set_values():
    """State should support value assignment and retrieval."""
    s = State()
    s.set("idea_title", "AI Planner")
    assert s.get("idea_title") == "AI Planner"


def test_state_update_merges_dicts():
    """update() should merge dictionaries into internal state."""
    s = State()
    s.update({"a": 1, "b": 2})
    assert s.get("a") == 1
    assert s.get("b") == 2


def test_state_missing_key_returns_none():
    """get() should return None for missing keys."""
    s = State()
    assert s.get("missing") is None


def test_state_clear_removes_all_keys():
    """clear() should remove all stored keys."""
    s = State()
    s.update({"a": 1, "b": 2})
    s.clear()
    assert len(s.data) == 0
