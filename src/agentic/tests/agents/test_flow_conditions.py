"""
Tests for flow condition functions.
"""

from src.agentic.pipeline.flow_conditions import (
    needs_clarification,
    needs_strategy,
)


def test_needs_clarification_true():
    state = {"missing_fields": ["problem_statement"]}
    assert needs_clarification(state) is True


def test_needs_clarification_false():
    state = {"missing_fields": []}
    assert needs_clarification(state) is False


def test_needs_strategy_true():
    state = {"rag_results": [], "user_wants_strategy": True}
    assert needs_strategy(state) is True


def test_needs_strategy_false():
    state = {"rag_results": ["something"], "user_wants_strategy": True}
    assert needs_strategy(state) is False
