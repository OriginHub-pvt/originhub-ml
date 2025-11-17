"""
Tests for EvaluatorAgent (schema / consistency validation).
"""

from src.agentic.core.state import State
from src.agentic.agents.evaluator_agent import EvaluatorAgent


def test_evaluator_agent_detects_missing_required_fields():
    agent = EvaluatorAgent()

    state = State()
    state.update({
        "idea_title": "X",
        "problem_statement": None,
        "target_users": "Y"
    })

    errors = agent.run(state)

    assert len(errors) == 1
    assert "problem_statement" in errors[0].lower()


def test_evaluator_agent_returns_empty_on_valid_data():
    agent = EvaluatorAgent()

    state = State()
    state.update({
        "idea_title": "X",
        "problem_statement": "Y",
        "target_users": "Z",
    })

    assert agent.run(state) == []
