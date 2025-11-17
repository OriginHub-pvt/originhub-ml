"""
Tests for ClarifierAgent (HITL).
"""

from src.agentic.core.state import State
from src.agentic.agents.clarifier_agent import ClarifierAgent


def test_clarifier_agent_generates_questions():
    agent = ClarifierAgent()

    state = State()
    missing = ["problem_statement", "target_users"]

    qs = agent.run(state, missing)

    assert len(qs) == 2
    assert "problem_statement" in qs[0].lower()
    assert "target_users" in qs[1].lower()


def test_clarifier_agent_updates_state_with_user_responses():
    agent = ClarifierAgent()
    state = State()

    user_answers = {
        "problem_statement": "Founders waste time validating ideas.",
        "target_users": "Early-stage founders."
    }

    agent.apply_user_responses(state, user_answers)

    assert state.get("problem_statement") == "Founders waste time validating ideas."
    assert state.get("target_users") == "Early-stage founders."
