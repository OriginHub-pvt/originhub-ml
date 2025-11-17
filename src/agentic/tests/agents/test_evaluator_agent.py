"""
Tests for EvaluatorAgent.

EvaluatorAgent responsibilities:
- Decides the next step in the pipeline based on the State:
    * If state.need_more_clarification == True → "clarify"
    * Else if state.is_new_idea == True → "strategize"
    * Else → "review"
- Writes decision to state.next_action
- Writes decision to state.agent_outputs["EvaluatorAgent"]
- Never uses the inference engine (no LLM call)
- Never crashes
"""

from src.agentic.core.state import State
from src.agentic.agents.evaluator_agent import EvaluatorAgent


# -------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------

def test_evaluator_decides_clarify_when_needed():
    agent = EvaluatorAgent()
    s = State()
    s.need_more_clarification = True

    updated = agent.run(s)

    assert updated.next_action == "clarify"
    assert updated.agent_outputs["EvaluatorAgent"] == "clarify"
    assert agent.last_decision == "clarify"


def test_evaluator_decides_strategize_when_new_idea():
    agent = EvaluatorAgent()
    s = State()
    s.need_more_clarification = False
    s.is_new_idea = True

    updated = agent.run(s)

    assert updated.next_action == "strategize"
    assert updated.agent_outputs["EvaluatorAgent"] == "strategize"
    assert agent.last_decision == "strategize"


def test_evaluator_decides_review_when_existing_idea():
    agent = EvaluatorAgent()
    s = State()
    s.need_more_clarification = False
    s.is_new_idea = False

    updated = agent.run(s)

    assert updated.next_action == "review"
    assert updated.agent_outputs["EvaluatorAgent"] == "review"
    assert agent.last_decision == "review"


def test_evaluator_never_crashes_if_state_missing_fields():
    agent = EvaluatorAgent()
    s = State()  # default flags: False

    updated = agent.run(s)

    assert updated.next_action == "review"   # default path
    assert updated.agent_outputs["EvaluatorAgent"] == "review"


def test_evaluator_updates_same_state_not_new_instance():
    agent = EvaluatorAgent()
    s = State()

    result = agent.run(s)
    assert result is s  # must return same state instance
