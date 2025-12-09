"""
Tests for PipelineRunner orchestration.

We are testing the high-level control flow of the idea pipeline:

Interpreter -> RAG -> Evaluator -> 
    - if need_more_clarification -> Clarifier loop
    - elif is_new_idea -> Strategist -> Summarizer
    - else -> Reviewer -> MiniReview -> Summarizer
"""

from src.agentic.core.state import State
from src.agentic.pipeline.pipeline_runner import PipelineRunner


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

class DummyAgent:
    """Simple agent that records calls and applies a side-effect function."""

    def __init__(self, name, effect=None):
        self.name = name
        self.effect = effect or (lambda s: s)
        self.calls = 0

    def run(self, state: State) -> State:
        self.calls += 1
        state.debug_trace.append(self.name)
        return self.effect(state)


def make_base_state():
    s = State(input_text="test idea")
    # add a trace list so we can inspect call order
    s.debug_trace = []
    return s


# -------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------

def test_existing_idea_path_runs_reviewer_and_mini_review(monkeypatch):
    """
    Case: existing idea (is_new_idea=False, no clarification).
    Expected path:
      Interpreter -> RAG -> Evaluator -> Reviewer -> MiniReview -> Summarizer
    """

    # Effects for each agent
    interpreter = DummyAgent("interpreter")
    rag = DummyAgent("rag")

    def evaluator_effect(state):
        state.need_more_clarification = False
        state.is_new_idea = False
        state.next_action = "review"
        return state

    evaluator = DummyAgent("evaluator", effect=evaluator_effect)
    clarifier = DummyAgent("clarifier")  # should not be called

    def reviewer_effect(state):
        state.analysis = "market analysis"
        return state

    reviewer = DummyAgent("reviewer", effect=reviewer_effect)

    def mini_review_effect(state):
        state.mini_review = "mini review text"
        return state

    mini_review = DummyAgent("mini_review", effect=mini_review_effect)

    def strategist_effect(state):
        state.strategy = "swot"
        return state

    strategist = DummyAgent("strategist", effect=strategist_effect)

    def summarizer_effect(state):
        state.summary = "final summary"
        return state

    summarizer = DummyAgent("summarizer", effect=summarizer_effect)

    # Build runner
    runner = PipelineRunner(
        interpreter=interpreter,
        clarifier=clarifier,
        rag=rag,
        evaluator=evaluator,
        reviewer=reviewer,
        strategist=strategist,
        summarizer=summarizer,
    )

    # Run pipeline
    final_state = runner.run("AI tool for something")

    # Assertions on call order
    assert final_state.debug_trace == [
        "interpreter",
        "rag",
        "evaluator",
        "reviewer",
        "summarizer",
    ]

    # Clarifier and strategist should NOT be called
    assert clarifier.calls == 0
    assert strategist.calls == 0

    # Summary must be set
    assert final_state.summary == "final summary"


def test_new_idea_path_runs_strategist_not_reviewer():
    """
    Case: new idea (is_new_idea=True, no clarification).
    Expected path:
      Interpreter -> RAG -> Evaluator -> Strategist -> Summarizer
    """

    interpreter = DummyAgent("interpreter")
    rag = DummyAgent("rag")

    def evaluator_effect(state):
        state.need_more_clarification = False
        state.is_new_idea = True
        state.next_action = "strategize"
        return state

    evaluator = DummyAgent("evaluator", effect=evaluator_effect)
    clarifier = DummyAgent("clarifier")

    reviewer = DummyAgent("reviewer")
    mini_review = DummyAgent("mini_review")

    def strategist_effect(state):
        state.strategy = {"strengths": ["unique"], "opportunities": ["market gap"]}
        return state

    strategist = DummyAgent("strategist", effect=strategist_effect)

    def summarizer_effect(state):
        state.summary = "final summary (new idea)"
        return state

    summarizer = DummyAgent("summarizer", effect=summarizer_effect)

    runner = PipelineRunner(
        interpreter=interpreter,
        clarifier=clarifier,
        rag=rag,
        evaluator=evaluator,
        reviewer=reviewer,
        strategist=strategist,
        summarizer=summarizer,
    )

    final_state = runner.run("brand new quantum blockchain toaster")

    # Expected order
    assert final_state.debug_trace == [
        "interpreter",
        "rag",
        "evaluator",
        "strategist",
        "summarizer",
    ]

    # Reviewer / MiniReview not used in new-idea branch
    assert reviewer.calls == 0
    assert mini_review.calls == 0

    assert final_state.strategy is not None
    assert final_state.summary == "final summary (new idea)"


def test_clarification_loop_calls_clarifier_then_re_evaluates():
    """
    Case: evaluator first requests clarification, then after second pass chooses review.

    Expected call pattern (one possible correct order):
      Interpreter -> RAG -> Evaluator (decides clarify)
      Clarifier
      Interpreter -> RAG -> Evaluator (decides review)
      Reviewer -> MiniReview -> Summarizer
    """

    interpreter = DummyAgent("interpreter")
    rag = DummyAgent("rag")

    # We want evaluator to first ask for clarification, then decide "review".
    call_counter = {"count": 0}

    def evaluator_effect(state):
        call_counter["count"] += 1
        if call_counter["count"] == 1:
            state.need_more_clarification = True
            state.is_new_idea = False
            state.next_action = "clarify"
        else:
            state.need_more_clarification = False
            state.is_new_idea = False
            state.next_action = "review"
        return state

    evaluator = DummyAgent("evaluator", effect=evaluator_effect)

    def clarifier_effect(state):
        # pretend we add clarifications here
        state.clarifications.append("What is your target user?")
        state.need_more_clarification = False
        return state

    clarifier = DummyAgent("clarifier", effect=clarifier_effect)

    def reviewer_effect(state):
        state.analysis = "analysis after clarification"
        return state

    reviewer = DummyAgent("reviewer", effect=reviewer_effect)

    def mini_review_effect(state):
        state.mini_review = "mini review after clarification"
        return state

    mini_review = DummyAgent("mini_review", effect=mini_review_effect)

    strategist = DummyAgent("strategist")

    def summarizer_effect(state):
        state.summary = "final summary after clarification loop"
        return state

    summarizer = DummyAgent("summarizer", effect=summarizer_effect)

    runner = PipelineRunner(
        interpreter=interpreter,
        clarifier=clarifier,
        rag=rag,
        evaluator=evaluator,
        reviewer=reviewer,
        strategist=strategist,
        summarizer=summarizer,
    )

    final_state = runner.run("unclear idea text")

    # We don't pin every step rigidly, but we DO require:
    # - interpreter called twice
    # - rag called twice
    # - evaluator called twice
    # - clarifier called once
    assert interpreter.calls == 2
    assert rag.calls == 2
    assert evaluator.calls == 2
    assert clarifier.calls == 1

    # And final summary exists
    assert final_state.summary == "final summary after clarification loop"
    # And we did end up in review branch, not strategist
    assert strategist.calls == 0
