"""
Tests for the State class.

The State object is the shared memory passed between agents.
These tests verify:
- initialization
- attribute safety
- update/merge behavior
- flag setting
- reset capability
- serialization (to_dict)
"""

from src.agentic.core.state import State


def test_state_initializes_with_defaults():
    """State should initialize with empty/default values."""

    s = State()

    assert s.input_text == ""
    assert s.interpreted is None
    assert s.clarifications == []
    assert s.rag_results == []
    assert s.analysis is None
    assert s.strategy is None
    assert s.is_new_idea is False
    assert s.need_more_clarification is False


def test_state_allows_setting_input_text():
    """State should store user input text."""
    s = State(input_text="hello world")
    assert s.input_text == "hello world"


def test_state_update_interpreted_json():
    """State should allow storing interpreted JSON dict."""
    s = State()
    s.interpreted = {"title": "AI tool", "domain": "productivity"}
    assert s.interpreted["title"] == "AI tool"
    assert s.interpreted["domain"] == "productivity"


def test_state_clarifications_append():
    """State should append clarifications safely."""
    s = State()
    s.add_clarification("clarify feature set")
    s.add_clarification("clarify target users")

    assert s.clarifications == [
        "clarify feature set",
        "clarify target users",
    ]


def test_state_add_rag_results():
    """State should allow adding retrieved RAG entries."""
    s = State()
    s.add_rag_results([
        {"text": "existing idea", "score": 0.12},
        {"text": "similar startup", "score": 0.20},
    ])

    assert len(s.rag_results) == 2
    assert s.rag_results[0]["text"] == "existing idea"


def test_state_flags_update():
    """State flags should be settable and readable."""
    s = State()
    s.is_new_idea = True
    s.need_more_clarification = True

    assert s.is_new_idea is True
    assert s.need_more_clarification is True


def test_state_merge_updates_fields():
    """
    merge() should update fields from another State instance.
    Useful when multiple agents enrich the same state.
    """
    s1 = State(input_text="idea A")
    s1.interpreted = {"title": "A"}

    s2 = State()
    s2.clarifications.append("Need more detail")
    s2.is_new_idea = True

    s1.merge(s2)

    assert s1.interpreted == {"title": "A"}
    assert s1.clarifications == ["Need more detail"]
    assert s1.is_new_idea is True


def test_state_to_dict_serializes_all_fields():
    """to_dict() should serialize all public fields cleanly."""
    s = State(
        input_text="hello",
        interpreted={"title": "Hello"},
        analysis="analysis text",
        strategy="go to market",
    )
    s.add_clarification("clarify description")
    s.add_rag_results([{"text": "match"}])
    s.is_new_idea = True

    d = s.to_dict()

    assert d["input_text"] == "hello"
    assert d["interpreted"]["title"] == "Hello"
    assert d["clarifications"] == ["clarify description"]
    assert d["rag_results"][0]["text"] == "match"
    assert d["analysis"] == "analysis text"
    assert d["strategy"] == "go to market"
    assert d["is_new_idea"] is True
    assert d["need_more_clarification"] is False


def test_state_reset_clears_dynamic_fields():
    """
    reset() should clear agent-generated fields but preserve input_text.
    This is helpful if re-processing the same idea.
    """
    s = State(input_text="hello")

    s.interpreted = {"x": 1}
    s.clarifications = ["c"]
    s.rag_results = [{}]
    s.analysis = "a"
    s.strategy = "b"
    s.is_new_idea = True
    s.need_more_clarification = True

    s.reset()

    assert s.input_text == "hello"
    assert s.interpreted is None
    assert s.clarifications == []
    assert s.rag_results == []
    assert s.analysis is None
    assert s.strategy is None
    assert s.is_new_idea is False
    assert s.need_more_clarification is False
