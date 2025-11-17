"""
Unit tests for PromptBuilder.
"""

from src.agentic.ml.prompt_builder import PromptBuilder


def test_interpreter_prompt_contains_all_json_fields():
    """Interpreter prompt should contain all required JSON keys."""

    prompt = PromptBuilder.interpreter_prompt("Test idea")

    required_fields = [
        '"idea_title"',
        '"problem_statement"',
        '"target_users"',
        '"context"',
        '"pain_points"',
        '"solution_features"',
        '"domain"',
        '"constraints"',
        '"use_cases"',
    ]

    for field in required_fields:
        assert field in prompt


def test_interpreter_prompt_includes_raw_text():
    """Interpreter prompt should embed the raw input text."""

    text = "AI for student productivity."
    prompt = PromptBuilder.interpreter_prompt(text)

    assert text in prompt
    assert "Extract structured JSON" in prompt


def test_interpreter_prompt_is_deterministic():
    """Interpreter prompt should be deterministic for same input."""

    text = "Some idea"
    p1 = PromptBuilder.interpreter_prompt(text)
    p2 = PromptBuilder.interpreter_prompt(text)

    assert p1 == p2


def test_summarizer_prompt_contains_prefix_and_text():
    """Summarizer prompt should include instruction prefix and original text."""

    text = "Build an AI assistant for course planning."
    prompt = PromptBuilder.summarizer_prompt(text)

    assert "Summarize this idea in 3–4 bullet points" in prompt
    assert text in prompt
