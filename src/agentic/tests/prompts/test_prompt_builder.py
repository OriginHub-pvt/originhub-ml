"""
Tests for PromptBuilder.

Ensures that:
- Each prompt method returns a non-empty string
- Prompts include the key inputs (idea text, interpreted dict, rag_results, etc.)
- Prompts mention the task clearly enough (clarify / review / strategy / summary)
"""

from src.agentic.prompts.prompt_builder import PromptBuilder


def test_interpreter_prompt_includes_input_text():
    pb = PromptBuilder()
    text = "AI tool for founders to manage ideas"
    prompt = pb.interpreter_prompt(text)

    assert isinstance(prompt, str)
    assert len(prompt) > 0
    assert text in prompt
    assert "JSON" in prompt or "json" in prompt


def test_clarifier_prompt_includes_interpreted_dict():
    pb = PromptBuilder()
    interpreted = {"title": "AI Notion", "domain": "productivity"}
    prompt = pb.clarifier_prompt(interpreted)

    assert isinstance(prompt, str)
    assert "clarifying" in prompt.lower()
    assert "question" in prompt.lower()
    # String form of dict should appear
    assert str(interpreted) in prompt


def test_review_prompt_includes_interpreted_and_rag_results():
    pb = PromptBuilder()
    interpreted = {"title": "AI CRM"}
    rag_results = [{"id": "1", "name": "HubSpot AI"}]

    prompt = pb.review_prompt(interpreted, rag_results)

    assert isinstance(prompt, str)
    assert "review" in prompt.lower() or "analysis" in prompt.lower()
    assert str(interpreted) in prompt
    assert str(rag_results) in prompt


def test_mini_review_prompt_includes_analysis_and_rag_results():
    pb = PromptBuilder()
    interpreted = {"title": "Idea"}
    rag_results = [{"id": "x"}]
    analysis = "There are 3 similar tools."

    prompt = pb.mini_review_prompt(interpreted, rag_results, analysis)

    assert isinstance(prompt, str)
    assert "mini" in prompt.lower()
    assert "review" in prompt.lower()
    assert str(interpreted) in prompt
    assert str(rag_results) in prompt
    assert analysis in prompt


def test_strategy_prompt_mentions_swot_and_includes_interpreted():
    pb = PromptBuilder()
    interpreted = {"title": "New AI startup"}

    prompt = pb.strategy_prompt(interpreted)

    assert isinstance(prompt, str)
    assert "swot" in prompt.lower() or "strategy" in prompt.lower()
    assert str(interpreted) in prompt


def test_summarizer_prompt_includes_all_sources():
    pb = PromptBuilder()
    interpreted = {"title": "Idea"}
    analysis = "market review"
    mini_review = "mini compare"
    strategy = "strategy text"

    prompt = pb.summarizer_prompt(
        interpreted,
        analysis,
        mini_review,
        strategy,
    )

    assert isinstance(prompt, str)
    assert "final" in prompt.lower() or "summary" in prompt.lower()
    assert str(interpreted) in prompt
    assert analysis in prompt
    assert mini_review in prompt
    assert strategy in prompt
