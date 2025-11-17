"""
Prompt Builder Module
=====================

Provides reusable prompt templates for agents.
"""


class PromptBuilder:
    """Constructs prompts for all agents."""

    @staticmethod
    def interpreter_prompt(text: str) -> str:
        """
        Parameters
        ----------
        text : str
            Raw user text.

        Returns
        -------
        str
            Formatted JSON extraction prompt.
        """
        return f"""
You are an information extraction model. Extract structured JSON from the text below.

TEXT:
{text}

Return ONLY valid JSON in this format:
{{
  "idea_title": "",
  "problem_statement": "",
  "target_users": "",
  "context": "",
  "pain_points": [],
  "solution_features": [],
  "domain": "",
  "constraints": [],
  "use_cases": []
}}
"""

    @staticmethod
    def summarizer_prompt(text: str) -> str:
        """
        Parameters
        ----------
        text : str
            Text to summarize.

        Returns
        -------
        str
            Summarization prompt.
        """
        return f"Summarize this idea in 3–4 bullet points:\n\n{text}"
