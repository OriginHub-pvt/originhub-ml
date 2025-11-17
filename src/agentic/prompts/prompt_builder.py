"""
PromptBuilder
=============

Central place to construct prompts for all agents.

This keeps prompt templates in one location so they are easy
to iterate on and easy to test.
"""

from typing import Any, Dict, List, Optional


class PromptBuilder:
    """
    Builds prompts for each agent in the pipeline.
    """

    def __init__(
        self,
        language: str = "en",
    ):
        """
        Parameters
        ----------
        language : str
            Reserved for future multilingual support. Currently unused.
        """
        self.language = language

    # ------------------------------------------------------------------
    # 1) Interpreter
    # ------------------------------------------------------------------

    def interpreter_prompt(self, input_text: str) -> str:
        """
        Parameters
        ----------
        input_text : str
            Raw user idea text.

        Returns
        -------
        str
            Prompt instructing the model to return a single JSON object.
        """
        return f"""
            You are an assistant that converts startup or product ideas into a clean JSON object.

            User idea:
            \"\"\"{input_text}\"\"\"

            Return a single JSON object with fields like:
            - "title": short name of the idea
            - "one_line": one line summary
            - "problem": what problem it solves
            - "solution": how it solves it
            - "domain": high-level category (e.g., productivity, devtools, health, finance)
            - "target_users": short description of the target users
            - "key_features": list of 3–7 features
            - "stage": one of ["idea", "prototype", "launched", "other"]
            - "extra_notes": any other relevant details

            Important:
            - Respond with ONLY valid JSON.
            - Do not add explanations or comments.
        """

    # ------------------------------------------------------------------
    # 2) Clarifier
    # ------------------------------------------------------------------

    def clarifier_prompt(self, interpreted: Dict[str, Any]) -> str:
        """
        Parameters
        ----------
        interpreted : dict
            Parsed idea JSON from InterpreterAgent.

        Returns
        -------
        str
            Prompt asking the model to generate clarifying questions.
        """
        return f"""
            You are a clarifying assistant.

            Here is the current structured representation of the idea:
            {interpreted}

            Your task:
            - Identify missing or ambiguous information.
            - Write 0–5 short clarifying questions that would help you understand the idea better.

            Respond with a JSON array of strings, for example:
            [
            "Who is the primary target user?",
            "How will this product make money?"
            ]
            If everything is already clear, return [].
        """

    # ------------------------------------------------------------------
    # 3) Reviewer (market review)
    # ------------------------------------------------------------------

    def review_prompt(
        self,
        interpreted: Dict[str, Any],
        rag_results: List[Dict[str, Any]],
    ) -> str:
        """
        Parameters
        ----------
        interpreted : dict
            Idea JSON.
        rag_results : list of dict
            Retrieved similar tools / companies from vector DB.

        Returns
        -------
        str
            Prompt for generating a short market analysis.
        """
        return f"""
            You are a product and market analysis assistant.

            Idea (structured):
            {interpreted}

            Similar existing tools / companies (from retrieval):
            {rag_results}

            Write a concise market analysis (5–10 sentences) that covers:
            - How this idea fits into the current landscape.
            - Overlaps with existing tools.
            - Gaps or opportunities you see.
            - Any high-level risks.

            Respond in plain English, no JSON needed.
        """

    # ------------------------------------------------------------------
    # 4) MiniReview (differentiation)
    # ------------------------------------------------------------------

    def mini_review_prompt(
    self,
    interpreted: Dict[str, Any],
    rag_results: List[Dict[str, Any]],
    analysis: Optional[str],
    ) -> str:
        """
        Parameters
        ----------
        interpreted : dict
            Idea JSON.
        rag_results : list of dict
            Retrieved similar tools / companies.
        analysis : str or None
            Market analysis text from ReviewerAgent.

        Returns
        -------
        str
            Prompt for a short competitor comparison mini-review.
        """
        return f"""
            You are a product differentiation assistant generating a mini review of the idea.

            Idea:
            {interpreted}

            Similar tools / competitors:
            {rag_results}

            Market analysis:
            {analysis}

            Task:
            - Briefly explain how this idea is similar to existing solutions.
            - Highlight 2–5 ways it could be different or better.
            - Mention any obvious red flags.

            You may respond either:
            - As plain text (1–2 short paragraphs), or
            - As JSON with fields like "unique_points" and "risks".
        """


    # ------------------------------------------------------------------
    # 5) Strategist (SWOT / strategy)
    # ------------------------------------------------------------------

    def strategy_prompt(self, interpreted: Dict[str, Any]) -> str:
        """
        Parameters
        ----------
        interpreted : dict
            Idea JSON for a brand-new idea (no strong matches in DB).

        Returns
        -------
        str
            Prompt for generating a SWOT or strategic view.
        """
        return f"""
            You are a startup strategist.

            Here is a new idea with little or no close competition:
            {interpreted}

            Produce a SWOT-style strategic view that covers:
            - strengths
            - weaknesses
            - opportunities
            - threats

            You may respond either:
            - As JSON with keys "strengths", "weaknesses", "opportunities", "threats",
            - Or as a structured plain text answer with headings.
        """

    # ------------------------------------------------------------------
    # 6) Summarizer (final answer)
    # ------------------------------------------------------------------

    def summarizer_prompt(
        self,
        interpreted: Dict[str, Any],
        analysis: Optional[str],
        mini_review: Any,
        strategy: Any,
    ) -> str:
        """
        Parameters
        ----------
        interpreted : dict
            The structured idea.
        analysis : str or None
            Market review text (existing solutions).
        mini_review : any
            Mini-review information (text or dict).
        strategy : any
            Strategic / SWOT info (text or dict), may be None.

        Returns
        -------
        str
            Prompt for producing the final answer for the end user.
        """
        return f"""
            You are a summarizer that produces a final, user-friendly summary.

            Idea (structured):
            {interpreted}

            Market analysis (may be None):
            {analysis}

            Mini-review (may be text or JSON):
            {mini_review}

            Strategy / SWOT (may be text, JSON, or None):
            {strategy}

            Task:
            - Produce a clear, concise final summary of the idea and its context.
            - If there is market analysis, briefly mention similar tools and how this idea compares.
            - If there is strategy info, mention 2–3 key strengths or opportunities.
            - End with 2–3 suggested next steps for the user.

            You may respond either as:
            - A plain text answer, or
            - A JSON object with at least "final_summary" and "next_steps".
        """
