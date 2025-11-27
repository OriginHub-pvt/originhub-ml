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
                {input_text}

                RETURN REQUIREMENTS (strict):
                    - Output EXACTLY one JSON object and NOTHING ELSE (no explanations, no code fences).
                    - Ensure the following REQUIRED fields are present and NON-EMPTY: "title", "description", "problem".
                        * "title": a short name (max 6 words), e.g. "Smart Grocery List".
                        * "description": 1-2 concise sentences summarizing the idea.
                        * "problem": 1 sentence describing the user problem this idea addresses.

                Additional optional fields (fill when available):
                    - "one_line": one-line summary
                    - "solution": how it solves the problem
                    - "domain": high-level category (e.g., productivity, health)
                    - "target_users": short description of the target users
                    - "key_features": list of 3–7 short feature strings
                    - "stage": one of ["idea", "prototype", "launched", "other"]
                    - "extra_notes": any other relevant details
                
                Important:
                    - Respond with ONLY valid JSON.
                    - Do not add explanations, commentary, or any extra text outside the JSON object.
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
            Prompt asking the model to generate field-specific clarifying questions.
        """
        required_fields = [
            ("title", "What is the title of your idea?"),
            ("description", "Please provide a description for your idea."),
            ("problem", "What problem does your idea solve?")
        ]
        missing = [q for f, q in required_fields if not interpreted.get(f)]
        if missing:
            questions = '\n'.join([f'- {q}' for q in missing])
            # Build a sanitized summary of the interpreted fields (no long text)
            safe_items = []
            try:
                import json as _json
                for k, v in (interpreted or {}).items():
                    if isinstance(v, str):
                        snippet = v.strip().replace('\n', ' ')[:120]
                        safe_items.append(f"{k}: \"{snippet}\"")
                    else:
                        safe_items.append(f"{k}: {type(v).__name__}")
                safe_summary = _json.dumps(safe_items, ensure_ascii=False)
            except Exception:
                safe_summary = str(list((interpreted or {}).keys()))

            # Ask the LLM to produce conversational, field-specific questions with short examples
            return f"""
            You are a friendly clarifying assistant.

            Current interpreted fields (sanitized):
            {safe_summary}

            The following required details are missing:
            {questions}

            CRITICAL: Output ONLY a JSON array of short question strings, and NOTHING ELSE. Do NOT echo the interpreted object, do not include any explanatory text, code fences, or metadata. If you cannot produce questions, return an empty JSON array `[]`.

            For each missing field, produce ONE short, conversational question that asks the user to provide that field. Phrase each question in second-person (e.g., "What is the title of your idea?") and include a very short example in parentheses after the question to guide the user (for example: "(e.g., 'Smart Grocery List')"). Keep questions under 20 words.

            Return a JSON array of strings only, in the order of the missing fields. The runner will ask them one at a time. Example response:
            [
              "What is the title of your idea? (e.g., 'Smart Grocery List')",
              "Please provide a short description of your idea. (2–3 sentences)"
            ]

            If everything is already clear, return an empty JSON array: [].
            """
        # Fallback to generic prompt if nothing is missing
        return f"""
            You are a clarifying assistant.

            Here is the current structured representation of the idea:
            {interpreted}

            If any required information is missing or unclear, ask a specific question for each missing field. Respond with a JSON array of strings. If everything is already clear, return [].
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
            Prompt for generating a SWOT or strategic view with research and action plan.
        """
        return f"""
            You are a startup strategist and research analyst.

            Here is a new idea with little or no close competition:
            {interpreted}

            Provide a comprehensive strategic analysis that includes:

            1. MARKET RESEARCH:
               - Target market size and growth potential
               - Key customer segments and their pain points
               - Market trends and dynamics
               - Potential barriers to entry

            2. COMPETITIVE LANDSCAPE:
               - Indirect competitors or adjacent solutions
               - What makes this idea unique
               - Potential future competitors

            3. SWOT ANALYSIS:
               - Strengths: Core advantages and capabilities
               - Weaknesses: Potential challenges and limitations
               - Opportunities: Market gaps and growth areas
               - Threats: Risks and external challenges

            4. ACTION PLAN (Next Steps):
               - Immediate actions (Week 1-4):
                 * Validation steps
                 * Market research tasks
                 * Initial prototype/MVP requirements
               - Short-term goals (Month 2-3):
                 * Key milestones
                 * Resource requirements
                 * Early customer acquisition strategy
               - Medium-term strategy (Month 4-6):
                 * Scaling considerations
                 * Team building needs
                 * Funding requirements

            5. SUCCESS METRICS:
               - Key performance indicators to track
               - Validation criteria for proceeding to next phase

            Format your response as structured text with clear headings and bullet points.
            Be specific and actionable in your recommendations.
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
