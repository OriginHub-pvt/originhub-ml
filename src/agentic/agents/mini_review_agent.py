"""
MiniReviewAgent
===============

This agent generates a short competitor-comparison mini-review.
Uses the light model and takes into account:
- interpreted idea
- rag_results
- reviewer analysis (market review)

Outputs:
- state.mini_review
- state.agent_outputs["MiniReviewAgent"]
"""

import json
from typing import Any, Dict

from src.agentic.core.agent_base import AgentBase
from src.agentic.core.state import State


class MiniReviewAgent(AgentBase):
    """
    MiniReviewAgent helps the user understand how their idea differs from
    existing competitors by generating a short competitor comparison.
    """

    def __init__(
        self,
        inference_engine,
        prompt_builder,
        name: str = "MiniReviewAgent",
    ):
        """
        Parameters
        ----------
        inference_engine : InferenceEngine
            Light model engine.
        prompt_builder : object
            Provides mini_review_prompt(interpreted, rag_results, analysis).
        name : str
            Used for state.agent_outputs.
        """
        super().__init__(
            name=name,
            inference_engine=inference_engine,
            prompt_builder=prompt_builder,
            heavy=False,          # Mini review uses light model
            max_tokens=512,
            temperature=0.4,
            top_p=0.9,
            stop=None,
        )

    def build_prompt(self, state: State) -> str:
        """
        Parameters
        ----------
        state : State

        Returns
        -------
        str
            Prompt for LLM.
        """
        return self.prompt_builder.mini_review_prompt(
            state.interpreted,
            state.rag_results,
            state.analysis
        )

    def run(self, state: State) -> State:
        """
        Execute the mini-review generation:
        - Build prompt
        - Run LLM (via AgentBase)
        - Try JSON parse, else treat as plain text
        - On failure → safe error handling
        """
        state = super().run(state)
        raw_output = state.agent_outputs.get(self.name, "")

        # If AgentBase recorded an error, stop cleanly
        if raw_output.lower().startswith("error"):
            state.mini_review = None
            return state

        cleaned = raw_output.strip()

        # Attempt JSON parse
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                state.mini_review = parsed
                return state
        except Exception:
            # Not JSON
            pass

        # Plain text fallback
        state.mini_review = cleaned
        return state
