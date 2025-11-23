"""
SummarizerAgent
===============

This agent produces the final user-facing summary by combining:
- interpreted idea
- reviewer output
- mini-review
- strategist output (if new idea)

It uses the light model for fast summarization.

Outputs:
- state.summary
- state.agent_outputs["SummarizerAgent"]
"""

import json
from typing import Any, Dict

from src.agentic.core.agent_base import AgentBase
from src.agentic.core.state import State
from src.agentic.utils.json_utils import extract_first_json


class SummarizerAgent(AgentBase):
    """
    SummarizerAgent generates a polished, concise final summary that is sent to the UI.
    """

    def __init__(
        self,
        inference_engine,
        prompt_builder,
        name: str = "SummarizerAgent",
    ):
        """
        Parameters
        ----------
        inference_engine : InferenceEngine
            LLM inference engine (light model).
        prompt_builder : object
            Provides summarizer_prompt(interpreted, analysis, mini_review, strategy).
        name : str
            Key under state.agent_outputs.
        """
        super().__init__(
            name=name,
            inference_engine=inference_engine,
            prompt_builder=prompt_builder,
            heavy=False,           # Summarizer MUST use light model
            max_tokens=512,
            temperature=0.3,
            top_p=0.95,
            stop=None,
        )

    def build_prompt(self, state: State) -> str:
        """
        Build summarization prompt by passing all relevant state components.

        Parameters
        ----------
        state : State

        Returns
        -------
        str
            Prompt for LLM.
        """
        return self.prompt_builder.summarizer_prompt(
            state.interpreted,
            state.analysis,
            state.mini_review,
            state.strategy,
        )

    def run(self, state: State) -> State:
        """
        Execute summarization:
        - Build prompt
        - Call LLM via AgentBase
        - Parse JSON or fallback to plain text
        - Store final summary into state.summary
        - Store raw output too
        """
        state = super().run(state)
        print("Summarizer")
        raw_output = state.agent_outputs.get(self.name, "")

        # If AgentBase recorded an LLM failure
        if raw_output.lower().startswith("error"):
            state.summary = None
            return state

        cleaned = raw_output.strip()

        # Prefer the first JSON object found anywhere in the output
        parsed = extract_first_json(cleaned)
        if parsed is not None:
            state.summary = parsed
            return state

        # Try direct loads as a last resort
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                state.summary = parsed
                return state
        except Exception:
            pass

        # Fallback: plain text summary
        state.summary = cleaned
        return state
