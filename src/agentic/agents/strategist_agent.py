"""
StrategistAgent
===============

This agent creates a strategic blueprint or SWOT analysis for a brand-new idea.
Uses the heavy model for deep reasoning.

Outputs:
- state.strategy  (parsed SWOT dict or plain text)
- state.agent_outputs["StrategistAgent"] (raw output or error)
"""

import json
from typing import Any, Dict

from src.agentic.core.agent_base import AgentBase
from src.agentic.core.state import State


class StrategistAgent(AgentBase):
    """
    StrategistAgent generates SWOT analysis or strategic guidance
    for ideas that are flagged as new (state.is_new_idea == True).
    """

    def __init__(
        self,
        inference_engine,
        prompt_builder,
        name: str = "StrategistAgent",
    ):
        """
        Parameters
        ----------
        inference_engine : InferenceEngine
            LLM inference engine (heavy model).
        prompt_builder : object
            Provides strategy_prompt(interpreted).
        name : str
            Name stored under state.agent_outputs.
        """
        super().__init__(
            name=name,
            inference_engine=inference_engine,
            prompt_builder=prompt_builder,
            heavy=True,          # Strategist must use HEAVY model
            max_tokens=1024,
            temperature=0.25,
            top_p=0.95,
            stop=None,
        )

    def build_prompt(self, state: State) -> str:
        """
        Build strategy prompt using interpreted idea JSON.

        Parameters
        ----------
        state : State

        Returns
        -------
        str
            Prompt for the LLM.
        """
        return self.prompt_builder.strategy_prompt(state.interpreted)

    def run(self, state: State) -> State:
        """
        Execute StrategistAgent:
        - Build prompt
        - Generate strategic analysis using heavy LLM
        - Store raw output
        - Parse JSON if present
        - Else store text directly
        - Handle errors gracefully
        """
        # Let AgentBase do prompt building, LLM call, and base error handling
        state = super().run(state)

        raw_output = state.agent_outputs.get(self.name, "")

        # If AgentBase stored an error message...
        if raw_output.lower().startswith("error"):
            state.strategy = None
            return state

        # Try JSON first
        cleaned = raw_output.strip()
        try:
            parsed_swot = json.loads(cleaned)
            if isinstance(parsed_swot, dict):
                state.strategy = parsed_swot
                return state
        except Exception:
            # Not valid JSON → treat as plain text
            pass

        # Plain text fallback
        state.strategy = cleaned
        return state
