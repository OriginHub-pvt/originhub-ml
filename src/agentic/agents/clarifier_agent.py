"""
ClarifierAgent
==============

This agent examines the interpreted idea JSON and determines
what additional clarifying questions are needed from the user.
"""

import json
from typing import Any, List

from src.agentic.core.agent_base import AgentBase
from src.agentic.core.state import State


class ClarifierAgent(AgentBase):
    """
    ClarifierAgent generates clarifying questions for missing context
    in the user's idea.
    """

    def __init__(
        self,
        inference_engine,
        prompt_builder,
        name: str = "ClarifierAgent",
    ):
        """
        Parameters
        ----------
        inference_engine : InferenceEngine
            Model inference engine (uses light model).
        prompt_builder : object
            Provides clarifier_prompt().
        name : str
            Name used in state.agent_outputs.
        """
        super().__init__(
            name=name,
            inference_engine=inference_engine,
            prompt_builder=prompt_builder,
            heavy=False,          # Light/fast model
            max_tokens=256,
            temperature=0.3,
            top_p=0.95,
            stop=None,
        )

    def build_prompt(self, state: State) -> str:
        """
        Parameters
        ----------
        state : State
            Pipeline state containing interpreted idea.

        Returns
        -------
        str
            Prompt for LLM derived from interpreted JSON.
        """
        return self.prompt_builder.clarifier_prompt(state.interpreted)

    def run(self, state: State) -> State:
        """
        Run the clarifier agent:
        - build prompt
        - generate questions using LLM
        - parse resulting list of questions
        - update state.clarifications + flags
        """
        # First run base class logic (prompt + inference + storing raw output)
        state = super().run(state)

        raw_output = state.agent_outputs.get(self.name, "")

        # Attempt to parse output as JSON list
        questions: List[str] = []
        try:
            parsed = json.loads(raw_output)

            if isinstance(parsed, list):
                # keep only string questions
                questions = [q for q in parsed if isinstance(q, str)]
            else:
                raise ValueError("Expected a JSON list of questions")

        except Exception as e:
            # Model did not return a proper list
            state.agent_outputs[self.name] = f"error: {type(e).__name__}: {e}"
            return state

        # Add parsed questions to state
        for q in questions:
            state.add_clarification(q)

        # Set high-level flag
        state.need_more_clarification = len(questions) > 0

        return state