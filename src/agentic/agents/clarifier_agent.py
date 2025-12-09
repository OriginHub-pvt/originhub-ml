"""
ClarifierAgent
==============

This agent examines the interpreted idea JSON and determines
what additional clarifying questions are needed from the user.
"""

import json
import re
from typing import Any, List

from src.agentic.core.agent_base import AgentBase
from src.agentic.core.state import State
from src.agentic.utils.json_utils import extract_first_json


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

        # Attempt to parse output as JSON list. Be robust to noisy outputs
        questions: List[str] = []
        try:
            # Prefer extracting a JSON array of questions if present anywhere in the output.
            # Some LLMs echo the prompt or return multiple JSON blocks (an object then a list).
            parsed = extract_first_json(raw_output)

            # If the first JSON is a dict (e.g. echoed interpreted schema), try to find a later JSON array.
            if isinstance(parsed, dict):
                arr_match = re.search(r"\[.*\]", raw_output, flags=re.DOTALL)
                if arr_match:
                    try:
                        arr = json.loads(arr_match.group(0))
                        if isinstance(arr, list):
                            parsed = arr
                    except Exception:
                        pass

            # If we still don't have a parsed value, try to parse whole output as JSON
            if parsed is None:
                parsed = json.loads(raw_output)

            if isinstance(parsed, list):
                questions = [q for q in parsed if isinstance(q, str)]
            else:
                raise ValueError("Expected a JSON list of questions")

        except Exception:
            # Fallback: try to recover simple newline- or bullet-separated lines
            try:
                lines = [l.strip() for l in raw_output.splitlines() if l.strip()]
                # remove common list markers
                candidates = [l.lstrip('-*•0123456789. ').strip() for l in lines]
                # keep only short lines as questions
                questions = [c for c in candidates if len(c) > 10 and c.endswith('?')]
            except Exception:
                questions = []

        # Add parsed/fallback questions to state
        for q in questions:
            state.add_clarification(q)

        # Set high-level flag
        state.need_more_clarification = len(questions) > 0

        # If parsing failed and no questions were found, record error for diagnostics
        if not questions and raw_output:
            state.agent_outputs[self.name] = f"error: Could not parse questions from output: {raw_output}"

        return state