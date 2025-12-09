"""
InterpreterAgent
================

The first agent in the agentic pipeline.
Takes user input text and converts it into a structured JSON object
describing the idea (title, problem, domain, users, features, etc.).
"""

import json
import re
from typing import Any, Dict

from src.agentic.core.agent_base import AgentBase
from src.agentic.core.state import State
from src.agentic.utils.json_utils import extract_first_json, extract_all_jsons
import time


class InterpreterAgent(AgentBase):
    """
    InterpreterAgent converts raw user input text into structured idea data.
    """

    def __init__(
        self,
        inference_engine,
        prompt_builder,
        name: str = "InterpreterAgent",
    ):
        """
        Parameters
        ----------
        inference_engine : InferenceEngine
            Model inference engine.
        prompt_builder : object
            Object providing interpreter_prompt().
        name : str
            Agent name shown in state.agent_outputs.
        """
        super().__init__(
            name=name,
            inference_engine=inference_engine,
            prompt_builder=prompt_builder,
            heavy=False,   # Interpreter uses heavy model by default
            max_tokens=256,
            temperature=0.2,
            top_p=0.95,
            stop=None,
        )

    def build_prompt(self, state: State) -> str:
        """
        Parameters
        ----------
        state : State
            Current pipeline state.

        Returns
        -------
        str
            Prompt built using prompt_builder.
        """
        return self.prompt_builder.interpreter_prompt(state.input_text)

    def run(self, state: State) -> State:
        """
        Run the interpreter: build prompt → generate → parse JSON.
        If required fields are missing, call ClarifierAgent to get questions for the user.
        """
        state = super().run(state)
        raw_output = state.agent_outputs.get(self.name, "")

        # Attempt to parse model output into JSON. Extract all JSON blocks and
        # pick the one most relevant to the user input (token overlap). This
        # prevents example JSON blocks from being selected when they precede
        # the actual result.
        try:
            candidates = extract_all_jsons(raw_output)
            chosen = None

            # Collect tokens from input_text
            input_text = (state.input_text or "")
            input_tokens = set([t for t in re.findall(r"\w+", input_text.lower()) if len(t) > 2])

            best_score = -1
            for obj in candidates:
                # Prefer dicts for interpreted idea
                if isinstance(obj, dict):
                    # Concatenate string values from the dict
                    vals = " ".join([str(v) for v in obj.values() if isinstance(v, (str, int, float))])
                    vals_tokens = set([t for t in re.findall(r"\w+", vals.lower()) if len(t) > 2])
                    # simple overlap score
                    score = len(input_tokens & vals_tokens)
                    if score > best_score:
                        best_score = score
                        chosen = obj

            # If no dict candidates or all scores zero, prefer the last dict found
            if chosen is None and candidates:
                for obj in reversed(candidates):
                    if isinstance(obj, dict):
                        chosen = obj
                        break

            if chosen is None:
                # final fallback: try extract_first_json (existing behavior)
                chosen = extract_first_json(raw_output)

            state.interpreted = chosen

            # If no valid JSON found, treat as error
            if state.interpreted is None:
                state.agent_outputs[self.name] = f"error: No valid JSON found in output"
                return state

            # Warn if chosen JSON seems unrelated (low overlap)
            if state.interpreted and best_score <= 0:
                print("[InterpreterAgent] Warning: chosen JSON has low overlap with input_text — model may have echoed an example.")

        except Exception as e:
            state.interpreted = None
            state.agent_outputs[self.name] = f"error: JSONDecodeError: {e}"
            return state

        # Check for required fields
        missing_fields = []
        for field in ["title", "description", "problem"]:
            if not state.interpreted.get(field):
                missing_fields.append(field)

        if missing_fields:
            print(f"[InterpreterAgent] Missing required fields: {missing_fields}. Calling ClarifierAgent.")
            # Dynamically import ClarifierAgent to avoid circular import
            from src.agentic.agents.clarifier_agent import ClarifierAgent
            clarifier = ClarifierAgent(
                inference_engine=self.engine,
                prompt_builder=self.prompt_builder,
            )
            state = clarifier.run(state)
            # Set flag so pipeline/user can handle clarifications
            state.need_more_clarification = True
            return state

        # All required fields present
        state.need_more_clarification = False
        return state
