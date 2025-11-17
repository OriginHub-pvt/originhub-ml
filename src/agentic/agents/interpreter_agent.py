"""
InterpreterAgent
================

The first agent in the agentic pipeline.
Takes user input text and converts it into a structured JSON object
describing the idea (title, problem, domain, users, features, etc.).
"""

import json
from typing import Any, Dict

from src.agentic.core.agent_base import AgentBase
from src.agentic.core.state import State


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
            heavy=True,   # Interpreter uses heavy model by default
            max_tokens=512,
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
        """
        # Call base class to handle prompt + raw generation + agent_outputs
        state = super().run(state)

        raw_output = state.agent_outputs.get(self.name, "")

        # Attempt to parse model output into JSON
        try:
            parsed: Dict[str, Any] = json.loads(raw_output)
            state.interpreted = parsed
        except Exception as e:
            # JSON parsing failed → mark as error
            state.interpreted = None
            state.agent_outputs[self.name] = f"error: JSONDecodeError: {e}"

        return state
