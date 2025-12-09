"""
ReviewerAgent
=============

This agent generates a mini market review for an idea using the light model.
Triggered when vector DB shows that similar solutions already exist.

Output:
- state.analysis  (clean natural language summary)
- state.agent_outputs["ReviewerAgent"] (raw LLM output or error)
"""

from src.agentic.core.agent_base import AgentBase
from src.agentic.core.state import State


class ReviewerAgent(AgentBase):
    """
    ReviewerAgent generates a short market analysis comparing the user's idea
    to existing solutions (rag_results).
    """

    def __init__(
        self,
        inference_engine,
        prompt_builder,
        name: str = "ReviewerAgent",
    ):
        """
        Parameters
        ----------
        inference_engine : InferenceEngine
            LLM inference engine.
        prompt_builder : object
            Provides review_prompt(interpreted, rag_results).
        name : str
            Name stored in state.agent_outputs.
        """
        super().__init__(
            name=name,
            inference_engine=inference_engine,
            prompt_builder=prompt_builder,
            heavy=False,          # Reviewer uses the light model
            max_tokens=256,
            temperature=0.4,
            top_p=0.9,
            stop=None,
        )

    def build_prompt(self, state: State) -> str:
        """
        Parameters
        ----------
        state : State
            Contains interpreted idea + rag_results.

        Returns
        -------
        str
            Prompt for the LLM.
        """
        return self.prompt_builder.review_prompt(
            state.interpreted,
            state.rag_results
        )

    def run(self, state: State) -> State:
        """
        Execute the ReviewerAgent:
        - Build prompt
        - Generate a short review using LLM
        - Store raw output
        - Store cleaned analysis in state.analysis
        """
        # Let AgentBase handle prompt creation, LLM call, error handling
        state = super().run(state)
        raw_output = state.agent_outputs.get(self.name, "")

        # If an error occurred, AgentBase already stored "error: ..."
        if raw_output.lower().startswith("error"):
            state.analysis = None
            return state

        # Reviewer output is natural language; store it directly
        state.analysis = raw_output.strip()
        return state