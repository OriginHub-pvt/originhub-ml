"""
EvaluatorAgent
==============

The routing agent of the pipeline.
Decides which agent should run next based on the State:

- If clarification is needed → ClarifierAgent
- If the idea is new          → StrategistAgent
- Otherwise                   → ReviewerAgent
"""

from src.agentic.core.state import State


class EvaluatorAgent:
    """
    EvaluatorAgent makes branching decisions in the agent graph.
    Does not call any LLM. Pure rule-based.
    """

    def __init__(self, name: str = "EvaluatorAgent"):
        """
        Parameters
        ----------
        name : str
            Agent name stored under agent_outputs.
        """
        self.name = name
        self.last_decision = None

    def run(self, state: State) -> State:
        """
        Evaluate the state and decide next step.

        Parameters
        ----------
        state : State

        Returns
        -------
        State
            Mutated state with next_action field.
        """
        if not hasattr(state, "agent_outputs") or state.agent_outputs is None:
            state.agent_outputs = {}

        # -------------------------------
        # Decision logic
        # -------------------------------
        if state.need_more_clarification:
            decision = "clarify"

        elif state.is_new_idea:
            decision = "strategize"

        else:
            decision = "review"

        # Store decision
        state.next_action = decision
        state.agent_outputs[self.name] = decision

        # Track internally for debugging
        self.last_decision = decision

        return state
