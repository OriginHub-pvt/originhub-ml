"""
PipelineRunner
==============

This class orchestrates the full agent workflow:

Interpreter -> RAG -> Evaluator -> 
  - Clarifier loop (if needed)
  - Reviewer + MiniReview (existing idea)
  - Strategist (new idea)
-> Summarizer

This is the test-target orchestration engine used BEFORE integrating with Google ADK.
All agents follow the AgentBase API signature:
    agent.run(state) -> state
"""

from src.agentic.core.state import State

class PipelineRunner:
    """
    Executes the entire multi-agent reasoning pipeline.
    """

    def __init__(
        self,
        interpreter,
        clarifier,
        rag,
        evaluator,
        reviewer,
        strategist,
        summarizer,
    ):
        """
        Stores references to each agent.
        Each agent must implement:
            run(state) -> state
        """

        self.interpreter = interpreter
        self.clarifier = clarifier
        self.rag = rag
        self.evaluator = evaluator
        self.reviewer = reviewer
        self.strategist = strategist
        # MiniReview agent removed — summarizer will consume reviewer output directly
        self.summarizer = summarizer

    # ---------------------------------------------------------------
    # Main pipeline execution
    # ---------------------------------------------------------------

    def run(self, input_text: str) -> State:
        """
        Parameters
        ----------
        input_text : str

        Returns
        -------
        State
            Final state object after entire reasoning pipeline.
        """

        # Initialize state
        state = State(input_text=input_text)
        state.debug_trace = []   # Used by tests to assert call order

        # ------------- MAIN LOOP FOR CLARIFICATION -----------------
        while True:

            # 1. Interpreter
            state = self.interpreter.run(state)

            # 2. RAG
            state = self.rag.run(state)

            # 3. Evaluator
            state = self.evaluator.run(state)

            # ---------- Evaluator Decision Handling ----------

            # A) Clarification Loop
            if getattr(state, "need_more_clarification", False) is True:
                state = self.clarifier.run(state)
                # Loop back through interpreter/rag/evaluator
                continue

            # B) New Idea → Strategist
            if getattr(state, "is_new_idea", False) is True:
                state = self.strategist.run(state)
                state = self.summarizer.run(state)
                return state

            # C) Existing Idea → Reviewer + MiniReview
            state = self.reviewer.run(state)
            state = self.summarizer.run(state)
            return state
