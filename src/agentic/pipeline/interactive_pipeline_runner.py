"""
InteractivePipelineRunner
=========================

A conversational, message-based controller for the multi-agent pipeline.

Behaves like a single unified AI assistant:
- Accepts user messages turn-by-turn
- Runs the appropriate agent
- Handles clarifier loops
- Ends after final summary is produced
"""

import json
import re

from src.agentic.core.state import State


class InteractivePipelineRunner:
    """
    Conversational orchestrator for the agentic system.

    Flow:
    1. InterpreterAgent processes first message.
    2. ClarifierAgent runs if need_more_clarification == True.
    3. RAGAgent searches Weaviate.
    4. EvaluatorAgent decides next agent: 'clarify', 'strategize', 'review'.
    5. StrategistAgent OR ReviewerAgent runs.
    6. MiniReviewAgent runs only if existing competitors found.
    7. SummarizerAgent produces final output.
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
        self.interpreter = interpreter
        self.clarifier = clarifier
        self.rag = rag            # MUST be RAGAgent(vector_db)
        self.evaluator = evaluator  # MUST be EvaluatorAgent()
        # MiniReview removed — keep reviewer only
        self.reviewer = reviewer
        self.strategist = strategist
        self.summarizer = summarizer

        self.state = State()
        self.is_done = False
        self.waiting_for_clarification = False

    # -------------------------------------------------------
    # MAIN ENTRY — Accept user message
    # -------------------------------------------------------
    def handle_user_message(self, user_message: str) -> str:
        self.state.last_user_message = user_message

        # User responding to clarifier question
        if self.waiting_for_clarification:
            return self._process_clarification_answer(user_message)

        # First turn → Interpreter must run
        if self.state.interpreted is None:
            return self._run_interpreter(user_message)

        # Otherwise continue normal pipeline
        return self._continue_pipeline()

    # =======================================================
    # INTERNAL PIPELINE STEPS
    # =======================================================

    def _run_interpreter(self, message: str) -> str:
        # Ensure the interpreter sees the user's latest message
        self.state.input_text = message
        # Clear any previously interpreted content so the interpreter runs fresh
        self.state.interpreted = None
        self.state = self.interpreter.run(self.state)

        if self.state.need_more_clarification:
            return self._ask_next_clarification_question()

        return self._continue_pipeline()

    # ---------------------------------------------
    # Clarifier Loop
    # ---------------------------------------------
    def _ask_next_clarification_question(self) -> str:
        questions = self.state.clarifications

        if not questions:
            # No parsed questions available — show clarifier raw output if present for diagnostics
            self.waiting_for_clarification = True
            clarifier_raw = self.state.agent_outputs.get(self.clarifier.name, None)
            if clarifier_raw:
                # Try to extract the first JSON array ([...] ) from the raw output
                arr_match = re.search(r"\[.*?\]", clarifier_raw, flags=re.DOTALL)
                if arr_match:
                    try:
                        arr = json.loads(arr_match.group(0))
                        if isinstance(arr, list) and arr:
                            # set parsed clarifications and return the first question
                            self.state.clarifications = [q for q in arr if isinstance(q, str)]
                            if self.state.clarifications:
                                self.waiting_for_clarification = True
                                return self.state.clarifications[0]
                    except Exception:
                        # fall through to preview
                        pass

                # Shorten long outputs for preview
                preview = clarifier_raw if len(clarifier_raw) < 400 else clarifier_raw[:400] + "..."
                return (
                    "I need more details to understand your idea. The clarifier returned:\n\n"
                    f"{preview}\n\nPlease respond with the requested details."
                )
            return "I need more details to understand your idea. Please describe it more clearly."

        question = questions[0]
        self.waiting_for_clarification = True
        return question

    def _process_clarification_answer(self, answer: str) -> str:
        self.waiting_for_clarification = False
        # Map the user's answer into the interpreted fields if possible.
        # Ensure interpreted dict exists
        if self.state.interpreted is None:
            self.state.interpreted = {}

        # If we have pending clarification questions, assume the current answer
        # corresponds to the first question in the list.
        if getattr(self.state, 'clarifications', None):
            try:
                current_q = self.state.clarifications.pop(0)
            except Exception:
                current_q = None

            field = None
            if current_q:
                ql = current_q.lower()
                if 'title' in ql:
                    field = 'title'
                elif 'description' in ql:
                    field = 'description'
                elif 'problem' in ql:
                    field = 'problem'

            # Fallback: store under a generic clarified_X key
            if not field:
                idx = len([k for k in self.state.interpreted.keys() if k.startswith('clarified_')]) + 1
                field = f'clarified_{idx}'

            # Save the answer
            self.state.interpreted[field] = answer

        # After ingesting the answer, determine whether more clarifications remain
        if getattr(self.state, 'clarifications', None):
            self.state.need_more_clarification = True
            return self._ask_next_clarification_question()

        # No more clarifications needed — continue the pipeline
        self.state.need_more_clarification = False
        return self._continue_pipeline()

    # ---------------------------------------------
    # Continue Pipeline Beyond Clarification
    # ---------------------------------------------
    def _continue_pipeline(self) -> str:

        # 1. RAG search (Weaviate)
        self.state = self.rag.run(self.state)

        # 2. Decide branch
        self.state = self.evaluator.run(self.state)

        # ----- Strategize branch -----
        if self.state.next_action == "strategize":
            self.state = self.strategist.run(self.state)
            self.state = self.summarizer.run(self.state)
            self.is_done = True
            return self.state.summary

        # ----- Review branch -----
        if self.state.next_action == "review":
            self.state = self.reviewer.run(self.state)

            # MiniReview step removed — summarizer consumes reviewer output
            self.state = self.summarizer.run(self.state)
            self.is_done = True
            return self.state.summary

        # ----- Clarifier branch -----
        if self.state.next_action == "clarify":
            return self._ask_next_clarification_question()

        return "⚠️ Unexpected pipeline state."
