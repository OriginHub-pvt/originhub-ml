"""
InteractivePipelineRunner
=========================

A conversational, message-based controller for the multi-agent pipeline.

Behaves like a single unified AI assistant:
- Accepts user messages turn-by-turn
- Runs the appropriate agent
- Handles clarifier loops
- Produces comprehensive analysis
- Supports follow-up questions after analysis completion
- Maintains conversation history for context-aware responses
"""

import logging
import json
import re

from src.agentic.core.state import State

logger = logging.getLogger(__name__)


class InteractivePipelineRunner:
    """
    Conversational orchestrator for the agentic system.

    Flow:
    1. InterpreterAgent processes first message.
    2. ClarifierAgent runs if need_more_clarification == True.
    3. RAGAgent searches Weaviate.
    4. EvaluatorAgent decides next agent: 'clarify', 'strategize', 'review'.
    5. StrategistAgent OR ReviewerAgent runs.
    6. SummarizerAgent produces final output.
    7. Follow-up questions are handled conversationally with full context.
    
    After analysis completion, users can ask follow-up questions about:
    - Clarification on specific points
    - Deep dives into action plans
    - Market research details
    - Implementation guidance
    - Any aspect of the analysis
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
        self.analysis_complete = False
        self.conversation_history = []

    # -------------------------------------------------------
    # MAIN ENTRY — Accept user message
    # -------------------------------------------------------
    def handle_user_message(self, user_message: str) -> str:
        self.state.last_user_message = user_message
        
        # Add to conversation history
        self.conversation_history.append({"role": "user", "content": user_message})
        logger.debug(f"User message received: {len(user_message)} chars")

        # User responding to clarifier question
        if self.waiting_for_clarification:
            logger.info("Processing clarification response")
            return self._process_clarification_answer(user_message)
        
        # Analysis complete - handle as follow-up question
        if self.analysis_complete:
            logger.info("Processing follow-up question")
            return self._handle_followup_question(user_message)

        # First turn → Interpreter must run
        if self.state.interpreted is None:
            logger.info("Running interpreter agent")
            return self._run_interpreter(user_message)

        # Otherwise continue normal pipeline
        logger.info("Continuing pipeline")
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
            self.analysis_complete = True
            
            # Handle summary as dict or string
            summary_text = self._format_summary(self.state.summary)
            self.conversation_history.append({"role": "assistant", "content": summary_text})
            return summary_text

        # ----- Review branch -----
        if self.state.next_action == "review":
            self.state = self.reviewer.run(self.state)

            # MiniReview step removed — summarizer consumes reviewer output
            self.state = self.summarizer.run(self.state)
            self.analysis_complete = True
            
            # Handle summary as dict or string
            summary_text = self._format_summary(self.state.summary)
            response = summary_text + "\n\n---\nI've completed the analysis. Feel free to ask follow-up questions!"
            self.conversation_history.append({"role": "assistant", "content": response})
            return response

        # ----- Clarifier branch -----
        if self.state.next_action == "clarify":
            return self._ask_next_clarification_question()

        return "WARNING: Unexpected pipeline state."
    
    # ---------------------------------------------
    # Helper Methods
    # ---------------------------------------------
    def _format_summary(self, summary) -> str:
        """
        Format summary - handle both dict and string formats.
        
        Parameters
        ----------
        summary : dict or str or None
            Summary from SummarizerAgent
            
        Returns
        -------
        str
            Formatted summary text
        """
        if summary is None:
            return "Analysis completed but no summary was generated."
        
        if isinstance(summary, str):
            return summary
        
        if isinstance(summary, dict):
            # Pretty print JSON summary
            return json.dumps(summary, indent=2)
        
        return str(summary)
    
    # ---------------------------------------------
    # Follow-up Question Handler
    # ---------------------------------------------
    def _handle_followup_question(self, question: str) -> str:
        """
        Handle follow-up questions after initial analysis is complete.
        Uses conversation history and analysis context to provide informed answers.
        """
        # Build context from analysis results
        context_parts = [
            "You are an AI assistant helping with business idea analysis.",
            "\nPrevious Analysis Context:",
        ]
        
        if self.state.interpreted:
            context_parts.append(f"\nIdea: {json.dumps(self.state.interpreted, indent=2)}")
        
        if self.state.strategy:
            strategy_str = self.state.strategy if isinstance(self.state.strategy, str) else json.dumps(self.state.strategy, indent=2)
            context_parts.append(f"\nStrategic Analysis: {strategy_str}")
        
        if self.state.analysis:
            context_parts.append(f"\nCompetitive Review: {self.state.analysis}")
        
        if self.state.summary:
            context_parts.append(f"\nSummary: {self.state.summary}")
        
        # Add recent conversation history (last 4 exchanges)
        recent_history = self.conversation_history[-8:] if len(self.conversation_history) > 8 else self.conversation_history
        if recent_history:
            context_parts.append("\nRecent Conversation:")
            for msg in recent_history:
                role = msg['role'].capitalize()
                content = msg['content'][:200] + "..." if len(msg['content']) > 200 else msg['content']
                context_parts.append(f"{role}: {content}")
        
        context_parts.append(f"\nCurrent Question: {question}")
        context_parts.append("\nProvide a helpful, specific answer based on the analysis context. Be concise but informative.")
        
        prompt = "\n".join(context_parts)
        
        # Use heavy model for follow-up questions to maintain quality
        try:
            response = self.summarizer.engine.generate(
                prompt=prompt,
                heavy=False,  # Use 7B model
                temperature=0.7,
                max_tokens=512,
                stop=None,
            )
            
            self.conversation_history.append({"role": "assistant", "content": response})
            return response
        except Exception as e:
            error_msg = f"I encountered an error processing your question: {str(e)}"
            self.conversation_history.append({"role": "assistant", "content": error_msg})
            return error_msg
