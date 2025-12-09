from adk import agent

from src.agentic.core.state import State

# Import agent classes directly from their modules to avoid relying on
# a package-level `agents.__init__` and to keep constructor signatures clear.
from src.agentic.agents.interpreter_agent import InterpreterAgent
from src.agentic.agents.clarifier_agent import ClarifierAgent
from src.agentic.agents.rag_agent import RAGAgent
from src.agentic.agents.reviewer_agent import ReviewerAgent
from src.agentic.agents.strategist_agent import StrategistAgent
from src.agentic.agents.evaluator_agent import EvaluatorAgent
from src.agentic.agents.summarizer_agent import SummarizerAgent

"""
Each ADK @agent simply delegates to your existing fully-tested Python agent.
"""


@agent
class InterpreterNode:
    def run(self, state: State):
        return InterpreterAgent(
            inference_engine=state.context.inference_engine,
            prompt_builder=state.context.prompt_builder,
        ).run(state)


@agent
class ClarifierNode:
    def run(self, state: State):
        return ClarifierAgent(
            inference_engine=state.context.inference_engine,
            prompt_builder=state.context.prompt_builder,
        ).run(state)


@agent
class RAGNode:
    def run(self, state: State):
        # RAGAgent expects a vector_db/retriever parameter (named `vector_db`).
        return RAGAgent(
            vector_db=state.context.retriever,
        ).run(state)


@agent
class ReviewerNode:
    def run(self, state: State):
        return ReviewerAgent(
            inference_engine=state.context.inference_engine,
            prompt_builder=state.context.prompt_builder,
        ).run(state)


@agent
class StrategistNode:
    def run(self, state: State):
        return StrategistAgent(
            inference_engine=state.context.inference_engine,
            prompt_builder=state.context.prompt_builder,
        ).run(state)


@agent
class EvaluatorNode:
    def run(self, state: State):
        # EvaluatorAgent is rule-based and does not take inference_engine
        return EvaluatorAgent().run(state)


@agent
class SummarizerNode:
    def run(self, state: State):
        return SummarizerAgent(
            inference_engine=state.context.inference_engine,
            prompt_builder=state.context.prompt_builder,
        ).run(state)
