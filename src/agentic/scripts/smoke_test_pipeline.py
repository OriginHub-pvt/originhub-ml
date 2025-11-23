"""Smoke test runner for the agentic pipeline using lightweight mocks.

This file runs the PipelineRunner with stubbed inference and retriever
so you can validate the control flow without heavy GGUF models or Weaviate.
"""

from unittest.mock import MagicMock

from src.agentic.ml.inference_engine import InferenceEngine
from src.agentic.ml.model_manager import ModelManager
from src.agentic.prompts.prompt_builder import PromptBuilder

from src.agentic.agents.interpreter_agent import InterpreterAgent
from src.agentic.agents.clarifier_agent import ClarifierAgent
from src.agentic.agents.rag_agent import RAGAgent
from src.agentic.agents.reviewer_agent import ReviewerAgent
from src.agentic.agents.strategist_agent import StrategistAgent
from src.agentic.agents.evaluator_agent import EvaluatorAgent
from src.agentic.agents.summarizer_agent import SummarizerAgent

from src.agentic.pipeline.pipeline_runner import PipelineRunner


class DummyRetriever:
    def search(self, query_text: str):
        # Return a non-empty list to trigger the reviewer path
        return [{"id": "1", "title": "Existing Product", "summary": "An existing competitor", "distance": 0.1}]


def run_smoke():
    # Create a fake inference engine using MagicMock
    fake_engine = MagicMock()
    # Interpreter returns a JSON object string
    fake_engine.generate.side_effect = [
        '{"title": "Test Idea", "one_line": "a test"}',  # Interpreter
        '[]',  # Clarifier (no questions)
        'Market analysis text',  # Reviewer
        '{"final_summary": "summary", "next_steps": ["do X"]}',  # Summarizer
    ]

    # Use a minimal ModelManager placeholder to satisfy InferenceEngine API.
    mm = MagicMock()
    engine = InferenceEngine(mm)
    engine.generate = fake_engine.generate

    prompts = PromptBuilder()

    interpreter = InterpreterAgent(engine, prompts)
    clarifier = ClarifierAgent(engine, prompts)
    rag = RAGAgent(vector_db=DummyRetriever())
    reviewer = ReviewerAgent(engine, prompts)
    # MiniReview removed — not instantiated
    strategist = StrategistAgent(engine, prompts)
    evaluator = EvaluatorAgent()
    summarizer = SummarizerAgent(engine, prompts)

    runner = PipelineRunner(
        interpreter=interpreter,
        clarifier=clarifier,
        rag=rag,
        evaluator=evaluator,
        reviewer=reviewer,
        strategist=strategist,
        summarizer=summarizer,
    )

    final = runner.run("A sample idea for smoke testing")
    print("Smoke test final summary:", final.summary)


if __name__ == "__main__":
    run_smoke()
