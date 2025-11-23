#!/usr/bin/env python3
"""
Run the full agentic pipeline end-to-end in terminal.

Usage:
    python scripts/run_pipeline.py "your idea text here"
"""

import sys
import os

# ------------------ LLM / AGENT IMPORTS ------------------

from src.agentic.ml.model_manager import ModelManager
from src.agentic.ml.inference_engine import InferenceEngine
from src.agentic.prompts.prompt_builder import PromptBuilder

from src.agentic.agents.interpreter_agent import InterpreterAgent
from src.agentic.agents.clarifier_agent import ClarifierAgent
from src.agentic.agents.rag_agent import RAGAgent
from src.agentic.agents.reviewer_agent import ReviewerAgent
from src.agentic.agents.strategist_agent import StrategistAgent
from src.agentic.agents.evaluator_agent import EvaluatorAgent
from src.agentic.agents.summarizer_agent import SummarizerAgent

from src.agentic.pipeline.pipeline_runner import PipelineRunner

# ------------------ WEAVIATE ------------------

from weaviate import connect_to_local
from src.agentic.rag.weaviate_retriever import WeaviateRetriever
from dotenv import load_dotenv

load_dotenv()

WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "weaviate")
WEAVIATE_PORT = int(os.getenv("WEAVIATE_PORT", 8080))
WEAVIATE_GRPC_PORT = int(os.getenv("WEAVIATE_GRPC_PORT", 50051))
WEAVIATE_COLLECTION = os.getenv("WEAVIATE_COLLECTION", "ArticleSummary")


# ------------------------------------------------------------
# Main Runner
# ------------------------------------------------------------

def run_pipeline(user_text: str):
    print("\n🚀 Running OriginHub Agentic Pipeline...\n")
    print("📝 User Input:", user_text)
    print("\n----------------------------------------\n")

    # -----------------------------
    # MODEL LOADING
    # -----------------------------
    MODEL_7B_PATH = os.getenv("MODEL_7B_PATH", "models/qwen-7b/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf")
    MODEL_1B_PATH = os.getenv("MODEL_1B_PATH", "models/qwen-1.5b/model.gguf")

    print("🔧 Loading local LLM models... (this may take a moment)\n")

    model_manager = ModelManager(
        model7b_path=MODEL_7B_PATH,
        model1b_path=MODEL_1B_PATH,
        context_size=int(os.getenv("MODEL_CONTEXT_SIZE", 4096)),
        n_threads=int(os.getenv("MODEL_N_THREADS", 4)),
        gpu_layers=int(os.getenv("MODEL_GPU_LAYERS", 20)),
    )

    inference = InferenceEngine(model_manager)
    prompts = PromptBuilder()

    # -----------------------------
    # WEAVIATE SETUP
    # -----------------------------
    print("🔍 Connecting to Weaviate...\n")

    client = connect_to_local(
        host=WEAVIATE_HOST,
        port=WEAVIATE_PORT,
        grpc_port=WEAVIATE_GRPC_PORT,
    )

    retriever = WeaviateRetriever(
        client=client,
        collection_name=WEAVIATE_COLLECTION,
    )

    # -----------------------------
    # AGENT INSTANTIATION
    # -----------------------------
    interpreter = InterpreterAgent(inference, prompts)
    clarifier = ClarifierAgent(inference, prompts)
    rag = RAGAgent(vector_db=retriever)
    reviewer = ReviewerAgent(inference, prompts)
    strategist = StrategistAgent(inference, prompts)
    evaluator = EvaluatorAgent()
    summarizer = SummarizerAgent(inference, prompts)

    # -----------------------------
    # PIPELINE RUNNER
    # -----------------------------
    runner = PipelineRunner(
        interpreter=interpreter,
        clarifier=clarifier,
        rag=rag,
        evaluator=evaluator,
        reviewer=reviewer,
        strategist=strategist,
        summarizer=summarizer,
    )

    final_state = runner.run(user_text)

    # -----------------------------
    # RESULTS
    # -----------------------------
    print("\n🧠 CALL TRACE (execution order):")
    print(final_state.debug_trace)
    print("\n----------------------------------------")

    print("\n📘 FINAL SUMMARY:")
    print(final_state.summary)
    print("\n----------------------------------------")

    print("\n🗂 FULL FINAL STATE:")
    print(final_state.to_dict())
    print("\n----------------------------------------\n")

    print("✨ Pipeline execution complete.\n")


# ------------------------------------------------------------
# CLI Entry Point
# ------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\n❗ Please provide an idea as CLI argument.")
        print("Example:")
        print("   python scripts/run_pipeline.py \"AI tool to manage founder ideas\"")
        sys.exit(1)

    user_text = " ".join(sys.argv[1:])
    run_pipeline(user_text)
