#!/usr/bin/env python3
"""
Interactive conversational runner for the OriginHub multi-agent system.
Behaves like ChatGPT in the terminal.
"""

import sys
import os

from dotenv import load_dotenv
load_dotenv()

# -------------------------------
#   MODEL / PROMPTS
# -------------------------------
from src.agentic.ml.model_factory import create_model_manager, get_backend_info
from src.agentic.ml.inference_engine import InferenceEngine
from src.agentic.prompts.prompt_builder import PromptBuilder

# -------------------------------
#   AGENTS
# -------------------------------
from src.agentic.agents.interpreter_agent import InterpreterAgent
from src.agentic.agents.clarifier_agent import ClarifierAgent
from src.agentic.agents.rag_agent import RAGAgent
from src.agentic.agents.reviewer_agent import ReviewerAgent
from src.agentic.agents.strategist_agent import StrategistAgent
from src.agentic.agents.evaluator_agent import EvaluatorAgent
from src.agentic.agents.summarizer_agent import SummarizerAgent

# -------------------------------
#   RETRIEVER (Weaviate)
# -------------------------------
from weaviate import connect_to_local
from src.agentic.rag.weaviate_retriever import WeaviateRetriever

# -------------------------------
#   INTERACTIVE PIPELINE
# -------------------------------
from src.agentic.pipeline.interactive_pipeline_runner import InteractivePipelineRunner

from load_dotenv import load_dotenv

load_dotenv()

def main():
    print("\n🧠 OriginHub Interactive Agentic Assistant")
    print("Type your messages below. Type 'exit' to quit.\n")

    # ------------------------------------------------------------
    # Load model paths from environment so users can override defaults
    # and preload models before entering the interactive loop.
    # ------------------------------------------------------------
    # Model paths are handled by the factory based on backend

    import time

    # Show backend information
    get_backend_info()
    MODEL_7B_PATH = os.getenv("MODEL_7B_PATH", "models/qwen-7b/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf")
    MODEL_1B_PATH = os.getenv("MODEL_1B_PATH", "models/qwen-1.5b/model.gguf")
    print("\n🔧 Loading models (this may take a moment)...\n")
    preload_models = os.getenv("MODEL_PRELOAD", "true").lower() not in ("0", "false", "no")

    t_start_models = time.time()
    model_manager = create_model_manager(
        model7b_path=MODEL_7B_PATH,
        model1b_path=MODEL_1B_PATH,
        context_size=int(os.getenv("MODEL_CONTEXT_SIZE", 4096)),
        n_threads=int(os.getenv("MODEL_N_THREADS", 4)),
        gpu_layers=int(os.getenv("MODEL_GPU_LAYERS", 20)),
        preload=preload_models,
    )
    if hasattr(model_manager, 'preload_all'):
        model_manager.preload_all()
    t_end_models = time.time()
    print(f"[Startup] Model initialization completed in {t_end_models - t_start_models:.2f}s")

    inference = InferenceEngine(model_manager)
    prompts = PromptBuilder()

    print("✅ Models loaded. Starting interactive session.\n")

    # If MODEL_PRELOAD=true models were loaded above and the app will only
    # start after model load. If MODEL_PRELOAD=false the app starts without
    # loading models; models will be instantiated on-demand at first use.

    # ============================================================
    # 2) CONNECT TO WEAVIATE
    # ============================================================
    client = connect_to_local(
        host=os.getenv("WEAVIATE_HOST", "localhost"),
        port=int(os.getenv("WEAVIATE_PORT", 8081)),
        grpc_port=int(os.getenv("WEAVIATE_GRPC_PORT", 50051)),
    )

    # build retriever wrapper
    collection_name = os.getenv("WEAVIATE_COLLECTION", "ArticleSummary")
    retriever = WeaviateRetriever(client, collection_name)

    # ============================================================
    # 3) INSTANTIATE AGENTS
    # ============================================================
    interpreter = InterpreterAgent(inference, prompts)
    clarifier = ClarifierAgent(inference, prompts)

    # IMPORTANT: RAGAgent takes ONLY the retriever
    rag = RAGAgent(retriever)

    # IMPORTANT: EvaluatorAgent takes NO inference/prompt
    evaluator = EvaluatorAgent()

    reviewer = ReviewerAgent(inference, prompts)
    strategist = StrategistAgent(inference, prompts)
    summarizer = SummarizerAgent(inference, prompts)

    # ============================================================
    # 4) CREATE INTERACTIVE PIPELINE RUNNER
    # ============================================================
    runner = InteractivePipelineRunner(
        interpreter=interpreter,
        clarifier=clarifier,
        rag=rag,
        evaluator=evaluator,
        reviewer=reviewer,
        strategist=strategist,
        summarizer=summarizer,
    )

    # ============================================================
    # 5) INTERACTIVE CHAT LOOP
    # ============================================================
    while True:
        user_message = input("You: ")

        if user_message.lower().strip() == "exit":
            print("Goodbye 👋")
            break

        response = runner.handle_user_message(user_message)
        # Pretty-print JSON responses for readability
        try:
            import json as _json

            if isinstance(response, dict):
                pretty = _json.dumps(response, indent=2)
                print(f"\nAgent 🤖: {pretty}\n")
            else:
                print(f"\nAgent 🤖: {response}\n")
        except Exception:
            print(f"\nAgent 🤖: {response}\n")

        # SummarizerAgent finished the pipeline
        if runner.is_done:
            print("\n🎉 Final summary produced. Conversation closed.\n")
            break

    # cleanup
    client.close()


if __name__ == "__main__":
    main()
