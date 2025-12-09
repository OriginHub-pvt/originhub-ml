"""
FastAPI application for OriginHub Agentic System.
Provides REST API endpoints for idea analysis and conversational AI.
"""

import os
import json
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from dotenv import load_dotenv
load_dotenv()

# Import models and inference
from src.agentic.ml.model_factory import create_model_manager
from src.agentic.ml.inference_engine import InferenceEngine
from src.agentic.prompts.prompt_builder import PromptBuilder

# Import agents
from src.agentic.agents.interpreter_agent import InterpreterAgent
from src.agentic.agents.clarifier_agent import ClarifierAgent
from src.agentic.agents.rag_agent import RAGAgent
from src.agentic.agents.reviewer_agent import ReviewerAgent
from src.agentic.agents.strategist_agent import StrategistAgent
from src.agentic.agents.evaluator_agent import EvaluatorAgent
from src.agentic.agents.summarizer_agent import SummarizerAgent

# Import pipeline and session management
from src.agentic.pipeline.interactive_pipeline_runner import InteractivePipelineRunner
from src.agentic.api.session_manager import SessionManager

# Import Weaviate
from weaviate import connect_to_local
from src.agentic.rag.weaviate_retriever import WeaviateRetriever

# ============================================================
# GLOBAL STATE
# ============================================================

app = FastAPI(
    title="OriginHub Agentic API",
    description="REST API for business idea analysis using multi-agent reasoning",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
_model_manager = None
_inference_engine = None
_prompts = None
_weaviate_client = None
_session_manager = SessionManager()
_pipeline_template = None
_is_ready = False


# ============================================================
# STARTUP / SHUTDOWN
# ============================================================

@app.on_event("startup")
async def startup_event():
    """Initialize models and connections on startup."""
    global _model_manager, _inference_engine, _prompts, _weaviate_client, _pipeline_template, _is_ready
    
    try:
        print("[API] Initializing models...")
        
        # Load model manager
        MODEL_7B_PATH = os.getenv("MODEL_7B_PATH", "models/qwen-7b/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf")
        MODEL_1B_PATH = os.getenv("MODEL_1B_PATH", "models/qwen-1.5b/model.gguf")
        
        _model_manager = create_model_manager(
            model7b_path=MODEL_7B_PATH,
            model1b_path=MODEL_1B_PATH,
            context_size=int(os.getenv("MODEL_CONTEXT_SIZE", 4096)),
            n_threads=int(os.getenv("MODEL_N_THREADS", 4)),
            gpu_layers=int(os.getenv("MODEL_GPU_LAYERS", 20)),
            preload=os.getenv("MODEL_PRELOAD", "false").lower() != "false",
        )
        
        _inference_engine = InferenceEngine(_model_manager)
        _prompts = PromptBuilder()
        print("[API] Models loaded successfully")
        
        # Connect to Weaviate
        print("[API] Connecting to Weaviate...")
        _weaviate_client = connect_to_local(
            host=os.getenv("WEAVIATE_HOST", "localhost"),
            port=int(os.getenv("WEAVIATE_PORT", 8081)),
            grpc_port=int(os.getenv("WEAVIATE_GRPC_PORT", 50051)),
        )
        print("[API] Weaviate connected successfully")
        
        # Create pipeline template
        collection_name = os.getenv("WEAVIATE_COLLECTION", "ArticleSummary")
        retriever = WeaviateRetriever(_weaviate_client, collection_name)
        
        interpreter = InterpreterAgent(_inference_engine, _prompts)
        clarifier = ClarifierAgent(_inference_engine, _prompts)
        rag = RAGAgent(retriever)
        evaluator = EvaluatorAgent()
        reviewer = ReviewerAgent(_inference_engine, _prompts)
        strategist = StrategistAgent(_inference_engine, _prompts)
        summarizer = SummarizerAgent(_inference_engine, _prompts)
        
        _pipeline_template = {
            "interpreter": interpreter,
            "clarifier": clarifier,
            "rag": rag,
            "evaluator": evaluator,
            "reviewer": reviewer,
            "strategist": strategist,
            "summarizer": summarizer,
        }
        
        _is_ready = True
        print("[API] ✓ API startup complete and ready to serve requests")
        
    except Exception as e:
        print(f"[API] ✗ Startup failed: {str(e)}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global _weaviate_client
    if _weaviate_client:
        _weaviate_client.close()
        print("[API] Weaviate client closed")


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy" if _is_ready else "initializing",
        "models_loaded": _model_manager is not None,
        "weaviate_connected": _weaviate_client is not None,
        "timestamp": datetime.now().isoformat(),
    }


# ============================================================
# SESSION MANAGEMENT
# ============================================================

@app.post("/sessions")
async def create_session():
    """Create a new conversation session."""
    if not _is_ready:
        raise HTTPException(status_code=503, detail="API not ready yet")
    
    # Create new runner instance
    runner = InteractivePipelineRunner(
        interpreter=_pipeline_template["interpreter"],
        clarifier=_pipeline_template["clarifier"],
        rag=_pipeline_template["rag"],
        evaluator=_pipeline_template["evaluator"],
        reviewer=_pipeline_template["reviewer"],
        strategist=_pipeline_template["strategist"],
        summarizer=_pipeline_template["summarizer"],
    )
    
    session_id = _session_manager.create_session(runner)
    
    return {
        "session_id": session_id,
        "created_at": datetime.now().isoformat(),
        "message": "Session created successfully"
    }


@app.get("/sessions/{session_id}")
async def get_session_info(session_id: str):
    """Get session information."""
    info = _session_manager.get_session_info(session_id)
    if not info:
        raise HTTPException(status_code=404, detail="Session not found")
    return info


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session."""
    success = _session_manager.delete_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"message": "Session deleted successfully"}


# ============================================================
# MESSAGE / ANALYSIS ENDPOINTS
# ============================================================

@app.post("/chat/{session_id}")
async def send_message(session_id: str, request: dict):
    """
    Send a message in a conversation session.
    
    Request body:
    {
        "message": "Your business idea or follow-up question"
    }
    """
    if not _is_ready:
        raise HTTPException(status_code=503, detail="API not ready yet")
    
    runner = _session_manager.get_session(session_id)
    if not runner:
        raise HTTPException(status_code=404, detail="Session not found")
    
    message = request.get("message", "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    try:
        # Process message
        response = runner.handle_user_message(message)
        
        # Increment message counter
        _session_manager.increment_message(session_id)
        
        # Parse response if JSON
        try:
            if isinstance(response, str) and response.strip().startswith("{"):
                response_data = json.loads(response)
            else:
                response_data = response
        except:
            response_data = response
        
        return {
            "response": response_data,
            "analysis_complete": runner.analysis_complete,
            "conversation_history": runner.conversation_history,
            "session_id": session_id,
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")


# ============================================================
# UTILITY ENDPOINTS
# ============================================================

@app.get("/info")
async def api_info():
    """Get API information."""
    return {
        "name": "OriginHub Agentic System API",
        "version": "1.0.0",
        "description": "REST API for multi-agent business idea analysis",
        "endpoints": {
            "health": "GET /health",
            "create_session": "POST /sessions",
            "get_session": "GET /sessions/{session_id}",
            "delete_session": "DELETE /sessions/{session_id}",
            "send_message": "POST /chat/{session_id}",
            "api_info": "GET /info",
        }
    }


def run(host: str = "0.0.0.0", port: int = 8000):
    """Run the API server."""
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )


if __name__ == "__main__":
    run()
