"""
FastAPI application for OriginHub Agentic System.
Provides REST API endpoints for idea analysis and conversational AI.
"""

import os
import json
import logging
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from dotenv import load_dotenv
load_dotenv()

# Configure logging with environment variable support
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
try:
    log_level_value = getattr(logging, log_level)
except AttributeError:
    log_level_value = logging.INFO
    log_level = "INFO"

logging.basicConfig(
    level=log_level_value,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.info(f"Logging level set to: {log_level}")

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
from weaviate import connect_to_local, connect_to_custom
from src.agentic.rag.weaviate_retriever import WeaviateRetriever

# Import Prometheus Instrumentator
from prometheus_fastapi_instrumentator import Instrumentator

# ============================================================
# GLOBAL STATE
# ============================================================

app = FastAPI(
    title="OriginHub Agentic API",
    description="REST API for business idea analysis using multi-agent reasoning",
    version="1.0.0"
)

Instrumentator().instrument(app).expose(app)

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
        logger.info("Initializing models...")
        
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
        logger.info("Models loaded successfully")
        
        # Connect to Weaviate
        logger.info("Connecting to Weaviate...")
        weaviate_host = os.getenv("WEAVIATE_HOST", "localhost")
        weaviate_port = int(os.getenv("WEAVIATE_PORT", 8081))
        weaviate_grpc_host = os.getenv("WEAVIATE_GRPC_HOST", weaviate_host)
        weaviate_grpc_port = int(os.getenv("WEAVIATE_GRPC_PORT", 50051))
        
        logger.info(f"Weaviate config: host={weaviate_host}:{weaviate_port}, grpc={weaviate_grpc_host}:{weaviate_grpc_port}")
        
        # Check if it's a remote connection
        is_remote = weaviate_host not in ["localhost", "127.0.0.1"]
        
        try:
            if is_remote:
                # For remote Weaviate instances, use HTTP connection with gRPC
                from weaviate import connect_to_custom
                _weaviate_client = connect_to_custom(
                    http_host=weaviate_host,
                    http_port=weaviate_port,
                    http_secure=False,
                    grpc_host=weaviate_grpc_host,
                    grpc_port=weaviate_grpc_port,
                    grpc_secure=False,
                )
            else:
                # For local connections, use connect_to_local
                _weaviate_client = connect_to_local(
                    host=weaviate_host,
                    port=weaviate_port,
                    grpc_port=weaviate_grpc_port,
                )
            logger.info("Weaviate connected successfully")
        except Exception as e:
            logger.warning(f"Failed to connect to Weaviate: {e}. RAG functionality will be limited.")
            logger.warning("Make sure HUGGINGFACE_APIKEY is set in environment variables if using Hugging Face vectorizer.")
            _weaviate_client = None
        
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
        logger.info("API startup complete and ready to serve requests")
        
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}", exc_info=True)
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global _weaviate_client
    if _weaviate_client:
        _weaviate_client.close()
        logger.info("Weaviate client closed")


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
    logger.info("Creating new session")
    if not _is_ready:
        logger.error("API not ready to create session")
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
    logger.info(f"Session created: {session_id}")
    
    return {
        "session_id": session_id,
        "created_at": datetime.now().isoformat(),
        "message": "Session created successfully"
    }


@app.get("/sessions/{session_id}")
async def get_session_info(session_id: str):
    """Get session information."""
    logger.info(f"Retrieving session info: {session_id}")
    info = _session_manager.get_session_info(session_id)
    if not info:
        logger.warning(f"Session not found: {session_id}")
        raise HTTPException(status_code=404, detail="Session not found")
    return info


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session."""
    logger.info(f"Deleting session: {session_id}")
    success = _session_manager.delete_session(session_id)
    if not success:
        logger.warning(f"Session not found for deletion: {session_id}")
        raise HTTPException(status_code=404, detail="Session not found")
    logger.info(f"Session deleted: {session_id}")
    return {"message": "Session deleted successfully"}


# ============================================================
# MESSAGE / ANALYSIS ENDPOINTS
# ============================================================

def _ensure_text_format(response: any) -> str:
    """
    Ensure response is plain text.
    Handles strings, dicts, and other types.
    """
    if isinstance(response, str):
        return response if response.strip() else "No response generated"
    elif isinstance(response, dict):
        return str(response)
    elif response is None:
        return "Analysis in progress..."
    else:
        return str(response)


def _format_conversation_history(history: list) -> list:
    """
    Replace newlines with <br> tags in conversation history.
    """
    formatted_history = []
    for msg in history:
        formatted_msg = msg.copy()
        if isinstance(formatted_msg.get("content"), str):
            formatted_msg["content"] = formatted_msg["content"].replace("\n", "<br>")
        formatted_history.append(formatted_msg)
    return formatted_history


def _handle_with_default_llm(message: str, inference_engine, conversation_history: list = None) -> str:
    """
    Default fallback handler that uses just the LLM to respond to any query.
    Used when full pipeline fails or as a catch-all fallback mode.
    """
    logger.info("Using default LLM handler mode")
    
    # Build a simple prompt with conversation context if available
    if conversation_history and len(conversation_history) > 0:
        # Include last few messages for context
        recent = conversation_history[-6:] if len(conversation_history) > 6 else conversation_history
        context = "Conversation History:\n"
        for msg in recent:
            context += f"- {msg['role'].upper()}: {msg['content'][:100]}...\n" if len(msg['content']) > 100 else f"- {msg['role'].upper()}: {msg['content']}\n"
        prompt = f"{context}\nUser's Current Question: {message}\n\nProvide a helpful response:"
    else:
        prompt = f"User: {message}\n\nAssistant: Please provide a helpful response."
    
    try:
        response = inference_engine.generate(
            prompt=prompt,
            heavy=False,  # Use light model for speed
            max_tokens=512,
            temperature=0.7,
            top_p=0.9,
            stop=None,
        )
        logger.info("Default LLM handler completed successfully")
        return response
    except Exception as e:
        logger.error(f"Default LLM handler failed: {e}", exc_info=True)
        return "I apologize, but I encountered an error processing your request. Please try again."


def _is_greeting_or_small_talk(message: str, inference_engine) -> bool:
    """
    Use LLM to detect if the message is just a greeting or small talk.
    Returns True if it should be handled with a simple greeting response.
    Returns False if it looks like a substantive query.
    """
    # Quick length check - very short messages are likely greetings
    if len(message.strip()) > 150:
        return False
    
    prompt = f"""Determine if the following message is just a GREETING or SMALL TALK, or if it's a SUBSTANTIVE QUERY that needs real analysis.

    Message: "{message}"

    Respond with ONLY one word:
    - "GREETING" if it's a hello, hi, how are you, thanks, or casual small talk
    - "QUERY" if it's asking about something substantial that needs real assistance

    Response:"""
    
    try:
        response = inference_engine.generate(
            prompt=prompt,
            heavy=False,
            max_tokens=10,
            temperature=0.0,  # Deterministic
            top_p=1.0,
            stop=None,
        ).strip().upper()
        
        logger.debug(f"Message classification (greeting): {response}")
        return response.startswith("GREETING")
    except Exception as e:
        logger.warning(f"Error detecting greeting: {e}. Defaulting to substantive query.")
        return False


def _handle_greeting(message: str, inference_engine) -> str:
    """
    Handle greeting messages with a friendly LLM response.
    """
    logger.info("Handling greeting/small talk")
    
    prompt = f"""The user just sent you a greeting or casual message. Respond warmly and briefly, then offer to help them with business idea analysis.

User: {message}

Your response (keep it brief and friendly, max 2 sentences):"""
    
    try:
        response = inference_engine.generate(
            prompt=prompt,
            heavy=False,
            max_tokens=100,
            temperature=0.7,
            top_p=0.9,
            stop=None,
        )
        logger.info("Greeting handled successfully")
        return response
    except Exception as e:
        logger.error(f"Error handling greeting: {e}", exc_info=True)
        return "Hello! I'm here to help you analyze your business ideas. What can I help you with?"


def _is_business_idea_query(message: str, inference_engine) -> bool:
    """
    Use LLM to determine if the message is about a new business idea or a follow-up question.
    Returns True if it should trigger the full agentic pipeline (new business idea).
    Returns False if it's a follow-up question about existing analysis.
    """
    prompt = f"""Analyze the following user message and determine if it describes a NEW BUSINESS IDEA/CONCEPT or if it is a FOLLOW-UP QUESTION about an existing analysis.

    User Message: "{message}"

    Respond with ONLY one word:
    - "IDEA" if the message describes a new business concept, startup idea, product, service, or business opportunity
    - "FOLLOWUP" if the message is asking questions about existing analysis, clarifications, or expansion on previous discussion

    Response:"""
    
    try:
        response = inference_engine.generate(
            prompt=prompt,
            heavy=False,  # Use light model for speed
            max_tokens=10,
            temperature=0.0,  # Deterministic
            top_p=1.0,
            stop=None,
        ).strip().upper()
        
        logger.debug(f"Query classification: {response}")
        return response.startswith("IDEA")
    except Exception as e:
        logger.warning(f"Error classifying query: {e}. Defaulting to business idea.")
        # Default to treating as idea if LLM fails
        return True




@app.post("/chat/{session_id}")
async def send_message(session_id: str, request: dict):
    """
    Send a message in a conversation session.
    If session doesn't exist, creates a new one.
    
    Request body:
    {
        "message": "Your business idea or follow-up question"
    }
    
    Response:
    {
        "response": "Markdown-formatted analysis",
        "response_type": "markdown",
        "analysis_complete": true/false,
        "conversation_history": [...],
        "session_id": "..."
    }
    """
    if not _is_ready:
        logger.error("API not ready to process message")
        raise HTTPException(status_code=503, detail="API not ready yet")
    
    runner = _session_manager.get_session(session_id)
    if not runner:
        logger.info(f"Session not found: {session_id}. Creating new session.")
        # Create new session if it doesn't exist
        runner = InteractivePipelineRunner(
            interpreter=_pipeline_template["interpreter"],
            clarifier=_pipeline_template["clarifier"],
            rag=_pipeline_template["rag"],
            evaluator=_pipeline_template["evaluator"],
            reviewer=_pipeline_template["reviewer"],
            strategist=_pipeline_template["strategist"],
            summarizer=_pipeline_template["summarizer"],
        )
        _session_manager.create_session(runner, session_id)
        logger.info(f"New session created: {session_id}")
    
    message = request.get("message", "").strip()
    if not message:
        logger.warning(f"Empty message for session: {session_id}")
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    try:
        logger.info(f"Processing message for session {session_id}")
        
        # Check if this is just a greeting or small talk
        if _is_greeting_or_small_talk(message, _inference_engine):
            logger.info(f"Detected greeting/small talk for session {session_id}")
            response = _handle_greeting(message, _inference_engine)
        else:
            # Check if this is a new business idea query or a follow-up question
            is_new_idea = _is_business_idea_query(message, _inference_engine)
            logger.info(f"Query type: {'Business Idea' if is_new_idea else 'Follow-up Question'}")
            
            # If it's not a business idea and session is already initialized, handle as follow-up
            if not is_new_idea and runner.state.interpreted is not None:
                logger.info(f"Handling as follow-up question for session {session_id}")
                response = runner._handle_followup_question(message)
            else:
                # Process as new idea through the full pipeline
                logger.info(f"Processing as new business idea for session {session_id}")
                response = runner.handle_user_message(message)
        
        # Increment message counter
        _session_manager.increment_message(session_id)
        
        # Ensure response is plain text formatted
        text_response = _ensure_text_format(response)
        # Replace newlines with HTML line breaks
        text_response = text_response.replace("\n", "<br>")
        logger.info(f"Message processed successfully for session {session_id}")
        
        return {
            "response": text_response,
            "response_type": "text",
            "analysis_complete": runner.analysis_complete,
            "conversation_history": _format_conversation_history(runner.conversation_history),
            "session_id": session_id,
        }
        
    except Exception as e:
        logger.error(f"Error processing message for session {session_id}: {str(e)}", exc_info=True)
        logger.info(f"Falling back to default LLM handler for session {session_id}")
        
        # Try the default LLM handler as fallback
        try:
            fallback_response = _handle_with_default_llm(message, _inference_engine, runner.conversation_history)
            runner.conversation_history.append({"role": "assistant", "content": fallback_response})
            
            # Replace newlines with HTML line breaks
            fallback_text = _ensure_text_format(fallback_response).replace("\n", "<br>")
            
            return {
                "response": fallback_text,
                "response_type": "text",
                "analysis_complete": False,
                "conversation_history": _format_conversation_history(runner.conversation_history),
                "session_id": session_id,
                "mode": "fallback_llm"
            }
        except Exception as fallback_error:
            logger.error(f"Fallback LLM handler also failed: {str(fallback_error)}", exc_info=True)
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


def run(host: str = None, port: int = None):
    """Run the API server."""
    # Read from environment variables if not provided
    host = host or os.getenv("API_HOST", "0.0.0.0")
    port = port or int(os.getenv("API_PORT", 8000))
    
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )


if __name__ == "__main__":
    run()
