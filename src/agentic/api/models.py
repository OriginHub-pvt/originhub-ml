"""
Data models for the API.
Defines request/response schemas.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class MessageRequest(BaseModel):
    """User message request."""
    message: str = Field(..., description="User message to process")
    session_id: Optional[str] = Field(None, description="Session ID for conversation tracking")


class AnalysisResult(BaseModel):
    """Analysis result in structured format."""
    final_summary: Optional[Dict[str, Any]] = Field(None, description="Final analysis summary")
    next_steps: Optional[List[str]] = Field(None, description="Recommended next steps")
    market_research: Optional[str] = Field(None, description="Market research findings")
    competitive_analysis: Optional[str] = Field(None, description="Competitive landscape")
    swot: Optional[Dict[str, Any]] = Field(None, description="SWOT analysis")


class MessageResponse(BaseModel):
    """API response for user message."""
    response: str = Field(..., description="Response text or JSON")
    analysis_complete: bool = Field(False, description="Whether analysis is complete")
    session_id: str = Field(..., description="Session ID for tracking")
    conversation_history: List[Dict[str, str]] = Field(default_factory=list, description="Conversation history")


class HealthCheckResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Status (healthy/unhealthy)")
    models_loaded: bool = Field(..., description="Whether models are loaded")
    weaviate_connected: bool = Field(..., description="Whether Weaviate is connected")
    message: str = Field(..., description="Status message")


class SessionInfo(BaseModel):
    """Session information."""
    session_id: str = Field(..., description="Unique session ID")
    created_at: str = Field(..., description="Session creation timestamp")
    last_activity: str = Field(..., description="Last activity timestamp")
    conversation_count: int = Field(..., description="Number of messages in session")
