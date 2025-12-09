"""
Session manager for handling multiple concurrent chat sessions.
"""

import uuid
from datetime import datetime
from typing import Dict, Optional
from src.agentic.pipeline.interactive_pipeline_runner import InteractivePipelineRunner


class SessionManager:
    """Manages multiple conversation sessions."""
    
    def __init__(self):
        self.sessions: Dict[str, Dict] = {}
    
    def create_session(self, runner: InteractivePipelineRunner) -> str:
        """Create a new session."""
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "runner": runner,
            "created_at": datetime.now(),
            "last_activity": datetime.now(),
            "message_count": 0,
        }
        return session_id
    
    def get_session(self, session_id: str) -> Optional[InteractivePipelineRunner]:
        """Get runner for a session."""
        if session_id not in self.sessions:
            return None
        self.sessions[session_id]["last_activity"] = datetime.now()
        return self.sessions[session_id]["runner"]
    
    def increment_message(self, session_id: str):
        """Increment message count for session."""
        if session_id in self.sessions:
            self.sessions[session_id]["message_count"] += 1
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False
    
    def get_session_info(self, session_id: str) -> Optional[Dict]:
        """Get session metadata."""
        if session_id not in self.sessions:
            return None
        session = self.sessions[session_id]
        return {
            "session_id": session_id,
            "created_at": session["created_at"].isoformat(),
            "last_activity": session["last_activity"].isoformat(),
            "message_count": session["message_count"],
        }
    
    def cleanup_old_sessions(self, max_age_hours: int = 24):
        """Remove sessions older than max_age_hours."""
        now = datetime.now()
        expired = []
        for session_id, session in self.sessions.items():
            age = (now - session["last_activity"]).total_seconds() / 3600
            if age > max_age_hours:
                expired.append(session_id)
        
        for session_id in expired:
            del self.sessions[session_id]
        
        return len(expired)
