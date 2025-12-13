"""
Streamlit Chat Application for OriginHub Agentic System

A user-friendly chat interface that connects to the running FastAPI server.
Provides conversational analysis of business ideas with follow-up question support.

Usage:
    streamlit run src/agentic/ui/streamlit_chat_app.py
"""

import streamlit as st
import requests
import json
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# Configuration
# ============================================================
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8004")
API_HEALTH_ENDPOINT = f"{API_BASE_URL}/health"
API_CREATE_SESSION_ENDPOINT = f"{API_BASE_URL}/sessions"
API_CHAT_ENDPOINT_TEMPLATE = f"{API_BASE_URL}/chat"
API_DELETE_SESSION_ENDPOINT_TEMPLATE = f"{API_BASE_URL}/sessions"

# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="OriginHub Agentic Chat",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Styling
# ============================================================
st.markdown("""
<style>
    .main-container {
        max-width: 1000px;
        margin: 0 auto;
    }
    
    .message-container {
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
    }
    
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    
    .assistant-message {
        background-color: #f5f5f5;
        border-left: 4px solid #4caf50;
    }
    
    .system-message {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
    }
    
    .error-message {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
    }
    
    .metadata {
        font-size: 0.85em;
        color: #666;
        margin-top: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Session State Initialization
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.session_id = None
    st.session_state.analysis_complete = False

if "api_status" not in st.session_state:
    st.session_state.api_status = None


# ============================================================
# Helper Functions
# ============================================================
def check_api_health() -> bool:
    """Check if API is running and healthy."""
    try:
        response = requests.get(API_HEALTH_ENDPOINT, timeout=5)
        return response.status_code == 200
    except Exception as e:
        st.error(f"Failed to connect to API: {str(e)}")
        return False


def send_message_to_api(message: str) -> Optional[dict]:
    """Send message to API and get response."""
    try:
        # If no session exists, create one first
        if not st.session_state.session_id:
            session_response = requests.post(
                API_CREATE_SESSION_ENDPOINT,
                timeout=10
            )
            if session_response.status_code != 200:
                st.error(f"Failed to create session: {session_response.status_code}")
                return None
            st.session_state.session_id = session_response.json()["session_id"]
        
        # Send message to existing session
        payload = {"message": message}
        
        response = requests.post(
            f"{API_CHAT_ENDPOINT_TEMPLATE}/{st.session_state.session_id}",
            json=payload,
            timeout=60  # Longer timeout for analysis
        )
        print(response.text)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.Timeout:
        st.error("Request timed out. Please try again.")
        return None
    except Exception as e:
        st.error(f"Failed to send message: {str(e)}")
        return None


def reset_session():
    """Reset the chat session."""
    try:
        if st.session_state.session_id:
            requests.delete(
                f"{API_DELETE_SESSION_ENDPOINT_TEMPLATE}/{st.session_state.session_id}",
                timeout=5
            )
        st.session_state.messages = []
        st.session_state.session_id = None
        st.session_state.analysis_complete = False
        st.success("Session reset successfully!")
    except Exception as e:
        st.error(f"Failed to reset session: {str(e)}")


def render_message(message: dict):
    """Render a single message in the chat."""
    role = message.get("role", "assistant")
    content = message.get("content", "")
    metadata = message.get("metadata", {})
    
    if role == "user":
        css_class = "user-message"
        title = "You"
    elif role == "system":
        css_class = "system-message"
        title = "System"
    else:
        css_class = "assistant-message"
        title = "Assistant"
    
    st.markdown(f"""
    <div class="message-container {css_class}">
        <strong>{title}</strong><br>
        {content}
    </div>
    """, unsafe_allow_html=True)
    
    # Show metadata if available
    if metadata and metadata.get("show_details"):
        with st.expander("View Details"):
            if "analysis_type" in metadata:
                st.write(f"**Analysis Type**: {metadata['analysis_type']}")
            if "is_new_idea" in metadata:
                st.write(f"**New Idea**: {'Yes' if metadata['is_new_idea'] else 'No'}")
            if "processing_time" in metadata:
                st.write(f"**Processing Time**: {metadata['processing_time']:.2f}s")


# ============================================================
# Main App
# ============================================================
def main():
    # Header
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.title("OriginHub Agentic Chat")
        st.markdown("Intelligent business idea analysis with AI agents")
    
    with col2:
        # API Status Indicator
        if check_api_health():
            st.success("API Connected")
            st.session_state.api_status = True
        else:
            st.error("API Disconnected")
            st.session_state.api_status = False
            st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        
        # Session info
        st.subheader("Session")
        if st.session_state.session_id:
            st.write(f"**Session ID**: `{st.session_state.session_id[:8]}...`")
        
        # Reset button
        if st.button("Reset Chat", use_container_width=True):
            reset_session()
            st.experimental_rerun()
        
        st.divider()
        
        # Information
        st.subheader("About")
        st.markdown("""
        This app connects to the **OriginHub Agentic System** API.
        
        ### How it works:
        1. Describe your business idea
        2. The system analyzes it using multiple AI agents
        3. Get strategic insights and market analysis
        4. Ask follow-up questions for deeper understanding
        
        ### Agents:
        - **Interpreter**: Structures your idea
        - **RAG**: Searches for similar ideas
        - **Evaluator**: Routes to analysis
        - **Strategist/Reviewer**: Analyzes idea
        - **Summarizer**: Creates final report
        """)
        
        st.divider()
        
        # Configuration
        st.subheader("Configuration")
        api_url = st.text_input(
            "API URL",
            value=API_BASE_URL,
            help="Base URL of the running API server"
        )
        if api_url != API_BASE_URL:
            os.environ["API_BASE_URL"] = api_url
    
    # Main chat area
    st.markdown("---")
    
    # Display chat history
    st.subheader("Conversation")
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.messages:
            render_message(message)
    
    # Input area
    st.markdown("---")
    
    col1, col2 = st.columns([0.85, 0.15])
    
    with col1:
        user_input = st.text_area(
            "Your message:",
            placeholder="Describe your business idea or ask a follow-up question...",
            height=80,
            key="user_input"
        )
    
    with col2:
        st.write("")  # Spacing
        send_button = st.button(
            "Send",
            use_container_width=True,
            type="primary",
            key="send_button"
        )
    
    # Handle message sending
    if send_button and user_input.strip():
        # Add user message to history
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })
        
        # Show loading state
        with st.spinner("Analyzing your idea..."):
            response = send_message_to_api(user_input)
        
        if response:
            # Store session ID
            if "session_id" in response:
                st.session_state.session_id = response["session_id"]
            
            # Add assistant response to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": response.get("response", ""),
                "metadata": response.get("metadata", {})
            })
            
            # Update analysis status
            if response.get("analysis_complete"):
                st.session_state.analysis_complete = True
            
            # Rerun to display new message
            st.experimental_rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9em;">
        OriginHub Agentic System | <a href="https://github.com/OriginHub-pvt/originhub-ml">GitHub</a>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
