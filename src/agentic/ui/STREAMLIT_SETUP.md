# Streamlit Chat App Setup Guide

## Overview

The Streamlit chat app provides a user-friendly web interface for the OriginHub Agentic System API. It mirrors the terminal chat pipeline but with a modern UI.

## Prerequisites

1. **Running API Server**: The FastAPI server must be running
2. **Python 3.10+**: Same Python version as the main project
3. **Dependencies**: Install additional UI dependencies

## Installation

### 1. Install Streamlit

Add to your requirements or install directly:

```bash
pip install streamlit>=1.28.0 requests
```

Or update `requirements.txt`:

```
streamlit>=1.28.0
requests>=2.31.0
```

Then install:

```bash
pip install -r requirements.txt
```

### 2. Configuration

Create/update your `.env` file:

```bash
# API Configuration
API_BASE_URL=http://localhost:8000

# Model Configuration (inherited from API)
MODEL_7B_PATH=models/qwen-7b/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf
MODEL_1B_PATH=models/qwen-1.5b/model.gguf
MODEL_GPU_LAYERS=30

# Weaviate Configuration (inherited from API)
WEAVIATE_HOST=localhost
WEAVIATE_PORT=8081
WEAVIATE_GRPC_PORT=50051
```

## Running the Stack

### Step 1: Start the API Server

```bash
# Terminal 1: Start the FastAPI server
python src/agentic/api/app.py
```

Expected output:

```
[Startup] Model initialization completed in X.XXs
Uvicorn running on http://127.0.0.1:8000
```

### Step 2: Start Streamlit App

```bash
# Terminal 2: Start Streamlit
streamlit run src/agentic/ui/streamlit_chat_app.py
```

Expected output:

```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.X.X:8501
```

### Step 3: Open in Browser

Navigate to `http://localhost:8501`

## Features

### Chat Interface

- **Message Input**: Text area for typing ideas or follow-up questions
- **Real-time Response**: Messages display as they arrive from API
- **Conversation History**: All messages maintained in session
- **Auto-scroll**: Follows latest messages

### Session Management

- **Session Tracking**: Each conversation gets a unique session ID
- **Reset Button**: Clear chat history and start fresh
- **Session ID Display**: Shows current session identifier

### Analysis Metadata

- **Details Expander**: View analysis type, idea novelty, processing time
- **Status Indicators**: Shows whether analysis is complete
- **Error Messages**: Clear error reporting for debugging

### Sidebar Controls

- **API Health Check**: Visual indicator of API connection status
- **Configuration**: Change API URL without restarting
- **Session Info**: View current session ID
- **About Section**: Help and system explanation

## Usage Examples

### Example 1: New Business Idea Analysis

1. Type your idea:

```
An AI-powered platform that learns your daily context and sleep brainwave
patterns to generate personalized morning insights and solutions based on
your subconscious thinking.
```

2. Click "Send 📤"

3. Get analysis with:
   - Market research
   - Competitive landscape
   - SWOT analysis
   - Action plan
   - Success metrics

### Example 2: Follow-up Questions

After initial analysis completes:

1. Ask follow-up:

```
How should I approach the market research phase?
```

2. Get contextual answer based on previous analysis

## Troubleshooting

### "✗ API Disconnected"

**Problem**: Streamlit can't reach the API

**Solutions**:

```bash
# 1. Check API is running (Terminal 1)
# Should see "Uvicorn running on http://127.0.0.1:8000"

# 2. Check firewall isn't blocking port 8000
# Try accessing http://localhost:8000/health in browser

# 3. Update API_BASE_URL in .env
API_BASE_URL=http://localhost:8000

# 4. Try changing API URL in Streamlit sidebar
# Enter custom URL in "Configuration" section
```

### Messages Not Sending

**Problem**: Click "Send" but nothing happens

**Solutions**:

```bash
# 1. Check API logs (Terminal 1) for errors
# 2. Verify message is not empty
# 3. Check network connectivity
# 4. Try resetting session: click "🔄 Reset Chat"
```

### Slow Responses

**Normal behavior**: Initial analysis takes 30-60 seconds (model inference)

**If > 2 minutes**:

```bash
# 1. Check API logs for hanging requests
# 2. Verify Weaviate is running:
#    curl http://localhost:8081/v1/.well-known/ready

# 3. Check system resources (CPU/GPU/Memory)
```

### Session Not Saving

**Problem**: Messages disappear after refresh

**Expected behavior**: Streamlit sessions are browser-specific. Refreshing clears history.

**Solution**: Use the "🔄 Reset Chat" button instead of browser refresh to properly reset

## Advanced Configuration

### Custom Styling

Edit the `<style>` block in the Streamlit app to customize colors:

```python
st.markdown("""
<style>
    .user-message {
        background-color: #e3f2fd;  # Change blue to another color
        border-left: 4px solid #2196f3;
    }
</style>
""", unsafe_allow_html=True)
```

### Increase Request Timeout

If analysis takes longer than 60 seconds:

```python
response = requests.post(
    API_CHAT_ENDPOINT,
    json=payload,
    timeout=120  # Change from 60 to 120 seconds
)
```

### Custom Message Rendering

Modify `render_message()` function to customize how messages appear:

```python
def render_message(message: dict):
    # Add your custom rendering logic here
    pass
```

## Architecture

```
┌─────────────────────────────────────┐
│    Streamlit Web App (Port 8501)   │
│  - Chat UI                          │
│  - Session Management               │
│  - Message Display                  │
└─────────────┬───────────────────────┘
              │ HTTP Requests (JSON)
              ↓
┌─────────────────────────────────────┐
│    FastAPI Server (Port 8000)       │
│  - REST Endpoints                   │
│  - Session Routing                  │
│  - Agent Pipeline                   │
└─────────────┬───────────────────────┘
              │ Agent Orchestration
              ↓
┌─────────────────────────────────────┐
│    Multi-Agent System               │
│  - Interpreter, RAG, Strategist... │
│  - Weaviate Vector DB              │
│  - LLM Models (GPU-accelerated)     │
└─────────────────────────────────────┘
```

## Performance Tips

1. **First Run is Slow**: Models are loaded on first API call (~10-30s)
2. **Keep API Running**: Don't restart API between chat sessions
3. **GPU Memory**: Ensure 8GB+ VRAM for smooth operation
4. **Network**: Local LAN is faster than internet

## Security Notes

- **Local Deployment**: Current setup is for local/LAN use only
- **API Exposed**: API runs on public port - restrict network access
- **No Authentication**: Add auth layer before production use
- **Data Privacy**: Chat history stored only in browser session memory

## Common Issues & Solutions

| Issue                       | Cause                  | Solution                                            |
| --------------------------- | ---------------------- | --------------------------------------------------- |
| Port 8501 in use            | Another app using port | Change port: `streamlit run ... --server.port 8502` |
| Slow typing response        | Model is thinking      | Wait 30-60s, check GPU usage                        |
| Messages duplicate          | Browser caching        | Hard refresh: Ctrl+Shift+R (or Cmd+Shift+R on Mac)  |
| "No module named streamlit" | Not installed          | `pip install streamlit`                             |

## Next Steps

1. **Customize UI**: Modify colors, layout, messages in `streamlit_chat_app.py`
2. **Add Features**: Implement export, history saving, user profiles
3. **Deploy**: Use Streamlit Cloud for remote access
4. **Integrate Auth**: Add login system for multi-user

## Support

For issues or questions:

1. Check Streamlit docs: https://docs.streamlit.io
2. Review API logs for backend errors
3. Check FastAPI endpoints: http://localhost:8000/docs
