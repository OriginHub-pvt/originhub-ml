# OriginHub API Quick Reference

## Start API Server

```bash
python src/agentic/scripts/api_server.py
```

## Interactive API Docs

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Basic Flow

### 1. Create Session

```bash
curl -X POST http://localhost:8000/sessions
```

Response:

```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "created_at": "2025-12-08T10:30:00.123456"
}
```

### 2. Send Message

```bash
curl -X POST http://localhost:8000/chat/{session_id} \
  -H "Content-Type: application/json" \
  -d '{"message":"AI app for personalized fitness"}'
```

Response:

```json
{
  "response": {
    "final_summary": "...",
    "next_steps": ["..."],
    "market_research": "...",
    "competitive_analysis": "...",
    "swot": {...}
  },
  "analysis_complete": false,
  "conversation_history": [...],
  "session_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

### 3. Ask Follow-up Questions (if analysis_complete)

```bash
curl -X POST http://localhost:8000/chat/{session_id} \
  -H "Content-Type: application/json" \
  -d '{"message":"Tell me more about the market research"}'
```

### 4. Delete Session

```bash
curl -X DELETE http://localhost:8000/sessions/{session_id}
```

## Python Integration

```python
import requests

# Create session
session_res = requests.post('http://localhost:8000/sessions')
session_id = session_res.json()['session_id']

# Send message
msg_res = requests.post(
    f'http://localhost:8000/chat/{session_id}',
    json={'message': 'Your business idea'}
)
result = msg_res.json()

# Get response
print(result['response'])
print(f"Analysis complete: {result['analysis_complete']}")

# Ask follow-up
if result['analysis_complete']:
    followup_res = requests.post(
        f'http://localhost:8000/chat/{session_id}',
        json={'message': 'More details please'}
    )
    print(followup_res.json()['response'])
```

## JavaScript/React Integration

```javascript
const apiUrl = "http://localhost:8000";

// Create session
const sessionRes = await fetch(`${apiUrl}/sessions`, { method: "POST" });
const { session_id } = await sessionRes.json();

// Send message
const msgRes = await fetch(`${apiUrl}/chat/${session_id}`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ message: "AI app idea" }),
});
const result = await msgRes.json();

// Display response
console.log(result.response);

// Ask follow-up if analysis complete
if (result.analysis_complete) {
  const followupRes = await fetch(`${apiUrl}/chat/${session_id}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message: "Follow-up question" }),
  });
  console.log(await followupRes.json());
}

// Cleanup
await fetch(`${apiUrl}/sessions/${session_id}`, { method: "DELETE" });
```

## Configuration (.env)

```bash
# API Server
API_HOST=0.0.0.0
API_PORT=8000

# Models
MODEL_7B_PATH=models/qwen-7b/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf
MODEL_1B_PATH=models/qwen-1.5b/model.gguf
MODEL_GPU_LAYERS=30
MODEL_PRELOAD=false

# Weaviate
WEAVIATE_HOST=localhost
WEAVIATE_PORT=8081
```

## Health Check

```bash
curl http://localhost:8000/health
```

## API Info

```bash
curl http://localhost:8000/info
```

## Common Status Codes

- **200 OK** - Success
- **400 Bad Request** - Invalid message or parameters
- **404 Not Found** - Session not found
- **503 Service Unavailable** - Models still loading
- **500 Internal Server Error** - Processing error

## Architecture

```
User/UI
  ↓
HTTP Request (REST API)
  ↓
FastAPI App
  ↓
Session Manager (tracks conversations)
  ↓
InteractivePipelineRunner (same logic as chat_pipeline.py)
  ↓
Agents (Interpreter, RAG, Evaluator, Strategist, Summarizer, etc)
  ↓
LLM Models + Weaviate Vector DB
  ↓
JSON Response
  ↓
HTTP Response
  ↓
User/UI Display
```

## Files Reference

- **API Server**: `src/agentic/scripts/api_server.py`
- **API Code**: `src/agentic/api/app.py`
- **Session Manager**: `src/agentic/api/session_manager.py`
- **Python Client**: `src/agentic/api/example_client.py`
- **Full Docs**: `src/agentic/api/API.md`
- **Integration Guide**: `src/agentic/api/INTEGRATION_GUIDE.md`
- **Setup Guide**: `API_SETUP.md`

## Deployment Tips

1. Use `API_HOST=0.0.0.0` for production
2. Add authentication if public-facing
3. Use load balancer for multiple instances
4. Monitor GPU memory and response times
5. Set appropriate `MODEL_GPU_LAYERS` for your hardware
