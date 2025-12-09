# OriginHub Agentic System API

REST API for the OriginHub multi-agent reasoning system. Enables UI connections for conversational business idea analysis.

## Quick Start

### 1. Start the API Server

```bash
python src/agentic/scripts/api_server.py
```

The server will start on `http://localhost:8000` (configurable via `.env`).

### 2. Access API Documentation

- **Interactive Docs (Swagger UI)**: http://localhost:8000/docs
- **Alternative Docs (ReDoc)**: http://localhost:8000/redoc

### 3. Connect Your UI

See the example client at `src/agentic/api/example_client.py` for implementation details.

---

## API Endpoints

### Health Check

```
GET /health
```

Check if the API is running and models are loaded.

**Response:**

```json
{
  "status": "healthy",
  "models_loaded": true,
  "weaviate_connected": true,
  "timestamp": "2025-12-08T10:30:00.123456"
}
```

### Session Management

#### Create Session

```
POST /sessions
```

Create a new conversation session.

**Response:**

```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "created_at": "2025-12-08T10:30:00.123456",
  "message": "Session created successfully"
}
```

#### Get Session Info

```
GET /sessions/{session_id}
```

Get metadata about a session.

**Response:**

```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "created_at": "2025-12-08T10:30:00.123456",
  "last_activity": "2025-12-08T10:35:00.123456",
  "message_count": 5
}
```

#### Delete Session

```
DELETE /sessions/{session_id}
```

End a conversation session and free up resources.

**Response:**

```json
{
  "message": "Session deleted successfully"
}
```

### Chat / Message Processing

#### Send Message

```
POST /chat/{session_id}
```

Send a user message and get the AI analysis response.

**Request:**

```json
{
  "message": "An AI that helps manage personal productivity using sleep brainwave data"
}
```

**Response:**

```json
{
  "response": {
    "final_summary": "...",
    "next_steps": ["...", "..."],
    ...
  },
  "analysis_complete": false,
  "conversation_history": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "session_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

### Info

```
GET /info
```

Get API information and available endpoints.

---

## Configuration

Set these in your `.env` file:

```bash
# API Server
API_HOST=0.0.0.0
API_PORT=8000

# Model Configuration
MODEL_7B_PATH=models/qwen-7b/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf
MODEL_1B_PATH=models/qwen-1.5b/model.gguf
MODEL_GPU_LAYERS=30
MODEL_CONTEXT_SIZE=4096
MODEL_N_THREADS=4
MODEL_PRELOAD=false

# Weaviate
WEAVIATE_HOST=localhost
WEAVIATE_PORT=8081
WEAVIATE_GRPC_PORT=50051
WEAVIATE_COLLECTION=ArticleSummary

# RAG Settings
RAG_NEW_THRESHOLD=0.8
```

---

## Usage Examples

### Python Client

```python
from src.agentic.api.example_client import OriginHubClient

# Initialize client
client = OriginHubClient("http://localhost:8000")

# Create session
session_id = client.create_session()

# Send message
result = client.send_message("AI app for personalized fitness coaching")

# Process response
print(result["response"])
print(f"Analysis complete: {result['analysis_complete']}")

# Ask follow-up questions
if result["analysis_complete"]:
    followup = client.send_message("What's the market size?")
    print(followup["response"])

# Cleanup
client.delete_session()
```

### JavaScript/Frontend

```javascript
// Create session
const sessionRes = await fetch("http://localhost:8000/sessions", {
  method: "POST",
});
const { session_id } = await sessionRes.json();

// Send message
const messageRes = await fetch(`http://localhost:8000/chat/${session_id}`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ message: "Your business idea here" }),
});
const result = await messageRes.json();

// Display response
console.log(result.response);
console.log(result.analysis_complete);

// Ask follow-up question if analysis is complete
if (result.analysis_complete) {
  const followupRes = await fetch(`http://localhost:8000/chat/${session_id}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message: "Tell me about the market research" }),
  });
  const followupResult = await followupRes.json();
  console.log(followupResult.response);
}
```

### cURL

```bash
# Create session
SESSION_ID=$(curl -s -X POST http://localhost:8000/sessions | jq -r '.session_id')

# Send message
curl -X POST http://localhost:8000/chat/$SESSION_ID \
  -H "Content-Type: application/json" \
  -d '{"message":"AI productivity app"}'

# Get session info
curl http://localhost:8000/sessions/$SESSION_ID

# Delete session
curl -X DELETE http://localhost:8000/sessions/$SESSION_ID
```

---

## Session Architecture

Sessions provide isolated conversation contexts:

- **One session = One conversation thread**
- **Multiple concurrent sessions supported**
- **Each session maintains full conversation history**
- **Analysis progress tracked per session**
- **Automatic cleanup of stale sessions (24h default)**

### Session Lifecycle

```
1. Create Session (POST /sessions)
   ↓
2. Send Messages (POST /chat/{session_id})
   ├─ Interpreter processes input
   ├─ RAG searches for similar ideas
   ├─ Evaluator routes to appropriate analysis
   └─ Summarizer produces final output
   ↓
3. Analysis Complete → Enable follow-up questions
   ↓
4. Send Follow-up Messages (POST /chat/{session_id})
   ├─ Context from full history used
   └─ Targeted response generation
   ↓
5. Delete Session (DELETE /sessions/{session_id})
```

---

## Error Handling

API returns standard HTTP status codes:

- **200 OK** - Request successful
- **400 Bad Request** - Invalid message or parameters
- **404 Not Found** - Session doesn't exist
- **503 Service Unavailable** - Models still loading
- **500 Internal Server Error** - Processing error

Example error response:

```json
{
  "detail": "Session not found"
}
```

---

## Performance Considerations

- **Model Loading**: ~4-5 seconds on first startup
- **Message Processing**: 2-10 seconds depending on analysis complexity
- **GPU Acceleration**: ~40-60 tokens/sec with RTX 4070
- **Concurrent Sessions**: Recommend max 5-10 with 8GB VRAM

---

## Deployment

### Docker (Optional)

See main README for Docker setup instructions.

### Production Deployment

For production, consider:

1. **Load Balancing**: Use Nginx/HAProxy for multiple API instances
2. **Monitoring**: Track API health, response times, error rates
3. **Logging**: Centralize logs from all instances
4. **Database**: Store sessions in persistent storage for recovery
5. **Auth**: Add authentication (API keys, JWT) if public-facing
6. **Rate Limiting**: Implement per-session or per-IP rate limits

---

## Troubleshooting

### Models won't load

- Check `MODEL_7B_PATH` and `MODEL_1B_PATH` point to valid files
- Verify CUDA is installed and detected
- Check available GPU memory

### Weaviate connection fails

- Ensure Weaviate is running: `docker-compose up weaviate`
- Verify `WEAVIATE_HOST` and `WEAVIATE_PORT` settings
- Check Weaviate logs

### Slow responses

- Check GPU utilization: `nvidia-smi`
- Reduce `MODEL_CONTEXT_SIZE` if needed
- Monitor system memory and disk space

### Session not found

- Sessions expire after 24 hours of inactivity
- Check session_id is correct
- Create a new session if expired
