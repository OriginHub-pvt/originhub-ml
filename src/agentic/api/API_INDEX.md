# OriginHub API Documentation Index

Complete REST API implementation for the OriginHub Agentic System. Use this to connect any frontend/UI.

## 📚 Documentation Files

### Start Here

1. **[API_IMPLEMENTATION_COMPLETE.md](./API_IMPLEMENTATION_COMPLETE.md)** ⭐

   - Overview of what was built
   - Quick start guide
   - Architecture diagram
   - Comparison with chat pipeline

2. **[API_QUICK_REFERENCE.md](./API_QUICK_REFERENCE.md)** ⭐
   - One-page quick reference
   - cURL examples
   - Python & JavaScript examples
   - Common status codes

### Detailed Documentation

3. **[src/agentic/api/API.md](./src/agentic/api/API.md)**

   - Complete endpoint reference
   - Request/response formats
   - Configuration options
   - Error handling
   - Session architecture
   - Deployment guide

4. **[src/agentic/api/INTEGRATION_GUIDE.md](./src/agentic/api/INTEGRATION_GUIDE.md)**
   - Frontend integration examples:
     - React
     - Vue.js
     - Angular
     - Vanilla JavaScript
   - Integration checklist
   - Performance tips
   - Debugging guide

### Code Examples

5. **[src/agentic/api/example_client.py](./src/agentic/api/example_client.py)**

   - Python client library
   - Interactive chat interface
   - Reusable client class

6. **[src/agentic/api/complete_example.py](./src/agentic/api/complete_example.py)**
   - Full working example
   - Error handling
   - Response type handling
   - Multiple use cases

### Technical Reference

7. **[API_SETUP.md](./API_SETUP.md)**
   - Architecture overview
   - File structure
   - Key benefits
   - Comparison with chat pipeline

---

## 🚀 Quick Start

### 1. Start the API Server

```bash
python src/agentic/scripts/api_server.py
```

### 2. Access Documentation

- **Interactive Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

### 3. Try a Quick Test

```bash
# Create session
SESSION=$(curl -s -X POST http://localhost:8000/sessions | jq -r '.session_id')

# Send message
curl -X POST http://localhost:8000/chat/$SESSION \
  -H "Content-Type: application/json" \
  -d '{"message":"AI app idea"}'

# Cleanup
curl -X DELETE http://localhost:8000/sessions/$SESSION
```

---

## 📋 API Endpoints Overview

```
Session Management
├── POST   /sessions              Create new session
├── GET    /sessions/{id}         Get session info
└── DELETE /sessions/{id}         Delete session

Chat/Analysis
└── POST   /chat/{id}             Send message, get analysis

Utility
├── GET    /health                Health check
└── GET    /info                  API information
```

---

## 🔧 Core Files

**API Implementation:**

- `src/agentic/api/app.py` - FastAPI application
- `src/agentic/api/session_manager.py` - Session management
- `src/agentic/api/models.py` - Pydantic models
- `src/agentic/scripts/api_server.py` - Server launcher

**Clients & Examples:**

- `src/agentic/api/example_client.py` - Python client
- `src/agentic/api/complete_example.py` - Full example
- `src/agentic/tests/api/test_api.py` - Unit tests

---

## 🎯 Common Tasks

### I want to...

**Start the API**
→ See [Quick Start](#-quick-start)

**Integrate with React**
→ See [INTEGRATION_GUIDE.md](./src/agentic/api/INTEGRATION_GUIDE.md#react-example)

**Integrate with Vue.js**
→ See [INTEGRATION_GUIDE.md](./src/agentic/api/INTEGRATION_GUIDE.md#vuejs-example)

**Integrate with Angular**
→ See [INTEGRATION_GUIDE.md](./src/agentic/api/INTEGRATION_GUIDE.md#angular-example)

**Use plain JavaScript**
→ See [INTEGRATION_GUIDE.md](./src/agentic/api/INTEGRATION_GUIDE.md#vanilla-javascript-example)

**Use Python**
→ See [API_QUICK_REFERENCE.md](./API_QUICK_REFERENCE.md#python-integration)

**Use cURL**
→ See [API_QUICK_REFERENCE.md](./API_QUICK_REFERENCE.md#2-send-message)

**Deploy to production**
→ See [API.md - Deployment](./src/agentic/api/API.md#deployment)

**Understand the architecture**
→ See [API_SETUP.md](./API_SETUP.md)

**See a complete example**
→ See [complete_example.py](./src/agentic/api/complete_example.py)

**Configure settings**
→ Edit `.env` (see [API.md - Configuration](./src/agentic/api/API.md#configuration))

**Debug issues**
→ See [INTEGRATION_GUIDE.md - Debugging](./src/agentic/api/INTEGRATION_GUIDE.md#debugging)

---

## 📊 API Response Format

All responses include:

```json
{
  "response": "Analysis or answer",
  "analysis_complete": false,
  "conversation_history": [
    { "role": "user", "content": "..." },
    { "role": "assistant", "content": "..." }
  ],
  "session_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

---

## 🔐 Configuration

Key `.env` settings:

```bash
API_HOST=0.0.0.0        # Server host
API_PORT=8000           # Server port
MODEL_GPU_LAYERS=30     # GPU acceleration
MODEL_PRELOAD=false     # Preload models on startup
```

Full list in `.env.example` under "API Server Configuration"

---

## 🧪 Testing

Run tests:

```bash
pytest src/agentic/tests/api/ -v
```

Run Python example:

```bash
python src/agentic/api/complete_example.py
```

Use Python client:

```bash
python src/agentic/api/example_client.py http://localhost:8000
```

---

## 📈 Performance

- **Model Loading**: ~4-5 seconds (if not preloaded)
- **Message Processing**: 2-10 seconds
- **GPU Speed**: ~40-60 tokens/sec (RTX 4070)
- **Concurrent Sessions**: 5-10 recommended (8GB VRAM)

---

## 🤝 Session Management

Each session maintains:

- Unique conversation thread
- Full message history
- Analysis progress
- User context

Sessions auto-cleanup after 24 hours of inactivity.

---

## ✅ Checklist for Integration

- [ ] Start API server: `python src/agentic/scripts/api_server.py`
- [ ] Access docs: http://localhost:8000/docs
- [ ] Test health: `curl http://localhost:8000/health`
- [ ] Create session: `curl -X POST http://localhost:8000/sessions`
- [ ] Send message: `curl -X POST /chat/{id}`
- [ ] Choose your frontend framework
- [ ] Copy example from INTEGRATION_GUIDE.md
- [ ] Implement in your frontend
- [ ] Test end-to-end
- [ ] Deploy to production

---

## 📞 Support

For issues:

1. Check [Troubleshooting](#troubleshooting) in API_IMPLEMENTATION_COMPLETE.md
2. Check [Debugging](#debugging) in INTEGRATION_GUIDE.md
3. Review health check: `curl http://localhost:8000/health`
4. Check API logs: Look at server console output

---

## 📚 Related Documentation

- **Main README**: [README.md](./README.md)
- **Agentic System**: [src/agentic/README.md](./src/agentic/README.md)
- **Chat Pipeline**: [src/agentic/scripts/chat_pipeline.py](./src/agentic/scripts/chat_pipeline.py)

---

## 🎓 Architecture Overview

```
UI/Frontend
    ↓
REST API (FastAPI)
    ↓
Session Manager
    ↓
Interactive Pipeline Runner
    ↓
Agents (Interpreter, Strategist, etc.)
    ↓
LLM Models + Weaviate Vector DB
```

Each component is independent but shares the same core logic.

---

## 🏆 Key Features

✅ RESTful API for any frontend
✅ Session-based conversations
✅ Concurrent user support
✅ Follow-up question support
✅ Interactive Swagger documentation
✅ GPU acceleration (same as chat pipeline)
✅ Error handling
✅ Example code in multiple languages
✅ Production-ready
✅ Scalable architecture

---

**Last Updated**: December 8, 2025
**API Version**: 1.0.0
**Status**: ✅ Production Ready

---

## Next Steps

1. **Start the API**

   ```bash
   python src/agentic/scripts/api_server.py
   ```

2. **Pick your frontend**

   - React → [INTEGRATION_GUIDE.md](./src/agentic/api/INTEGRATION_GUIDE.md)
   - Vue → [INTEGRATION_GUIDE.md](./src/agentic/api/INTEGRATION_GUIDE.md)
   - Angular → [INTEGRATION_GUIDE.md](./src/agentic/api/INTEGRATION_GUIDE.md)
   - JavaScript → [INTEGRATION_GUIDE.md](./src/agentic/api/INTEGRATION_GUIDE.md)

3. **Implement the integration**

   - Copy example code
   - Adapt to your UI
   - Test with interactive docs

4. **Deploy**
   - See [API.md - Deployment](./src/agentic/api/API.md#deployment)

You're all set! 🚀
