"""
OriginHub API Integration Guide for Frontend Developers

This guide explains how to integrate the OriginHub Agentic System API
into your frontend application.
"""

# =============================================================================

# QUICK START

# =============================================================================

"""

1. Start the API server:
   python src/agentic/scripts/api_server.py
2. The API will be available at http://localhost:8000
3. Interactive docs at http://localhost:8000/docs

"""

# =============================================================================

# BASIC INTEGRATION (React Example)

# =============================================================================

REACT_EXAMPLE = """
import React, { useState } from 'react';

const OriginHubChat = () => {
const [sessionId, setSessionId] = useState(null);
const [messages, setMessages] = useState([]);
const [input, setInput] = useState('');
const [loading, setLoading] = useState(false);

// Create a new session
const startSession = async () => {
const res = await fetch('http://localhost:8000/sessions', {
method: 'POST'
});
const data = await res.json();
setSessionId(data.session_id);
setMessages([]);
};

// Send a message
const sendMessage = async (e) => {
e.preventDefault();
if (!input.trim() || !sessionId) return;

    setLoading(true);
    const userMessage = input;
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: userMessage }]);

    try {
      const res = await fetch(`http://localhost:8000/chat/${sessionId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userMessage })
      });

      const data = await res.json();
      const responseText =
        typeof data.response === 'object'
          ? JSON.stringify(data.response, null, 2)
          : data.response;

      setMessages(prev => [...prev, { role: 'assistant', content: responseText }]);
    } catch (err) {
      console.error('Error:', err);
      setMessages(prev => [...prev, {
        role: 'error',
        content: 'Failed to send message'
      }]);
    } finally {
      setLoading(false);
    }

};

return (
<div className="chat-container">
{!sessionId ? (
<button onClick={startSession}>Start Chat</button>
) : (
<>
<div className="messages">
{messages.map((msg, i) => (
<div key={i} className={`message ${msg.role}`}>
{msg.content}
</div>
))}
</div>
<form onSubmit={sendMessage}>
<input
value={input}
onChange={(e) => setInput(e.target.value)}
placeholder="Describe your business idea..."
disabled={loading}
/>
<button type="submit" disabled={loading}>
{loading ? 'Thinking...' : 'Send'}
</button>
</form>
</>
)}
</div>
);
};

export default OriginHubChat;
"""

# =============================================================================

# VUE.JS EXAMPLE

# =============================================================================

VUE_EXAMPLE = """
<template>

  <div class="chat-container">
    <button v-if="!sessionId" @click="startSession">
      Start Chat
    </button>
    <div v-else>
      <div class="messages">
        <div v-for="(msg, i) in messages" :key="i" :class="`message ${msg.role}`">
          {{ msg.content }}
        </div>
      </div>
      <form @submit.prevent="sendMessage">
        <input
          v-model="input"
          placeholder="Describe your business idea..."
          :disabled="loading"
        />
        <button type="submit" :disabled="loading">
          {{ loading ? 'Thinking...' : 'Send' }}
        </button>
      </form>
    </div>
  </div>
</template>

<script>
export default {
  data() {
    return {
      sessionId: null,
      messages: [],
      input: '',
      loading: false,
      apiUrl: 'http://localhost:8000'
    };
  },
  methods: {
    async startSession() {
      const res = await fetch(`${this.apiUrl}/sessions`, { method: 'POST' });
      const data = await res.json();
      this.sessionId = data.session_id;
    },
    async sendMessage() {
      if (!this.input.trim() || !this.sessionId) return;

      this.loading = true;
      const msg = this.input;
      this.input = '';
      this.messages.push({ role: 'user', content: msg });

      try {
        const res = await fetch(`${this.apiUrl}/chat/${this.sessionId}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: msg })
        });

        const data = await res.json();
        const content = typeof data.response === 'object'
          ? JSON.stringify(data.response, null, 2)
          : data.response;

        this.messages.push({ role: 'assistant', content });
      } catch (err) {
        console.error('Error:', err);
      } finally {
        this.loading = false;
      }
    }
  }
};
</script>

"""

# =============================================================================

# ANGULAR EXAMPLE

# =============================================================================

ANGULAR_EXAMPLE = """
import { Component } from '@angular/core';
import { HttpClient } from '@angular/common/http';

@Component({
selector: 'app-originhub-chat',
template: `    <div class="chat-container">
      <button *ngIf="!sessionId" (click)="startSession()">
        Start Chat
      </button>
      <div *ngIf="sessionId">
        <div class="messages">
          <div *ngFor="let msg of messages" [ngClass]="'message ' + msg.role">
            {{ msg.content }}
          </div>
        </div>
        <form (ngSubmit)="sendMessage()">
          <input
            [(ngModel)]="input"
            name="input"
            placeholder="Describe your business idea..."
            [disabled]="loading"
          />
          <button type="submit" [disabled]="loading">
            {{ loading ? 'Thinking...' : 'Send' }}
          </button>
        </form>
      </div>
    </div>
 `
})
export class OriginHubChatComponent {
sessionId: string | null = null;
messages: any[] = [];
input = '';
loading = false;
apiUrl = 'http://localhost:8000';

constructor(private http: HttpClient) {}

startSession() {
this.http.post(`${this.apiUrl}/sessions`, {}).subscribe((data: any) => {
this.sessionId = data.session_id;
});
}

sendMessage() {
if (!this.input.trim() || !this.sessionId) return;

    this.loading = true;
    const msg = this.input;
    this.input = '';
    this.messages.push({ role: 'user', content: msg });

    this.http.post(`${this.apiUrl}/chat/${this.sessionId}`, { message: msg })
      .subscribe(
        (data: any) => {
          const content = typeof data.response === 'object'
            ? JSON.stringify(data.response, null, 2)
            : data.response;
          this.messages.push({ role: 'assistant', content });
        },
        (err) => console.error('Error:', err),
        () => { this.loading = false; }
      );

}
}
"""

# =============================================================================

# VANILLA JAVASCRIPT EXAMPLE

# =============================================================================

VANILLA_JS_EXAMPLE = """
class OriginHubChat {
constructor(containerId, apiUrl = 'http://localhost:8000') {
this.container = document.getElementById(containerId);
this.apiUrl = apiUrl;
this.sessionId = null;
this.messages = [];
this.init();
}

async init() {
this.container.innerHTML = '<button id="startBtn">Start Chat</button>';
document.getElementById('startBtn').addEventListener('click',
() => this.startSession()
);
}

async startSession() {
const res = await fetch(`${this.apiUrl}/sessions`, { method: 'POST' });
const data = await res.json();
this.sessionId = data.session_id;
this.renderChat();
}

renderChat() {
this.container.innerHTML = `      <div class="messages" id="messages"></div>
      <form onsubmit="return false;" id="chatForm">
        <input id="input" placeholder="Describe your idea..."/>
        <button type="submit">Send</button>
      </form>
   `;

    document.getElementById('chatForm').addEventListener('submit',
      () => this.sendMessage()
    );

}

async sendMessage() {
const input = document.getElementById('input');
const msg = input.value.trim();
if (!msg) return;

    input.value = '';
    this.addMessage('user', msg);

    try {
      const res = await fetch(`${this.apiUrl}/chat/${this.sessionId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: msg })
      });

      const data = await res.json();
      const content = typeof data.response === 'object'
        ? JSON.stringify(data.response, null, 2)
        : data.response;

      this.addMessage('assistant', content);
    } catch (err) {
      console.error('Error:', err);
    }

}

addMessage(role, content) {
const messagesDiv = document.getElementById('messages');
const msgEl = document.createElement('div');
msgEl.className = `message ${role}`;
msgEl.textContent = content;
messagesDiv.appendChild(msgEl);
messagesDiv.scrollTop = messagesDiv.scrollHeight;
}
}

// Initialize
const chat = new OriginHubChat('chatContainer');
"""

# =============================================================================

# KEY INTEGRATION POINTS

# =============================================================================

INTEGRATION_CHECKLIST = """
✓ Session Management

- Create session before first message
- Store session_id for conversation continuity
- Delete session on logout/exit

✓ Message Handling

- Send text messages to /chat/{session_id}
- Handle analysis_complete flag
- Display JSON responses prettified

✓ Error Handling

- Check HTTP status codes
- Display user-friendly error messages
- Retry logic for transient failures

✓ UI Patterns

- Show loading state while processing
- Display conversation history
- Support follow-up questions
- Pretty-print JSON analysis results

✓ Performance

- Session reuse across messages
- Lazy-load conversation history
- Debounce rapid message sends
- Cache session data locally

✓ Accessibility

- Keyboard navigation
- Screen reader support
- Mobile-responsive design
- ARIA labels for dynamic content
  """

# =============================================================================

# CORS CONFIGURATION

# =============================================================================

CORS_INFO = """
The API has CORS enabled for all origins by default.
This is suitable for development.

For production, configure CORS in .env:

- Backend: src/agentic/api/app.py
- Modify: allow_origins = ["your-frontend-domain.com"]

Example:
allow_origins = ["https://app.originhub.com", "https://www.originhub.com"]
"""

# =============================================================================

# PERFORMANCE TIPS

# =============================================================================

PERFORMANCE_TIPS = """

1. Session Caching

   - Store sessionId in localStorage
   - Restore session on page reload

2. Message Queuing

   - Queue messages if API is busy
   - Process queue sequentially

3. Streaming (if implemented)

   - Use Server-Sent Events for real-time updates
   - Stream long responses token-by-token

4. Debouncing

   - Debounce typing indicators
   - Debounce auto-save of draft messages

5. Response Caching
   - Cache analysis results client-side
   - Reuse results for similar queries
     """

# =============================================================================

# DEBUGGING

# =============================================================================

DEBUG_TIPS = """

1. Check API Health
   GET http://localhost:8000/health

2. View API Documentation
   http://localhost:8000/docs (Swagger UI)
   http://localhost:8000/redoc (ReDoc)

3. Monitor Network

   - Open browser DevTools
   - Check Network tab for API calls
   - View request/response bodies

4. Check Console

   - Log session_id and message flow
   - Monitor API response times

5. Test with cURL
   # Create session
   curl -X POST http://localhost:8000/sessions
   # Send message
   curl -X POST http://localhost:8000/chat/{sessionId} \\
   -H "Content-Type: application/json" \\
   -d '{"message":"Your message"}'
   """

if **name** == "**main**":
print("OriginHub API Integration Guide")
print("\nReact Example:")
print(REACT_EXAMPLE)
print("\nVue.js Example:")
print(VUE_EXAMPLE)
print("\nAngular Example:")
print(ANGULAR_EXAMPLE)
print("\nVanilla JavaScript Example:")
print(VANILLA_JS_EXAMPLE)
print("\nIntegration Checklist:")
print(INTEGRATION_CHECKLIST)
print("\nPerformance Tips:")
print(PERFORMANCE_TIPS)
