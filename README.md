# Chronic Disease Chatbot 🏥

A **4-Agent LangGraph** system for intelligent chronic disease management, powered by **Gemini 1.5 Pro/Flash**, **Tavily**, **ChromaDB**, and **SQLite**.

---

## Architecture

```
User ↔ [FastAPI] ↔ [Orchestrator (Gemini 1.5 Pro)]
                          │
            ┌─────────────┼─────────────┐
            ▼             ▼             ▼
    [Knowledge Agent] [Memory Agent] [Action Agent]
    (Tavily Search)  (ChromaDB+SQL) (Calendar+Email)
            │             │             │
            └─────────────┼─────────────┘
                          ▼
                   [Orchestrator]
                          ▼
                   [Final Response]
```

### Agent Responsibilities

| Agent | Model | Role | Tools |
|-------|-------|------|-------|
| **Orchestrator** | Gemini 1.5 Pro | Planner & Router | — |
| **Knowledge** | Gemini 1.5 Flash | Web Researcher | Tavily Search |
| **Memory** | Gemini 1.5 Flash | Librarian | ChromaDB, SQLite |
| **Action** | Gemini 1.5 Flash | Doer | Google Calendar, SendGrid |

---

## Project Structure

```
chronic_chatbot/
├── chronic_chatbot/
│   ├── __init__.py         # Package init
│   ├── config.py           # Centralised settings (reads .env)
│   ├── state.py            # Shared LangGraph AgentState TypedDict
│   ├── graph.py            # Workflow builder & router
│   ├── main.py             # FastAPI application
│   └── agents/
│       ├── __init__.py
│       ├── orchestrator.py # Main Agent – planner & router
│       ├── knowledge.py    # Knowledge Agent – Tavily search
│       ├── memory.py       # Memory Agent – ChromaDB + SQLite
│       └── action.py       # Action Agent – Calendar + Email
├── tests/
│   ├── __init__.py
│   └── test_graph.py       # Unit + integration tests
├── data/                   # Auto-created: SQLite DB + ChromaDB store
├── credentials/            # Google OAuth files (not committed)
├── requirements.txt
├── setup.py
├── .env.example            # Copy to .env and fill in your keys
└── .gitignore
```

---

## Quickstart

### 1. Clone & Setup

```bash
git clone <your-repo-url>
cd chronic_chatbot

# Create and activate the virtual environment
python3 -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# Install all dependencies
pip install -r requirements.txt

# Install the package in editable mode
pip install -e .
```

### 2. Configure Secrets

```bash
cp .env.example .env
# Edit .env and fill in your API keys:
#   GOOGLE_API_KEY    ← Gemini
#   TAVILY_API_KEY    ← Web search
#   SENDGRID_API_KEY  ← Email notifications
#   SENDER_EMAIL      ← Your sending email
```

### 3. Google Calendar (Optional)

1. Create a project at [Google Cloud Console](https://console.cloud.google.com)
2. Enable the **Google Calendar API**
3. Create **OAuth 2.0 credentials** → download as `credentials.json`
4. Place it at `credentials/google_calendar_credentials.json`

### 4. Run

```bash
source venv/bin/activate
python -m chronic_chatbot.main
# → API running at http://localhost:8000
```

### 5. Test the API

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "I feel dizzy again. Can you book an appointment with Dr. Smith?"}'
```

---

## Running Tests

```bash
# Unit tests (no API keys needed)
venv/bin/pytest tests/ -v

# Integration tests (requires real keys in .env)
venv/bin/pytest tests/ -v -m integration
```

---

## Development Roadmap

### Phase 1 – Foundation ✅
- [x] Project scaffolding & venv
- [x] All dependencies installed
- [x] State, Config, Graph modules
- [x] All 4 agent skeletons

### Phase 2 – Agent Implementation
- [ ] Orchestrator: prompt-engineer JSON routing precision
- [ ] Knowledge: tune Tavily search + summarisation prompt
- [ ] Memory: populate initial user profile (conditions, meds, doctors)
- [ ] Action: test Google Calendar OAuth flow

### Phase 3 – Integration
- [ ] End-to-end scenario tests
- [ ] Session persistence (Redis or LangGraph checkpointer)
- [ ] User profile bootstrapping endpoint

### Phase 4 – Production
- [ ] Swap SQLite → PostgreSQL
- [ ] Deploy FastAPI to Cloud Run / Railway
- [ ] Add rate limiting & auth (JWT)
- [ ] Frontend chat UI

---

## API Reference

### `POST /chat`

| Field | Type | Description |
|-------|------|-------------|
| `message` | string | User's message |
| `user_id` | string | Session identifier (default: `"default_user"`) |

**Response:**
```json
{
  "reply": "I've noted your dizziness (similar to Jan 12). I also sent an appointment request to Dr. Smith.",
  "agent_path": []
}
```

### `GET /health`
Returns `{"status": "ok"}`.
