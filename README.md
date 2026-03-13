# 🏥 HealthChat — Backend

> **4-Agent LangGraph system with MCP tool servers for chronic disease management**  
> Powered by Gemini 2.5 Flash Lite · MCP (Model Context Protocol) · Tavily · ChromaDB · SQLite · Google Calendar

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [MCP Server Design](#mcp-server-design)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Environment Variables](#environment-variables)
- [Running the Server](#running-the-server)
- [API Reference](#api-reference)
- [Agent System](#agent-system)
- [Database Schema](#database-schema)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)

---

## Overview

The backend is a **FastAPI** application backed by a **LangGraph** multi-agent workflow. When a user sends a message it flows through 4 specialist AI agents. The key architectural feature is that each sub-agent is an **MCP client** — it spawns a dedicated **MCP server subprocess** that exposes tools, and lets Gemini autonomously decide which tool(s) to call. This removes all manual intent-parsing and routing from the agent code itself.

All agents use **Gemini 2.5 Flash Lite** — fast and cost-efficient.

---

## Architecture

```
User message
     │
     ▼
┌─────────────────────────────────────────────────────┐
│                  Orchestrator Agent                 │
│  Analyses intent → routes to sub-agents via JSON   │
│  Synthesises final empathetic response              │
└──────┬───────────────┬────────────────┬─────────────┘
       │               │                │
       ▼               ▼                ▼
 ┌───────────┐  ┌────────────┐  ┌────────────────┐
 │ Knowledge │  │   Memory   │  │     Action     │
 │   Agent   │  │   Agent    │  │     Agent      │
 │ (client)  │  │  (client)  │  │   (client)     │
 └─────┬─────┘  └─────┬──────┘  └──────┬─────────┘
       │               │                │
   stdio MCP       stdio MCP        stdio MCP
       │               │                │
       ▼               ▼                ▼
 ┌───────────┐  ┌────────────┐  ┌────────────────┐
 │knowledge_ │  │ memory_    │  │ action_        │
 │server.py  │  │ server.py  │  │ server.py      │
 │           │  │            │  │                │
 │ Tavily    │  │ ChromaDB   │  │ Google Cal.    │
 │ search    │  │ + SQLite   │  │ + Gmail SMTP   │
 └───────────┘  └────────────┘  └────────────────┘
```

**Data flow per message:**

1. User message → **Orchestrator** detects intent, routes to sub-agent
2. Sub-agent calls `await client.get_tools()` — spawns MCP subprocess, loads exposed tools
3. Agent binds tools to Gemini → LLM picks correct tool(s) and calls them
4. Tool results bubble back through `memory_context` / `knowledge_context` / `action_result`
5. Orchestrator sees results, routes next or generates final response

---

## MCP Server Design

Each MCP server is a standalone Python script run as a **stdio subprocess**. They are completely isolated from the main process — no shared memory, no import-time side effects in FastAPI.

| MCP Server | Tools Exposed |
|---|---|
| `mcp_server/memory_server.py` | `log_symptom_dual`, `search_symptoms`, `query_medications`, `query_doctors`, `query_appointments`, `add_medication`, `add_doctor`, `add_appointment_record` |
| `mcp_server/knowledge_server.py` | `tavily_search` |
| `mcp_server/action_server.py` | `create_calendar_event`, `update_calendar_event`, `delete_calendar_event`, `send_email_to_doctor` |

**Why MCP?**

- ✅ No manual intent parsing — Gemini reads the tool signatures and picks the right one
- ✅ ChromaDB runs in an isolated subprocess — avoids Rust binding crashes in the main process
- ✅ Each server is independently testable with `python mcp_server/memory_server.py`
- ✅ Tools can be swapped or extended without touching agent code

---

## Tech Stack

| Layer | Technology |
|---|---|
| API Framework | FastAPI + Uvicorn (async) |
| AI Orchestration | LangGraph (async `ainvoke`) |
| LLM | Google Gemini 2.5 Flash Lite |
| Tool Protocol | MCP (Model Context Protocol) via `mcp` + `langchain-mcp-adapters` |
| Web Search | Tavily Search API (inside knowledge MCP server) |
| Vector Memory | ChromaDB 1.x — runs inside memory MCP subprocess |
| Structured DB | SQLite stdlib — inside memory MCP subprocess |
| Calendar | Google Calendar API v3 (OAuth2) — inside action MCP server |
| Email | Gmail SMTP + App Password (stdlib smtplib) — inside action MCP server |
| Config | python-dotenv + Pydantic |

---

## Project Structure

```
backend/
├── chronic_chatbot/
│   ├── agents/                       # LangGraph node functions (MCP clients)
│   │   ├── __init__.py               # Re-exports all 4 nodes
│   │   ├── orchestrator.py           # Planner — routes & synthesises (sync LLM)
│   │   ├── knowledge.py              # async: spawns knowledge_server, calls tavily_search
│   │   ├── memory.py                 # async: spawns memory_server, picks DB tools
│   │   └── action.py                 # async: spawns action_server, picks calendar/email tools
│   ├── mcp_server/                   # MCP tool servers (run as stdio subprocesses)
│   │   ├── __init__.py
│   │   ├── memory_server.py          # ChromaDB + SQLite tools
│   │   ├── knowledge_server.py       # Tavily search tool
│   │   └── action_server.py          # Google Calendar + Gmail tools
│   ├── config.py                     # Centralised settings from .env
│   ├── graph.py                      # LangGraph workflow (async)
│   ├── main.py                       # FastAPI app + REST endpoints
│   ├── state.py                      # AgentState TypedDict
│   └── utils.py                      # safe_content, safe_llm_invoke, strip_agent_prefix
├── credentials/
│   ├── google_calendar_credentials.json   # OAuth2 client credentials
│   └── google_calendar_token.json         # Cached OAuth token (auto-generated)
├── data/
│   ├── chronic_chatbot.db            # SQLite database
│   └── chroma_store/                 # ChromaDB vector store (Rust-backed)
├── tests/
│   ├── validate_apis.py              # API key + model connectivity check
│   └── test_graph.py                 # LangGraph unit test
├── .env                              # Your secrets (not in git)
├── .env.example                      # Template — copy to .env
├── requirements.txt                  # All Python dependencies
└── setup.py                          # Package definition
```

---

## Setup

### Prerequisites

- Python **3.10+** (3.11 recommended)
- `pip`

### 1. Enter the backend directory

```bash
cd /path/to/chronic_chatbot/backend
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate       # Linux/macOS
# venv\Scripts\activate        # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
pip install -e .               # Install the package in editable mode
```

> **Note on ChromaDB:** ChromaDB 1.x uses Rust bindings and runs as an MCP subprocess,
> avoiding any Rust/Python abi conflicts in the main FastAPI process.

### 4. Configure environment variables

```bash
cp .env.example .env
# Fill in your API keys (see section below)
```

### 5. Set up Google Calendar (optional)

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a project → Enable **Google Calendar API**
3. Create **OAuth 2.0 credentials** → Desktop app
4. Download `credentials.json` → save as `credentials/google_calendar_credentials.json`
5. On first appointment booking via chat or UI, a browser window opens for OAuth consent
6. Token is saved to `credentials/google_calendar_token.json` for future runs

If you skip this, the action agent returns a soft error message — the rest of the system works normally.

---

## Environment Variables

Copy `.env.example` to `.env` and fill in the values:

```env
# ── Required ──────────────────────────────────────────────────
GOOGLE_API_KEY=your_gemini_api_key_here
# Get from: https://aistudio.google.com/app/apikey

TAVILY_API_KEY=your_tavily_api_key_here
# Get from: https://app.tavily.com  (free: 1000 searches/month)

# ── Optional: Email via Gmail ─────────────────────────────────
GMAIL_SENDER_EMAIL=your_gmail@gmail.com
GMAIL_APP_PASSWORD=your_16_char_app_password
# Get App Password: myaccount.google.com/apppasswords
# (Enable 2-Step Verification first, then create an App Password)

# ── Optional: Google Calendar ─────────────────────────────────
GOOGLE_CALENDAR_CREDENTIALS_PATH=credentials/google_calendar_credentials.json
GOOGLE_CALENDAR_TOKEN_PATH=credentials/google_calendar_token.json

# ── Database (defaults work out of the box) ───────────────────
SQLITE_DB_PATH=data/chronic_chatbot.db
CHROMA_PERSIST_DIR=data/chroma_store
LOG_LEVEL=INFO
```

### Getting API Keys

| Key | Where | Cost |
|---|---|---|
| `GOOGLE_API_KEY` | [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) | Free tier available |
| `TAVILY_API_KEY` | [app.tavily.com](https://app.tavily.com) | Free: 1000 searches/month |
| Gmail App Password | Google Account → Security → App Passwords | Free |

---

## Running the Server

### Development (auto-reload)

```bash
source venv/bin/activate
uvicorn chronic_chatbot.main:app --host 0.0.0.0 --port 8000 --reload
```

### Production

```bash
uvicorn chronic_chatbot.main:app --host 0.0.0.0 --port 8000 --workers 2
```

### Verify it's running

```bash
curl http://localhost:8000/health
# → {"status": "ok", "service": "chronic_chatbot"}
```

**Swagger UI:** http://localhost:8000/docs

---

## API Reference

### `GET /health`
Health probe.
```json
{"status": "ok", "service": "chronic_chatbot"}
```

---

### `POST /chat`
Send a message to the AI health assistant.

**Request:**
```json
{
  "message": "I have been having headaches since this morning",
  "user_id": "user_123"
}
```

**Response:**
```json
{
  "reply": "I've logged your headache in your health records. Here's what you should know..."
}
```

> **Note:** Response time is 10–40 seconds — 2–4 async agent calls run sequentially, each spawning an MCP subprocess.

**What happens automatically in chat:**

| You say | Auto-logged |
|---|---|
| `"I have a headache"` | Symptom → ChromaDB + SQLite `/memory` page |
| `"I started taking Metformin 500mg"` | Medication → SQLite `/medications` page |
| `"Book appointment with Dr. X on Friday"` | Google Calendar event created |
| `"What medications am I on?"` | Reads from SQLite medications table |
| `"Have I had this before?"` | Semantic search over symptom history |

---

### Medications

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/medications` | List all medications |
| `POST` | `/api/medications` | Add a medication |
| `DELETE` | `/api/medications/{id}` | Remove a medication |

**POST body:**
```json
{
  "name": "Metformin",
  "dosage": "500mg",
  "frequency": "Twice daily",
  "startDate": "2024-01-15",
  "notes": "Take with meals"
}
```

---

### Appointments

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/appointments` | List all appointments |
| `POST` | `/api/appointments` | Book appointment (+ Google Calendar) |
| `PATCH` | `/api/appointments/{id}` | Reschedule (+ updates Calendar event) |
| `DELETE` | `/api/appointments/{id}` | Cancel (+ removes Calendar event) |

**POST body:**
```json
{
  "doctorName": "Dr. Sarah Johnson",
  "specialty": "Endocrinology",
  "dateTime": "2026-03-20T10:00:00",
  "location": "City Medical Center",
  "notes": "Quarterly diabetes review",
  "status": "scheduled"
}
```

**Response includes `calendarId` and `calendarMessage`:**
```json
{
  "id": 1,
  "calendarId": "abc123xyz",
  "calendarMessage": "✅ Calendar event created. View: https://..."
}
```

---

### Doctors

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/doctors` | List all doctors |
| `POST` | `/api/doctors` | Add a doctor |

---

### Symptom History

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/symptoms` | List logged symptoms |
| `POST` | `/api/symptoms` | Log a symptom manually |
| `DELETE` | `/api/symptoms/{id}` | Delete a symptom entry |

> Chat-reported symptoms are **automatically** logged to both ChromaDB (semantic) and SQLite (frontend display).

---

### Profile

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/profile` | Get all profile key-value pairs |
| `PUT` | `/api/profile` | Update profile |

---

## Agent System

### Orchestrator (`agents/orchestrator.py`)

The only **synchronous** agent. Uses explicit JSON routing with trigger-word rules:

| User intent | Route | Instruction sent |
|---|---|---|
| `"I have a headache"` | memory → knowledge → final | `"WRITE symptom: headache"` |
| `"What medications am I on?"` | memory → final | `"READ medications"` |
| `"What causes dizziness?"` | knowledge → final | `"Causes of dizziness in chronic patient"` |
| `"Book with Dr. X on Friday"` | action → final | `"Book appointment with Dr. X on Friday"` |
| General chat | final | (direct reply) |

### Memory Agent (`agents/memory.py`) — MCP client

**Async.** Calls `await client.get_tools()` to load tools from `memory_server.py`, binds them to Gemini. LLM selects and calls the right tool based on the orchestrator's instruction.

**Tools available:**
- `log_symptom_dual(symptom_text, severity, notes)` → ChromaDB + SQLite
- `search_symptoms(query, top_k)` → ChromaDB + SQLite merged
- `query_medications()` → SQLite
- `query_doctors(name_filter)` → SQLite
- `query_appointments()` → SQLite
- `add_medication(data)` → SQLite
- `add_doctor(data)` → SQLite
- `add_appointment_record(data)` → SQLite

### Knowledge Agent (`agents/knowledge.py`) — MCP client

**Async.** Calls `tavily_search` tool from `knowledge_server.py`, then summarises results with the LLM. Falls back to LLM-only response if Tavily is unavailable.

### Action Agent (`agents/action.py`) — MCP client

**Async.** LLM selects from action tools in `action_server.py`.

**Tools available:**
- `create_calendar_event(doctor_name, date_str, reason, location)` → returns `(message, event_id)`
- `update_calendar_event(event_id, doctor_name, date_str, reason, location)` → reschedule
- `delete_calendar_event(event_id)` → idempotent delete
- `send_email_to_doctor(doctor_email, doctor_name, subject, body)` → Gmail SMTP

---

## Database Schema

### SQLite (`data/chronic_chatbot.db`)

```sql
doctors        (id, name, specialty, email, phone, notes)
medications    (id, name, dose, frequency, start_date, end_date, notes)
appointments   (id, doctor_id, date_time, reason, status, calendar_id)
user_profile   (key TEXT PRIMARY KEY, value TEXT)
symptoms_log   (id, symptom, severity, notes, logged_at)
```

> Tables are auto-created on first run by `memory_server.py:init_db()`.

### ChromaDB (`data/chroma_store/`)

Collection: `symptom_history`
- Documents: symptom description text
- Metadata: `{"date": "ISO8601", "severity": int}`
- Embedding: default all-MiniLM-L6-v2 (ONNX)
- Backend: Rust (ChromaDB 1.x)

> ChromaDB runs entirely inside the `memory_server.py` MCP subprocess — no Rust abi conflicts with the FastAPI process.

---

## Testing

### Validate API keys and connections

```bash
source venv/bin/activate
python tests/validate_apis.py
```

Checks: Gemini API, Tavily, Gmail SMTP, Google Calendar.

### Test the chat endpoint

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "I have been feeling dizzy today", "user_id": "test"}'
```

### Test CRUD endpoints

```bash
# Add a medication
curl -X POST http://localhost:8000/api/medications \
  -H "Content-Type: application/json" \
  -d '{"name":"Metformin","dosage":"500mg","frequency":"Twice daily","startDate":"2024-01-01"}'

# List symptoms
curl http://localhost:8000/api/symptoms

# Log a symptom
curl -X POST http://localhost:8000/api/symptoms \
  -H "Content-Type: application/json" \
  -d '{"symptom":"Headache","severity":3,"notes":"After lunch"}'
```

### Test MCP servers directly

```bash
# Each MCP server runs as a standalone FastMCP app
source venv/bin/activate
python chronic_chatbot/mcp_server/memory_server.py    # runs in stdio mode
python chronic_chatbot/mcp_server/knowledge_server.py
python chronic_chatbot/mcp_server/action_server.py
```

---

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `GOOGLE_API_KEY not found` | `.env` not loaded | Run from `backend/` directory with venv active |
| `Model not found / 404` | Wrong model string | Check `.env` — use `gemini-2.0-flash` as fallback |
| ChromaDB Rust panic | Old issue — fixed by MCP isolation | Reinstall: `pip install chromadb>=1.0.0` |
| `No tools loaded from memory_server` | subprocess failed to start | Run `python chronic_chatbot/mcp_server/memory_server.py` to see error |
| `Tavily search failed` | Network or quota | Check `TAVILY_API_KEY`; free tier = 1000/month |
| `Calendar credentials not found` | Missing OAuth file | Add `credentials/google_calendar_credentials.json` |
| `Gmail auth failed` | Wrong password type | Use **App Password** (16 chars), not your Gmail login |
| `Port 8000 in use` | Old process running | `kill $(lsof -t -i:8000)` |
| Slow responses (30–40s) | MCP subprocess startup per request | Expected — each agent spawns a subprocess; use persistent MCP servers for production |
