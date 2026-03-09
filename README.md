# 🏥 HealthChat — Backend

> **4-Agent LangGraph system for chronic disease management**  
> Powered by Gemini 2.5 Flash Lite · Tavily · ChromaDB · SQLite · Google Calendar

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
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

The backend is a **FastAPI** application that exposes a chat endpoint backed by a multi-agent **LangGraph** workflow. When a user sends a message, it flows through 4 specialist AI agents that decide whether to answer from web knowledge, read/write the patient's health database, or take real-world actions like booking calendar appointments and sending emails.

All agents use **Gemini 2.5 Flash Lite** — the most cost-efficient model in the 2.5 series.

---

## Architecture

```
User message
     │
     ▼
┌─────────────────────────────────────────────┐
│              Orchestrator Agent             │
│  Analyses intent, routes to sub-agents,    │
│  synthesises final empathetic response     │
└──────┬──────────┬──────────┬───────────────┘
       │          │          │
       ▼          ▼          ▼
 ┌──────────┐ ┌─────────┐ ┌──────────────┐
 │Knowledge │ │ Memory  │ │   Action     │
 │  Agent   │ │  Agent  │ │   Agent      │
 │          │ │         │ │              │
 │ Tavily   │ │ChromaDB │ │ Google Cal.  │
 │ web search│ │SQLite   │ │ Gmail SMTP   │
 └──────────┘ └─────────┘ └──────────────┘
       │          │          │
       └──────────┴──────────┘
                  │
                  ▼
           Final Response
```

**Data Flow:**

1. User message → **Orchestrator** classifies intent
2. Orchestrator routes to 0–3 sub-agents sequentially
3. Results accumulate in shared `AgentState`
4. Orchestrator synthesises a final response
5. REST endpoint returns reply to frontend

---

## Tech Stack

| Layer | Technology |
|---|---|
| API Framework | FastAPI + Uvicorn |
| AI Orchestration | LangGraph |
| LLM | Google Gemini 2.5 Flash Lite |
| Web Search | Tavily Search API |
| Vector Memory | ChromaDB (ONNX embeddings, no PyTorch) |
| Structured DB | SQLite (stdlib — zero dependencies) |
| Calendar | Google Calendar API v3 (OAuth2) |
| Email | Gmail SMTP + App Password (stdlib smtplib) |
| Config | python-dotenv + Pydantic |

---

## Project Structure

```
backend/
├── chronic_chatbot/
│   ├── agents/
│   │   ├── __init__.py           # Re-exports all 4 nodes
│   │   ├── orchestrator.py       # Planner — routes & synthesises
│   │   ├── knowledge.py          # Tavily web search + LLM summary
│   │   ├── memory.py             # ChromaDB + SQLite read/write
│   │   └── action.py             # Google Calendar + Gmail
│   ├── config.py                 # Centralised settings from .env
│   ├── graph.py                  # LangGraph workflow builder
│   ├── main.py                   # FastAPI app + REST endpoints
│   ├── state.py                  # AgentState TypedDict
│   └── utils.py                  # safe_content, safe_llm_invoke, etc.
├── credentials/
│   ├── google_calendar_credentials.json   # OAuth2 credentials
│   └── google_calendar_token.json         # Cached OAuth token
├── data/
│   ├── chronic_chatbot.db        # SQLite database
│   └── chroma_store/             # ChromaDB vector store
├── tests/
│   ├── validate_apis.py          # API key + model validation script
│   └── test_graph.py             # LangGraph unit tests
├── .env                          # Your secrets (not in git)
├── .env.example                  # Template — copy to .env
├── requirements.txt              # Python dependencies
└── setup.py                      # Package definition
```

---

## Setup

### Prerequisites

- Python 3.11+
- `pip`

### 1. Clone and enter the directory

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

### 4. Configure environment variables

```bash
cp .env.example .env
# Edit .env and fill in your API keys (see section below)
```

### 5. Set up Google Calendar (optional)

If you want calendar booking to work:

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a project → Enable **Google Calendar API**
3. Create **OAuth 2.0 credentials** → Desktop app
4. Download `credentials.json` → save as `credentials/google_calendar_credentials.json`
5. On first run, a browser window opens for one-time OAuth consent
6. Token is saved to `credentials/google_calendar_token.json` for future use

If you skip this, the calendar features simply return a friendly error message — the rest of the system works normally.

---

## Environment Variables

Copy `.env.example` to `.env` and fill in the values:

```env
# ── Required ──────────────────────────────────────────────────
GOOGLE_API_KEY=your_gemini_api_key_here
# Get from: https://aistudio.google.com/app/apikey

TAVILY_API_KEY=your_tavily_api_key_here
# Get from: https://app.tavily.com

# ── Optional: Email Notifications ────────────────────────────
GMAIL_SENDER_EMAIL=your_gmail@gmail.com
GMAIL_APP_PASSWORD=your_16_char_app_password
# Get App Password: myaccount.google.com/apppasswords
# (Enable 2-Step Verification first)

# ── Optional: Google Calendar ─────────────────────────────────
GOOGLE_CALENDAR_CREDENTIALS_PATH=credentials/google_calendar_credentials.json
GOOGLE_CALENDAR_TOKEN_PATH=credentials/google_calendar_token.json

# ── Database (defaults work out of the box) ───────────────────
SQLITE_DB_PATH=data/chronic_chatbot.db
CHROMA_PERSIST_DIR=data/chroma_store
LOG_LEVEL=INFO
```

### Getting API Keys

| Key | Where to get |
|---|---|
| `GOOGLE_API_KEY` | [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) — free tier available |
| `TAVILY_API_KEY` | [app.tavily.com](https://app.tavily.com) — free tier: 1000 searches/month |
| Gmail App Password | Google Account → Security → 2-Step Verification → App Passwords |

---

## Running the Server

### Development (with auto-reload)

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

**Swagger UI (interactive docs):** http://localhost:8000/docs

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
  "reply": "I've logged your headache in your health records. Here's what you should know...",
  "agent_path": []
}
```

> **Note:** Response time is 10–30 seconds as 2–3 agents run sequentially.

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

> Doctors are also auto-created when appointments are booked.

---

### Symptom History

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/symptoms` | List logged symptoms |
| `POST` | `/api/symptoms` | Log a symptom |
| `DELETE` | `/api/symptoms/{id}` | Delete a symptom entry |

> Symptoms mentioned in **chat** are automatically logged to both ChromaDB  
> (for semantic search by the AI) and SQLite (for the frontend `/memory` page).

---

### Profile

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/profile` | Get profile key-value pairs |
| `PUT` | `/api/profile` | Update profile |

---

## Agent System

### Orchestrator (`agents/orchestrator.py`)

The decision-maker. Receives the user message, classifies intent, and routes to sub-agents one at a time:

| User says | Route |
|---|---|
| `"I have a headache"` | **memory** (WRITE) → **knowledge** → final |
| `"What causes dizziness?"` | **knowledge** → final |
| `"Book appointment with Dr. X"` | **action** → final |
| `"What medications am I on?"` | **memory** (READ) → final |
| `"Have I logged any symptoms?"` | **memory** (READ) → final |

### Knowledge Agent (`agents/knowledge.py`)

- Queries Tavily for medically-relevant web results
- Summarises into 3–5 bullet points via LLM
- Falls back to LLM-only answer if Tavily fails (network down, quota hit)

### Memory Agent (`agents/memory.py`)

- **WRITE symptom** → ChromaDB (semantic) + SQLite `symptoms_log`
- **WRITE medication** → SQLite `medications`
- **READ symptoms** → ChromaDB search + SQLite recent entries (merged)
- **READ medications** → SQLite query
- Instruction format: `"WRITE symptom: headache and nausea"`, `"READ medications"`

### Action Agent (`agents/action.py`)

- Parses natural-language booking instruction with LLM
- Creates/updates/deletes Google Calendar events
- Sends appointment request emails via Gmail SMTP
- All calendar operations return `(message, event_id)` tuples

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

### ChromaDB (`data/chroma_store/`)

Collection: `symptom_history`
- Documents: symptom description text
- Metadata: `{"date": "ISO8601", "severity": int}`
- Embedding: all-MiniLM-L6-v2 via ONNX (no PyTorch required)

---

## Testing

### Validate API keys and model

```bash
source venv/bin/activate
python tests/validate_apis.py
```

This checks: Gemini API, Tavily API, Gmail SMTP, Google Calendar.

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
  -d '{"name": "Metformin", "dosage": "500mg", "frequency": "Twice daily", "startDate": "2024-01-01"}'

# List medications
curl http://localhost:8000/api/medications

# Log a symptom
curl -X POST http://localhost:8000/api/symptoms \
  -H "Content-Type: application/json" \
  -d '{"symptom": "Headache", "severity": 3, "notes": "After lunch"}'
```

---

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `GOOGLE_API_KEY not found` | `.env` not loaded | Run from `backend/` directory with venv active |
| `Model gemini-2.5-flash-lite not found` | Invalid model name | Check `config.py` MODEL value; try `gemini-1.5-flash` |
| `Tavily search failed` | Network or quota | Check `TAVILY_API_KEY`; free tier = 1000/month |
| `Calendar credentials not found` | Missing OAuth file | Add `credentials/google_calendar_credentials.json` |
| `Gmail auth failed` | Wrong password type | Use **App Password** (16 chars), not your Gmail password |
| ChromaDB `sqlite3` version error | System sqlite too old | `pip install pysqlite3-binary` and patch `__init__` |
| `Port 8000 already in use` | Old process running | `kill $(lsof -t -i:8000)` |
