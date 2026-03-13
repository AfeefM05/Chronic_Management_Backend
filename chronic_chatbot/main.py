"""
main.py – FastAPI Application Entry Point

Exposes:
  POST /chat          – 4-agent LangGraph chat
  GET  /health        – health probe

  # Medications CRUD
  GET    /api/medications
  POST   /api/medications
  DELETE /api/medications/{id}

  # Appointments CRUD  (all three sync with Google Calendar)
  GET    /api/appointments
  POST   /api/appointments     → also creates a Calendar event
  PATCH  /api/appointments/{id}→ reschedules the Calendar event
  DELETE /api/appointments/{id}→ also deletes the Calendar event

  # Doctors
  GET    /api/doctors
  POST   /api/doctors

  # Profile (key-value store)
  GET    /api/profile
  PUT    /api/profile

  # Symptom history
  GET    /api/symptoms
  POST   /api/symptoms
"""

import logging
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage

from chronic_chatbot.config import LOG_LEVEL, SQLITE_DB_PATH
from chronic_chatbot.graph import graph_app
from chronic_chatbot.utils import safe_content
# Calendar helpers live in the MCP server module (action.py is now just the agent client)
from chronic_chatbot.mcp_server.action_server import (
    create_calendar_event,
    update_calendar_event,
    delete_calendar_event,
)

# ── Logging ───────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)


# ── DB helper (reuse memory agent's schema) ───────────────────
def get_conn() -> sqlite3.Connection:
    Path(SQLITE_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ── App Lifespan ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Chronic Chatbot starting up…")
    yield
    logger.info("🛑 Chronic Chatbot shutting down…")


# ── FastAPI App ───────────────────────────────────────────────
app = FastAPI(
    title="Chronic Disease Chatbot API",
    description="4-Agent LangGraph system for chronic disease management",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════
# Request / Response schemas
# ══════════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    message: str
    user_id: str = "default_user"

class ChatResponse(BaseModel):
    reply: str
    agent_path: list[str] = []

class MedicationIn(BaseModel):
    name: str
    dosage: str
    frequency: str
    startDate: str = ""
    notes: Optional[str] = None

class AppointmentIn(BaseModel):
    doctorName: str
    specialty: str = ""
    dateTime: str
    location: str = ""
    notes: Optional[str] = None
    status: str = "scheduled"

class AppointmentUpdate(BaseModel):
    doctorName: Optional[str] = None
    specialty: Optional[str] = None
    dateTime: Optional[str] = None
    location: Optional[str] = None
    notes: Optional[str] = None
    status: Optional[str] = None

class DoctorIn(BaseModel):
    name: str
    specialty: str = ""
    email: str = ""
    phone: str = ""
    notes: str = ""

class SymptomIn(BaseModel):
    symptom: str
    severity: int = 2
    notes: Optional[str] = None


# ── In-memory chat sessions ───────────────────────────────────
SESSIONS: dict = {}


# ══════════════════════════════════════════════════════════════
# Health
# ══════════════════════════════════════════════════════════════

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "chronic_chatbot"}


# ══════════════════════════════════════════════════════════════
# Chat
# ══════════════════════════════════════════════════════════════

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    logger.info(f"💬 New message from {request.user_id}: {request.message[:80]}")

    session = SESSIONS.get(request.user_id, {
        "messages": [],
        "user_profile": {},
        "next_step": "",
        "current_plan": [],
        "knowledge_context": None,
        "memory_context": None,
        "action_result": None,
    })
    session["messages"] = session["messages"] + [HumanMessage(content=request.message)]

    # ── Run the agent graph ─────────────────────────────────────
    error_reply: str | None = None
    try:
        result_state = await graph_app.ainvoke(session)
    except Exception as e:
        err_str = str(e)
        logger.error(f"Graph execution error: {e}", exc_info=True)

        # Keep the session so conversation history is preserved
        result_state = session

        # Map known error types to friendly messages
        if "Network is unreachable" in err_str or "Errno 101" in err_str:
            error_reply = (
                "I'm having trouble reaching the internet right now. "
                "Please check your connection and try again."
            )
        elif "bad character range" in err_str:
            error_reply = (
                "An internal routing error occurred. The team has been notified. "
                "Please retry your message."
            )
        elif "gemini" in err_str.lower() or "model" in err_str.lower():
            error_reply = (
                "The AI model is temporarily unavailable. "
                "Please try again in a few seconds."
            )
        else:
            error_reply = (
                "Something went wrong processing your request. "
                "Please try again — your conversation history is preserved."
            )

    SESSIONS[request.user_id] = result_state

    # ── If error occurred, return the friendly message ──────────
    if error_reply:
        logger.warning(f"Returning error message to user: {error_reply[:60]}")
        return ChatResponse(reply=error_reply)

    # ── Extract the final AI reply ──────────────────────────────
    DEFAULT_REPLY = (
        "I processed your request but couldn't generate a text response. "
        "Please try rephrasing your message."
    )
    messages = result_state.get("messages", [])
    final_reply = DEFAULT_REPLY

    for msg in reversed(messages):
        if not isinstance(msg, AIMessage):
            continue
        # safe_content handles both str and list-of-parts content
        content_str = safe_content(msg, "").strip()
        if content_str and not content_str.startswith("["):
            final_reply = content_str
            break

    # Last-resort: use whatever the last message says
    if final_reply == DEFAULT_REPLY and messages:
        final_reply = safe_content(messages[-1], DEFAULT_REPLY)

    logger.info(f"✅ Reply for {request.user_id}: {final_reply[:80]}")
    return ChatResponse(reply=final_reply)


# ══════════════════════════════════════════════════════════════
# Medications CRUD
# ══════════════════════════════════════════════════════════════

@app.get("/api/medications")
def get_medications():
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM medications").fetchall()
    return [dict(r) for r in rows]


@app.post("/api/medications", status_code=201)
def add_medication(med: MedicationIn):
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO medications (name, dose, frequency, start_date, notes) VALUES (?,?,?,?,?)",
            (med.name, med.dosage, med.frequency, med.startDate, med.notes),
        )
        new_id = cur.lastrowid
    return {"id": new_id, **med.model_dump()}


@app.delete("/api/medications/{med_id}")
def delete_medication(med_id: int):
    with get_conn() as conn:
        conn.execute("DELETE FROM medications WHERE id = ?", (med_id,))
    return {"deleted": med_id}


# ══════════════════════════════════════════════════════════════
# Appointments CRUD
# ══════════════════════════════════════════════════════════════

@app.get("/api/appointments")
def get_appointments():
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT a.id, a.date_time as dateTime, a.reason, a.status,
                   a.calendar_id as calendarId,
                   d.name as doctorName, d.specialty,
                   COALESCE(d.phone, '') as location
            FROM appointments a
            LEFT JOIN doctors d ON d.id = a.doctor_id
        """).fetchall()
    return [dict(r) for r in rows]


@app.post("/api/appointments", status_code=201)
def add_appointment(appt: AppointmentIn):
    # ── 1. Sync to Google Calendar ───────────────────────────
    cal_msg, event_id = create_calendar_event(
        doctor_name=appt.doctorName,
        date_str=appt.dateTime,
        reason=appt.notes or appt.specialty or "Medical appointment",
        location=appt.location,
    )
    logger.info(f"Calendar sync: {cal_msg}")

    # ── 2. Save to SQLite ────────────────────────────────────
    with get_conn() as conn:
        # Upsert doctor
        conn.execute(
            "INSERT OR IGNORE INTO doctors (name, specialty) VALUES (?, ?)",
            (appt.doctorName, appt.specialty),
        )
        row_d = conn.execute(
            "SELECT id FROM doctors WHERE name = ?", (appt.doctorName,)
        ).fetchone()
        doctor_id = row_d["id"] if row_d else None

        cur_a = conn.execute(
            "INSERT INTO appointments (doctor_id, date_time, reason, status, calendar_id) "
            "VALUES (?,?,?,?,?)",
            (doctor_id, appt.dateTime, appt.notes or appt.specialty, appt.status, event_id),
        )
        new_id = cur_a.lastrowid

    return {
        "id": new_id,
        "calendarId": event_id,
        "calendarMessage": cal_msg,
        **appt.model_dump(),
    }


@app.patch("/api/appointments/{appt_id}")
def reschedule_appointment(appt_id: int, update: AppointmentUpdate):
    """Reschedule or update an appointment — syncs to Google Calendar."""
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT a.*, d.name as doctorName, d.specialty
            FROM appointments a
            LEFT JOIN doctors d ON d.id = a.doctor_id
            WHERE a.id = ?
            """,
            (appt_id,),
        ).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail=f"Appointment {appt_id} not found")

    existing = dict(row)
    new_doctor   = update.doctorName or existing.get("doctorName", "")
    new_date     = update.dateTime   or existing.get("date_time", "")
    new_reason   = update.notes      or existing.get("reason", "Medical appointment")
    new_location = update.location   or ""
    new_status   = update.status     or existing.get("status", "scheduled")
    calendar_id  = existing.get("calendar_id", "")

    # ── 1. Update Google Calendar event ─────────────────────
    cal_msg = update_calendar_event(
        event_id=calendar_id,
        doctor_name=new_doctor,
        date_str=new_date,
        reason=new_reason,
        location=new_location,
    )
    logger.info(f"Calendar update: {cal_msg}")

    # Extract new event_id if the old one was missing and a new event was created
    new_cal_id = calendar_id  # keep existing unless create_calendar_event returned a new one

    # ── 2. Update SQLite ─────────────────────────────────────
    with get_conn() as conn:
        conn.execute(
            """
            UPDATE appointments
            SET date_time = ?, reason = ?, status = ?, calendar_id = ?
            WHERE id = ?
            """,
            (new_date, new_reason, new_status, new_cal_id, appt_id),
        )

    return {
        "id": appt_id,
        "calendarMessage": cal_msg,
        "dateTime": new_date,
        "status": new_status,
    }


@app.delete("/api/appointments/{appt_id}")
def delete_appointment(appt_id: int):
    """Cancel an appointment — also deletes the Google Calendar event."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT calendar_id FROM appointments WHERE id = ?", (appt_id,)
        ).fetchone()

    calendar_id = row["calendar_id"] if row else None

    # ── 1. Delete from Google Calendar ──────────────────────
    cal_msg = delete_calendar_event(calendar_id or "")
    logger.info(f"Calendar delete: {cal_msg}")

    # ── 2. Delete from SQLite ────────────────────────────────
    with get_conn() as conn:
        conn.execute("DELETE FROM appointments WHERE id = ?", (appt_id,))

    return {"deleted": appt_id, "calendarMessage": cal_msg}


# ══════════════════════════════════════════════════════════════
# Doctors
# ══════════════════════════════════════════════════════════════

@app.get("/api/doctors")
def get_doctors():
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM doctors").fetchall()
    return [dict(r) for r in rows]


@app.post("/api/doctors", status_code=201)
def add_doctor(doctor: DoctorIn):
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO doctors (name, specialty, email, phone, notes) VALUES (?,?,?,?,?)",
            (doctor.name, doctor.specialty, doctor.email, doctor.phone, doctor.notes),
        )
    return {"id": cur.lastrowid, **doctor.model_dump()}


# ══════════════════════════════════════════════════════════════
# Symptom history (stored in ChromaDB; expose list via memory agent's log)
# ══════════════════════════════════════════════════════════════

@app.get("/api/symptoms")
def get_symptoms():
    """Return all symptom entries from the SQLite side (simplified log)."""
    with get_conn() as conn:
        # Symptoms live in ChromaDB, so we just return a lightweight list
        # from a symptoms_log table we create here if missing
        conn.execute("""
            CREATE TABLE IF NOT EXISTS symptoms_log (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                symptom  TEXT NOT NULL,
                severity INTEGER DEFAULT 2,
                notes    TEXT,
                logged_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        rows = conn.execute("SELECT * FROM symptoms_log ORDER BY logged_at DESC").fetchall()
    return [dict(r) for r in rows]


@app.post("/api/symptoms", status_code=201)
def log_symptom(symptom: SymptomIn):
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS symptoms_log (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                symptom  TEXT NOT NULL,
                severity INTEGER DEFAULT 2,
                notes    TEXT,
                logged_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cur = conn.execute(
            "INSERT INTO symptoms_log (symptom, severity, notes) VALUES (?,?,?)",
            (symptom.symptom, symptom.severity, symptom.notes),
        )
    return {"id": cur.lastrowid, **symptom.model_dump()}


@app.delete("/api/symptoms/{symptom_id}")
def delete_symptom(symptom_id: int):
    with get_conn() as conn:
        conn.execute("DELETE FROM symptoms_log WHERE id = ?", (symptom_id,))
    return {"deleted": symptom_id}


# ══════════════════════════════════════════════════════════════
# Profile (key-value pairs)
# ══════════════════════════════════════════════════════════════

@app.get("/api/profile")
def get_profile():
    with get_conn() as conn:
        rows = conn.execute("SELECT key, value FROM user_profile").fetchall()
    return {r["key"]: r["value"] for r in rows}


@app.put("/api/profile")
def update_profile(data: dict):
    with get_conn() as conn:
        for key, value in data.items():
            conn.execute(
                "INSERT INTO user_profile (key, value) VALUES (?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, str(value)),
            )
    return {"updated": list(data.keys())}


# ── Dev Server Entry Point ────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("chronic_chatbot.main:app", host="0.0.0.0", port=8000, reload=True)
