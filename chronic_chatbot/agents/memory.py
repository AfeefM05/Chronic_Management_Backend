"""
agents/memory.py
────────────────
Memory Agent – The Librarian.

READ operations:
  • Semantic search of symptom history via ChromaDB (vector store)
  • Structured queries against SQLite (doctors, medications, appointments)

WRITE operations:
  • Log new symptoms to BOTH ChromaDB AND SQLite symptoms_log
  • Insert medications, appointments, doctors into SQLite

All errors are caught and returned as soft messages — never raises.
Uses shared utils for safe LLM calls and content extraction.

The Orchestrator sends instructions with explicit prefixes:
  "WRITE symptom: <text>"       → log_symptom_dual()
  "WRITE medication: ..."       → add_medication()
  "WRITE doctor: ..."           → add_doctor()
  "WRITE appointment: ..."      → add_appointment_record()
  "READ symptoms: <query>"      → search_symptoms()
  "READ medications"            → query_medications()
  "READ doctors"                → query_doctors()
  "READ appointments"           → query_appointments()
"""

import json
import logging
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

import chromadb
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from chronic_chatbot.config import (
    GOOGLE_API_KEY,
    SUBAGENT_MODEL,
    SQLITE_DB_PATH,
    CHROMA_PERSIST_DIR,
)
from chronic_chatbot.state import AgentState
from chronic_chatbot.utils import safe_llm_invoke, safe_content, strip_agent_prefix

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# ChromaDB
# ══════════════════════════════════════════════════════════════

Path(CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
symptom_collection = chroma_client.get_or_create_collection(
    name="symptom_history",
    metadata={"hnsw:space": "cosine"},
)


# ══════════════════════════════════════════════════════════════
# SQLite helpers
# ══════════════════════════════════════════════════════════════

def get_db_connection() -> sqlite3.Connection:
    Path(SQLITE_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create all tables on first run (idempotent)."""
    with get_db_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS doctors (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                name      TEXT NOT NULL,
                specialty TEXT,
                email     TEXT,
                phone     TEXT,
                notes     TEXT
            );

            CREATE TABLE IF NOT EXISTS medications (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT NOT NULL,
                dose        TEXT,
                frequency   TEXT,
                start_date  TEXT,
                end_date    TEXT,
                notes       TEXT
            );

            CREATE TABLE IF NOT EXISTS appointments (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                doctor_id   INTEGER REFERENCES doctors(id),
                date_time   TEXT NOT NULL,
                reason      TEXT,
                status      TEXT DEFAULT 'pending',
                calendar_id TEXT
            );

            CREATE TABLE IF NOT EXISTS user_profile (
                key   TEXT PRIMARY KEY,
                value TEXT
            );

            CREATE TABLE IF NOT EXISTS symptoms_log (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                symptom   TEXT NOT NULL,
                severity  INTEGER DEFAULT 2,
                notes     TEXT,
                logged_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
        """)
    logger.info("✅ SQLite DB initialised")


init_db()


# ══════════════════════════════════════════════════════════════
# LLM — for free-form instruction parsing when prefix is absent
# ══════════════════════════════════════════════════════════════

llm = ChatGoogleGenerativeAI(
    model=SUBAGENT_MODEL,
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
)

MEMORY_PARSE_PROMPT = """
You are a database instruction parser. Given the instruction below, output a JSON object:
  - "operation": "write" or "read"
  - "data_type":  "symptom" | "medication" | "doctor" | "appointment" | "profile"
  - "query_or_data": for writes: a dict with relevant fields; for reads: a search string

Rules:
  - experiencing/feeling/having a symptom → operation=write, data_type=symptom
  - starting/taking a medication → operation=write, data_type=medication
  - asking about history/past → operation=read

Return JSON only, no markdown.
Instruction: {instruction}
""".strip()


def _parse_free_form(instruction: str) -> dict:
    """Fallback: LLM-based parse for instructions without WRITE/READ prefix."""
    prompt  = MEMORY_PARSE_PROMPT.format(instruction=instruction)
    raw_str = safe_llm_invoke(
        llm,
        [HumanMessage(content=prompt)],
        fallback='{"operation":"read","data_type":"symptom","query_or_data":"recent"}',
    )
    try:
        cleaned = raw_str.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        result = json.loads(cleaned.strip())
        # Validate expected keys exist
        if "operation" in result and "data_type" in result:
            return result
    except (json.JSONDecodeError, ValueError):
        pass

    # Keyword fallback — no LLM needed
    low      = instruction.lower()
    is_write = any(w in low for w in [
        "i have", "i feel", "i'm feeling", "i've been", "i am",
        "experiencing", "hurts", "pain", "started taking", "prescribed",
    ])
    return {
        "operation":     "write" if is_write else "read",
        "data_type":     "symptom",
        "query_or_data": instruction,
    }


# ══════════════════════════════════════════════════════════════
# READ helpers
# ══════════════════════════════════════════════════════════════

def search_symptoms(query: str, top_k: int = 5) -> str:
    """Semantic search over ChromaDB + also return recent SQLite entries."""
    # 1. Semantic search via ChromaDB
    chroma_results: list[str] = []
    try:
        results = symptom_collection.query(query_texts=[query], n_results=top_k)
        docs  = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        for doc, meta in zip(docs, metas):
            chroma_results.append(f"• [{meta.get('date', '?')}] {doc}")
    except Exception as e:
        logger.warning(f"ChromaDB search error: {e}")

    # 2. Recent entries from SQLite symptoms_log (always include these)
    sql_results: list[str] = []
    try:
        with get_db_connection() as conn:
            rows = conn.execute(
                "SELECT symptom, severity, notes, logged_at FROM symptoms_log ORDER BY logged_at DESC LIMIT 10"
            ).fetchall()
        severity_labels = {1: "mild", 2: "moderate", 3: "moderate-severe", 4: "severe", 5: "critical"}
        for r in rows:
            sev = severity_labels.get(r["severity"], "?")
            note = f" ({r['notes']})" if r["notes"] else ""
            sql_results.append(f"• [{r['logged_at'][:16]}] {r['symptom']} — {sev}{note}")
    except Exception as e:
        logger.warning(f"SQLite symptom query error: {e}")

    # Merge — deduplicate by text
    all_lines = sql_results + [l for l in chroma_results if l not in sql_results]
    if not all_lines:
        return "No symptom history found in the database."
    return "Symptom history:\n" + "\n".join(all_lines)


def query_medications() -> str:
    with get_db_connection() as conn:
        rows = conn.execute("SELECT * FROM medications").fetchall()
    if not rows:
        return "No medications on record."
    return "Current medications:\n" + "\n".join(
        [f"• {r['name']} {r['dose'] or ''} — {r['frequency'] or ''}" for r in rows]
    )


def query_doctors(name_filter: str = "") -> str:
    with get_db_connection() as conn:
        if name_filter:
            rows = conn.execute(
                "SELECT * FROM doctors WHERE name LIKE ?", (f"%{name_filter}%",)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM doctors").fetchall()
    if not rows:
        return "No doctors found in records."
    return "Doctors on record:\n" + "\n".join(
        [f"• {r['name']} ({r['specialty']}) — {r['email'] or 'no email'} / {r['phone'] or 'no phone'}" for r in rows]
    )


def query_appointments() -> str:
    with get_db_connection() as conn:
        rows = conn.execute("""
            SELECT a.date_time, a.reason, a.status, d.name as doctor
            FROM appointments a LEFT JOIN doctors d ON d.id = a.doctor_id
            ORDER BY a.date_time DESC LIMIT 10
        """).fetchall()
    if not rows:
        return "No appointments on record."
    return "Appointments:\n" + "\n".join(
        [f"• [{r['date_time'][:16]}] {r['doctor']} — {r['reason']} ({r['status']})" for r in rows]
    )


# ══════════════════════════════════════════════════════════════
# WRITE helpers
# ══════════════════════════════════════════════════════════════

def log_symptom_dual(symptom_text: str, severity: int = 2, notes: str = "") -> str:
    """
    Write symptom to BOTH:
      1. ChromaDB (for semantic search by knowledge_agent later)
      2. SQLite symptoms_log (for the frontend /memory page)
    """
    now = datetime.now().isoformat()
    doc_id = str(uuid.uuid4())

    # 1. ChromaDB
    try:
        symptom_collection.add(
            documents=[symptom_text],
            metadatas=[{"date": now, "severity": severity}],
            ids=[doc_id],
        )
    except Exception as e:
        logger.warning(f"ChromaDB write error: {e}")

    # 2. SQLite symptoms_log
    try:
        with get_db_connection() as conn:
            conn.execute(
                "INSERT INTO symptoms_log (symptom, severity, notes, logged_at) VALUES (?,?,?,?)",
                (symptom_text, severity, notes or None, now),
            )
    except Exception as e:
        logger.error(f"SQLite symptom write error: {e}")
        return f"⚠️ Symptom partially logged (ChromaDB only): '{symptom_text}'"

    logger.info(f"🗄️ Symptom logged: {symptom_text!r}")
    return f"✅ Symptom logged to database: '{symptom_text}'"


def add_medication(data: dict) -> str:
    name = data.get("name") or data.get("medication") or str(data)
    dose = data.get("dose") or data.get("dosage", "")
    freq = data.get("frequency", "")
    with get_db_connection() as conn:
        # Check if already exists to avoid duplicates
        existing = conn.execute(
            "SELECT id FROM medications WHERE name LIKE ?", (f"%{name}%",)
        ).fetchone()
        if existing:
            return f"Medication '{name}' is already in your records."
        conn.execute(
            "INSERT INTO medications (name, dose, frequency, start_date, notes) VALUES (?,?,?,?,?)",
            (name, dose, freq, datetime.now().date().isoformat(), data.get("notes")),
        )
    return f"✅ Medication '{name}' saved to your records."


def add_doctor(data: dict) -> str:
    name = data.get("name", "Unknown")
    with get_db_connection() as conn:
        conn.execute(
            "INSERT INTO doctors (name, specialty, email, phone, notes) VALUES (?,?,?,?,?)",
            (name, data.get("specialty"), data.get("email"), data.get("phone"), data.get("notes")),
        )
    return f"✅ Doctor '{name}' added to your records."


def add_appointment_record(data: dict) -> str:
    doctor_name = data.get("doctor", data.get("doctor_name", "Unknown"))
    date_time   = data.get("date_time", data.get("dateTime", ""))
    reason      = data.get("reason", "Medical appointment")
    with get_db_connection() as conn:
        # Upsert doctor
        conn.execute(
            "INSERT OR IGNORE INTO doctors (name) VALUES (?)", (doctor_name,)
        )
        row = conn.execute(
            "SELECT id FROM doctors WHERE name = ?", (doctor_name,)
        ).fetchone()
        doc_id = row["id"] if row else None
        conn.execute(
            "INSERT INTO appointments (doctor_id, date_time, reason, status) VALUES (?,?,?,?)",
            (doc_id, date_time, reason, "scheduled"),
        )
    return f"✅ Appointment with {doctor_name} on {date_time} saved to records."


# ══════════════════════════════════════════════════════════════
# Instruction parser — prefix-first, LLM fallback
# ══════════════════════════════════════════════════════════════

def _parse_write_payload(data_type: str, payload_text: str) -> dict | str:
    """
    Convert free-form payload text into a dict for write operations.
    For symptoms, just return the text string.
    For medications, attempt to extract name/dose/frequency.
    """
    if data_type == "symptom":
        return payload_text.strip()

    # For medications: try to parse "name=..., dose=..., frequency=..."
    result = {}
    for kv in payload_text.split(","):
        kv = kv.strip()
        if "=" in kv:
            k, _, v = kv.partition("=")
            result[k.strip().lower()] = v.strip()
    if not result:
        result["name"] = payload_text.strip()
    return result


def parse_instruction(instruction: str) -> tuple[str, str, dict | str]:
    """
    Returns (operation, data_type, payload).

    Priority 1: Explicit prefixes from orchestrator
      "WRITE symptom: ..."
      "WRITE medication: ..."
      "READ symptoms: ..."
      "READ medications"
      "READ doctors"
      "READ appointments"

    Priority 2: LLM fallback for free-form instructions.
    """
    low = instruction.strip()

    if low.startswith("WRITE ") or low.startswith("write "):
        rest = low[6:].strip()
        if rest.startswith("symptom:") or rest.startswith("symptom "):
            payload_text = rest[rest.index(":")+1:].strip() if ":" in rest else rest[8:].strip()
            return ("write", "symptom", payload_text)
        elif rest.startswith("medication:") or rest.startswith("medication "):
            payload_text = rest[rest.index(":")+1:].strip() if ":" in rest else rest[11:].strip()
            return ("write", "medication", _parse_write_payload("medication", payload_text))
        elif rest.startswith("doctor:") or rest.startswith("doctor "):
            payload_text = rest[rest.index(":")+1:].strip() if ":" in rest else rest[7:].strip()
            return ("write", "doctor", _parse_write_payload("doctor", payload_text))
        elif rest.startswith("appointment:") or rest.startswith("appointment "):
            payload_text = rest[rest.index(":")+1:].strip() if ":" in rest else rest[12:].strip()
            return ("write", "appointment", _parse_write_payload("appointment", payload_text))

    if low.startswith("READ ") or low.startswith("read "):
        rest = low[5:].strip()
        if rest.startswith("symptom"):
            query = rest.replace("symptoms:", "").replace("symptom:", "").strip()
            return ("read", "symptom", query or "all")
        elif rest.startswith("medication"):
            return ("read", "medication", "all")
        elif rest.startswith("doctor"):
            name = rest.replace("doctors:", "").replace("doctor:", "").strip()
            return ("read", "doctor", name)
        elif rest.startswith("appointment"):
            return ("read", "appointment", "all")

    # Fallback: LLM parse
    parsed = _parse_free_form(instruction)
    return (
        parsed.get("operation", "read"),
        parsed.get("data_type", "symptom"),
        parsed.get("query_or_data", instruction),
    )


# ══════════════════════════════════════════════════════════════
# Main Node
# ══════════════════════════════════════════════════════════════

def memory_node(state: AgentState) -> dict:
    """LangGraph node – Memory Agent."""
    logger.info("🗄️  Memory Agent activated")

    # Extract instruction — safe_content handles list-of-parts responses
    raw_msg = state["messages"][-1]
    instruction = safe_content(raw_msg, "") if not isinstance(raw_msg.content, str) \
        else str(raw_msg.content)
    instruction = strip_agent_prefix(instruction, "memory")

    operation, data_type, payload = parse_instruction(instruction)
    logger.info(f"🗄️  op={operation} type={data_type} payload={str(payload)[:80]}")

    result = ""

    # ── READ ─────────────────────────────────────────────────
    if operation == "read":
        if data_type == "symptom":
            query = payload if isinstance(payload, str) else "recent"
            result = search_symptoms(query)
        elif data_type == "medication":
            result = query_medications()
        elif data_type == "doctor":
            name_hint = payload if isinstance(payload, str) else ""
            result = query_doctors(name_hint)
        elif data_type == "appointment":
            result = query_appointments()
        else:
            result = f"Unknown read type: {data_type}"

    # ── WRITE ────────────────────────────────────────────────
    elif operation == "write":
        if data_type == "symptom":
            # payload can be dict {"text": ..., "severity": ...} or plain string
            if isinstance(payload, dict):
                text     = payload.get("text") or payload.get("symptom") or instruction
                severity = int(payload.get("severity", 2))
                notes    = payload.get("notes", "")
            else:
                text     = str(payload)
                severity = 2
                notes    = ""
            result = log_symptom_dual(text, severity, notes)
        elif data_type == "medication":
            if isinstance(payload, str):
                payload = {"name": payload}
            result = add_medication(payload)
        elif data_type == "doctor":
            if isinstance(payload, str):
                payload = {"name": payload}
            result = add_doctor(payload)
        elif data_type == "appointment":
            if isinstance(payload, str):
                payload = {"reason": payload}
            result = add_appointment_record(payload)
        else:
            result = f"Unknown write type: {data_type}"

    logger.info(f"🗄️  Memory result: {result[:120]}")

    ai_msg = AIMessage(content=f"[Memory Agent Result]\n{result}")
    return {
        "messages":      [ai_msg],
        "memory_context": result,
    }
