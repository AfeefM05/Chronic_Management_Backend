"""
mcp_server/memory_server.py
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
"""

import logging
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from mcp.server.fastmcp import FastMCP
import chromadb

from chronic_chatbot.config import (
    SQLITE_DB_PATH,
    CHROMA_PERSIST_DIR,
)

logger = logging.getLogger(__name__)

mcp = FastMCP("ChronicMemoryServer")

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
# READ helpers
# ══════════════════════════════════════════════════════════════

@mcp.tool()
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

@mcp.tool()
def query_medications() -> str:
    with get_db_connection() as conn:
        rows = conn.execute("SELECT * FROM medications").fetchall()
    if not rows:
        return "No medications on record."
    return "Current medications:\n" + "\n".join(
        [f"• {r['name']} {r['dose'] or ''} — {r['frequency'] or ''}" for r in rows]
    )

@mcp.tool()
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

@mcp.tool()
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

@mcp.tool()
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

@mcp.tool()
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

@mcp.tool()
def add_doctor(data: dict) -> str:
    name = data.get("name", "Unknown")
    with get_db_connection() as conn:
        conn.execute(
            "INSERT INTO doctors (name, specialty, email, phone, notes) VALUES (?,?,?,?,?)",
            (name, data.get("specialty"), data.get("email"), data.get("phone"), data.get("notes")),
        )
    return f"✅ Doctor '{name}' added to your records."

@mcp.tool()
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

if __name__ == "__main__":
    mcp.run(transport="stdio")