"""
agents/memory.py
────────────────
Memory Agent – The Librarian.

Responsibilities:
  READ operations:
    • Semantic search of symptom history via ChromaDB (vector store)
    • Structured queries against SQLite (doctors, medications, appointments)
  WRITE operations:
    • Log new symptoms to ChromaDB
    • Insert/update structured data in SQLite

The agent parses the Orchestrator's instruction to determine
whether to READ or WRITE and what category of data to access.
"""

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path

import chromadb
from langchain_core.messages import AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from chronic_chatbot.config import (
    GOOGLE_API_KEY,
    SUBAGENT_MODEL,
    SQLITE_DB_PATH,
    CHROMA_PERSIST_DIR,
)
from chronic_chatbot.state import AgentState

logger = logging.getLogger(__name__)

# ── ChromaDB Client ───────────────────────────────────────────
Path(CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
# ChromaDB's default EF uses all-MiniLM-L6-v2 via ONNX (no PyTorch needed)
symptom_collection = chroma_client.get_or_create_collection(
    name="symptom_history",
    metadata={"hnsw:space": "cosine"},
)
# Note: chromadb automatically uses its bundled ONNX embedding function
# (based on all-MiniLM-L6-v2) when no embedding_function is specified.

# ── SQLite DB Setup ───────────────────────────────────────────
def get_db_connection() -> sqlite3.Connection:
    Path(SQLITE_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables on first run (idempotent)."""
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
        """)
    logger.info("✅ SQLite DB initialised")


init_db()

# ── LLM for instruction parsing ───────────────────────────────
llm = ChatGoogleGenerativeAI(
    model=SUBAGENT_MODEL,
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
)

MEMORY_PARSE_PROMPT = """
Parse the following instruction and return a JSON object with these fields:
  - "operation": "read" or "write"
  - "data_type": "symptom" | "doctor" | "medication" | "appointment" | "profile"
  - "query_or_data": the search query (for read) or data dict (for write)

Return JSON only, no markdown.

Instruction: {instruction}
""".strip()


def parse_instruction(instruction: str) -> dict:
    from langchain_core.messages import HumanMessage
    prompt = MEMORY_PARSE_PROMPT.format(instruction=instruction)
    response = llm.invoke([HumanMessage(content=prompt)])
    raw = response.content.strip()
    try:
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"operation": "read", "data_type": "symptom", "query_or_data": instruction}


# ── Read Helpers ──────────────────────────────────────────────

def search_symptoms(query: str, top_k: int = 5) -> str:
    """Semantic search over symptom history in ChromaDB."""
    results = symptom_collection.query(
        query_texts=[query],
        n_results=top_k,
    )
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    if not docs:
        return "No matching symptom history found."
    lines = []
    for doc, meta in zip(docs, metas):
        lines.append(f"• [{meta.get('date', 'unknown date')}] {doc}")
    return "\n".join(lines)


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
    return "\n".join([f"• {r['name']} ({r['specialty']}) – {r['email']} / {r['phone']}" for r in rows])


def query_medications() -> str:
    with get_db_connection() as conn:
        rows = conn.execute("SELECT * FROM medications").fetchall()
    if not rows:
        return "No medications on record."
    return "\n".join([f"• {r['name']} {r['dose']} – {r['frequency']}" for r in rows])


# ── Write Helpers ─────────────────────────────────────────────

def log_symptom(symptom_text: str) -> str:
    """Embed and store a new symptom report in ChromaDB."""
    import uuid
    doc_id = str(uuid.uuid4())
    symptom_collection.add(
        documents=[symptom_text],
        metadatas=[{"date": datetime.now().isoformat()}],
        ids=[doc_id],
    )
    return f"Symptom logged: '{symptom_text}'"


def add_doctor(data: dict) -> str:
    with get_db_connection() as conn:
        conn.execute(
            "INSERT INTO doctors (name, specialty, email, phone, notes) VALUES (?,?,?,?,?)",
            (data.get("name"), data.get("specialty"), data.get("email"), data.get("phone"), data.get("notes")),
        )
    return f"Doctor '{data.get('name')}' added."


def add_medication(data: dict) -> str:
    with get_db_connection() as conn:
        conn.execute(
            "INSERT INTO medications (name, dose, frequency, start_date, notes) VALUES (?,?,?,?,?)",
            (data.get("name"), data.get("dose"), data.get("frequency"), data.get("start_date"), data.get("notes")),
        )
    return f"Medication '{data.get('name')}' added."


# ── Main Node ─────────────────────────────────────────────────

def memory_node(state: AgentState) -> dict:
    """
    LangGraph node – Memory Agent.

    Parses the Orchestrator's instruction, routes to the correct
    read/write operation, and returns results in memory_context.
    """
    logger.info("🗄️  Memory Agent activated")

    instruction = state["messages"][-1].content
    if "[Orchestrator → memory]" in instruction:
        instruction = instruction.split("]", 1)[-1].strip()

    parsed = parse_instruction(instruction)
    operation = parsed.get("operation", "read")
    data_type = parsed.get("data_type", "symptom")
    payload = parsed.get("query_or_data", instruction)

    result = ""

    if operation == "read":
        if data_type == "symptom":
            result = search_symptoms(payload if isinstance(payload, str) else str(payload))
        elif data_type == "doctor":
            name_hint = payload if isinstance(payload, str) else payload.get("name", "")
            result = query_doctors(name_hint)
        elif data_type == "medication":
            result = query_medications()
        else:
            result = f"Unknown read data_type: {data_type}"

    elif operation == "write":
        if data_type == "symptom":
            text = payload if isinstance(payload, str) else json.dumps(payload)
            result = log_symptom(text)
        elif data_type == "doctor":
            result = add_doctor(payload if isinstance(payload, dict) else {})
        elif data_type == "medication":
            result = add_medication(payload if isinstance(payload, dict) else {})
        else:
            result = f"Unknown write data_type: {data_type}"

    logger.info(f"🗄️  Memory Agent result: {result[:100]}…")

    ai_msg = AIMessage(content=f"[Memory Agent Result]\n{result}")

    return {
        "messages": [ai_msg],
        "memory_context": result,
    }
