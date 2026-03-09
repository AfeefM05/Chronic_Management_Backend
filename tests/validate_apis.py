"""
validate_apis.py
────────────────
Tests every API key and model connection used in the chatbot.

Run with the project venv (IMPORTANT — always use venv python):
    cd /home/mohammed-afeef/Desktop/chronic_chatbot/backend
    venv/bin/python tests/validate_apis.py

Results are color-coded: ✅ PASS  ❌ FAIL  ⚠️  WARN  ⏭️  SKIP
"""

import sys
import os
import time
import smtplib
from pathlib import Path

# ── Guard: warn if not running inside the venv ─────────────────
VENV_PYTHON = Path(__file__).resolve().parents[1] / "venv" / "bin" / "python"
if not str(sys.executable).startswith(str(VENV_PYTHON.parent)):
    print(
        "\n\033[93m⚠️  WARNING: Not running inside the project venv!\033[0m\n"
        f"   Current Python : {sys.executable}\n"
        f"   Expected venv  : {VENV_PYTHON}\n\n"
        "   Run with:\n"
        "   \033[96mvenv/bin/python tests/validate_apis.py\033[0m\n"
        "   (from the backend/ directory)\n"
    )

# ── Add project root to sys.path ───────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]   # .../backend/
sys.path.insert(0, str(PROJECT_ROOT))

# ── Load .env ─────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

# ── ANSI colors ───────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

PASS_S = f"{GREEN}✅ PASS{RESET}"
FAIL_S = f"{RED}❌ FAIL{RESET}"
WARN_S = f"{YELLOW}⚠️  WARN{RESET}"
SKIP_S = f"{CYAN}⏭️  SKIP{RESET}"

results = []   # (name, status_str, detail)


def section(title: str):
    print(f"\n{BOLD}{CYAN}{'─' * 55}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'─' * 55}{RESET}")


def report(name: str, status: str, detail: str = ""):
    label = f"{name:<38}"
    print(f"  {label} {status}  {DIM}{detail}{RESET}")
    results.append((name, status, detail))


def env_check(var: str) -> str | None:
    val = os.environ.get(var, "").strip()
    if val:
        masked = val[:6] + "…" + val[-4:] if len(val) > 12 else "***"
        report(f"ENV: {var}", PASS_S, f"value={masked}")
        return val
    report(f"ENV: {var}", FAIL_S, "not set in .env")
    return None


# ══════════════════════════════════════════════════════════════
# SECTION 1 – Environment Variables
# ══════════════════════════════════════════════════════════════
section("1. Environment Variables")

GOOGLE_API_KEY     = env_check("GOOGLE_API_KEY")
TAVILY_API_KEY     = env_check("TAVILY_API_KEY")
GMAIL_SENDER_EMAIL = env_check("GMAIL_SENDER_EMAIL")
GMAIL_APP_PASSWORD = env_check("GMAIL_APP_PASSWORD")

# Optional – Google Calendar
GCAL_CREDS = os.environ.get("GOOGLE_CALENDAR_CREDENTIALS_PATH", "")
if GCAL_CREDS and (PROJECT_ROOT / GCAL_CREDS).exists():
    report("ENV: GOOGLE_CALENDAR_CREDENTIALS_PATH", PASS_S, f"{GCAL_CREDS} found")
elif GCAL_CREDS:
    report("ENV: GOOGLE_CALENDAR_CREDENTIALS_PATH", WARN_S, "set but file not found on disk")
else:
    report("ENV: GOOGLE_CALENDAR_CREDENTIALS_PATH", WARN_S, "not set — Calendar actions will be skipped")


# ══════════════════════════════════════════════════════════════
# SECTION 2 – Python Package Imports
# ══════════════════════════════════════════════════════════════
section("2. Python Package Imports")

# smtplib / email are stdlib — always available
for mod in ("smtplib", "email.mime.multipart", "email.mime.text", "sqlite3"):
    try:
        __import__(mod)
        report(f"stdlib: {mod}", PASS_S, "built-in")
    except ImportError:
        report(f"stdlib: {mod}", FAIL_S, "unexpected — check Python install")

# Third-party packages (must be in venv)
packages = [
    ("langgraph",              "langgraph"),
    ("langchain",              "langchain"),
    ("langchain_core",         "langchain-core"),
    ("langchain_google_genai", "langchain-google-genai"),
    ("tavily",                 "tavily-python"),
    ("chromadb",               "chromadb"),
    ("fastapi",                "fastapi"),
    ("pydantic",               "pydantic"),
    ("googleapiclient",        "google-api-python-client"),
    ("dotenv",                 "python-dotenv"),
]

for module, pkg in packages:
    try:
        __import__(module)
        report(f"import {module}", PASS_S)
    except ImportError:
        hint = f"  →  venv/bin/pip install {pkg}"
        report(f"import {module}", FAIL_S, hint)


# ══════════════════════════════════════════════════════════════
# SECTION 3 – Gemini API (Orchestrator — Pro)
# ══════════════════════════════════════════════════════════════
section("3. Gemini API – Model (gemini-2.5-flash-lite)")

if not GOOGLE_API_KEY:
    report("Gemini 1.5 Pro connection", SKIP_S, "GOOGLE_API_KEY not set")
else:
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.messages import HumanMessage

        t0 = time.time()
        llm_pro = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            google_api_key=GOOGLE_API_KEY,
            temperature=0,
        )
        resp = llm_pro.invoke([HumanMessage(content="Reply with exactly: GEMINI_PRO_OK")])
        latency = round((time.time() - t0) * 1000)
        content = resp.content.strip()

        if "GEMINI_PRO_OK" in content:
            report("Gemini 1.5 Pro – invoke()", PASS_S, f"reply='{content}'  {latency}ms")
        else:
            report("Gemini 1.5 Pro – invoke()", WARN_S, f"unexpected: '{content[:60]}'")
    except Exception as e:
        report("Gemini 1.5 Pro – invoke()", FAIL_S, str(e)[:120])


# ══════════════════════════════════════════════════════════════
# SECTION 4 – Gemini API (Sub-agent — Flash)
# ══════════════════════════════════════════════════════════════
section("4. Gemini API – Sub-Agent (same model, separate rate-limit check)")

if not GOOGLE_API_KEY:
    report("Gemini 1.5 Flash connection", SKIP_S, "GOOGLE_API_KEY not set")
else:
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.messages import HumanMessage

        t0 = time.time()
        llm_flash = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            google_api_key=GOOGLE_API_KEY,
            temperature=0,
        )
        resp = llm_flash.invoke([HumanMessage(content="Reply with exactly: GEMINI_FLASH_OK")])
        latency = round((time.time() - t0) * 1000)
        content = resp.content.strip()

        if "GEMINI_FLASH_OK" in content:
            report("Gemini 1.5 Flash – invoke()", PASS_S, f"reply='{content}'  {latency}ms")
        else:
            report("Gemini 1.5 Flash – invoke()", WARN_S, f"unexpected: '{content[:60]}'")
    except Exception as e:
        report("Gemini 1.5 Flash – invoke()", FAIL_S, str(e)[:120])


# ══════════════════════════════════════════════════════════════
# SECTION 5 – Gemini JSON Routing Mode
# ══════════════════════════════════════════════════════════════
section("5. Gemini JSON Routing Mode (Orchestrator Decision Test)")

if not GOOGLE_API_KEY:
    report("Gemini JSON routing test", SKIP_S, "GOOGLE_API_KEY not set")
else:
    try:
        import json
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.messages import HumanMessage, SystemMessage

        time.sleep(3)   # avoid free-tier rate-limit between Pro calls
        llm_pro = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            google_api_key=GOOGLE_API_KEY,
            temperature=0,
        )
        system = SystemMessage(content=(
            "You are a health routing engine. "
            "Return ONLY a JSON object with keys: reasoning, next_step. "
            "next_step must be one of: knowledge, memory, action, final_response."
        ))
        user = HumanMessage(
            content="I'm feeling dizzy again. Can you book an appointment with Dr. Smith?"
        )

        t0 = time.time()
        resp = llm_pro.invoke([system, user])
        latency = round((time.time() - t0) * 1000)
        raw = resp.content.strip()

        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        decision = json.loads(raw)
        next_step = decision.get("next_step", "")
        reasoning = decision.get("reasoning", "")[:80]

        if next_step in ("knowledge", "memory", "action", "final_response"):
            report("Gemini JSON routing – parse", PASS_S,
                   f"next_step='{next_step}'  {latency}ms")
            report("Gemini routing – reasoning", PASS_S, f"'{reasoning}…'")
        else:
            report("Gemini JSON routing – parse", WARN_S, f"unexpected next_step='{next_step}'")
    except json.JSONDecodeError as e:
        report("Gemini JSON routing – parse", FAIL_S, f"JSON decode error: {e}")
    except Exception as e:
        report("Gemini JSON routing – invoke", FAIL_S, str(e)[:120])


# ══════════════════════════════════════════════════════════════
# SECTION 6 – Tavily Search API
# ══════════════════════════════════════════════════════════════
section("6. Tavily Search API (Knowledge Agent)")

if not TAVILY_API_KEY:
    report("Tavily API connection", SKIP_S, "TAVILY_API_KEY not set")
else:
    try:
        from tavily import TavilyClient

        t0 = time.time()
        tc = TavilyClient(api_key=TAVILY_API_KEY)
        raw = tc.search(
            query="Type 2 Diabetes management guidelines",
            search_depth="basic",
            max_results=2,
        )
        latency = round((time.time() - t0) * 1000)

        # Normalise: newer SDK returns SearchResponse object, not dict
        if isinstance(raw, dict):
            resp = raw
        else:
            resp = {
                "results": getattr(raw, "results", []) or [],
                "answer":  getattr(raw, "answer",  "") or "",
            }
        num_results = len(resp.get("results", []))
        answer      = (resp.get("answer") or "")[:80]

        report("Tavily API – search()", PASS_S, f"results={num_results}  {latency}ms")
        if answer:
            report("Tavily API – answer snippet", PASS_S, f"'{answer}…'")
        else:
            report("Tavily API – answer snippet", WARN_S, "no answer field in response")
    except Exception as e:
        report("Tavily API – search()", FAIL_S, str(e)[:120])


# ══════════════════════════════════════════════════════════════
# SECTION 7 – ChromaDB (uses EphemeralClient to avoid disk/path issues)
# ══════════════════════════════════════════════════════════════
section("7. ChromaDB – Vector Store (Memory Agent)")

try:
    import chromadb

    # EphemeralClient = in-memory only, no disk path issues, perfect for validation
    t0 = time.time()
    client = chromadb.EphemeralClient()
    col = client.get_or_create_collection("api_validation_test")

    col.add(
        documents=["Validation test: patient reports mild dizziness after skipping breakfast"],
        metadatas=[{"date": "2026-03-07", "test": "true"}],
        ids=["validate-001"],
    )
    latency_write = round((time.time() - t0) * 1000)
    report("ChromaDB – EphemeralClient()", PASS_S, "in-memory client OK")
    report("ChromaDB – collection.add()", PASS_S, f"{latency_write}ms")

    # Semantic query
    t0 = time.time()
    search_result = col.query(query_texts=["lightheaded vertigo"], n_results=1)
    latency_query = round((time.time() - t0) * 1000)
    docs = search_result.get("documents", [[]])[0]

    if docs:
        report("ChromaDB – semantic query()", PASS_S,
               f"hit='{docs[0][:55]}…'  {latency_query}ms")
    else:
        report("ChromaDB – semantic query()", WARN_S, "no results returned")

    # Also verify the persistent store path is accessible (don't open it)
    chroma_dir = PROJECT_ROOT / "data" / "chroma_store"
    chroma_dir.mkdir(parents=True, exist_ok=True)
    report("ChromaDB – persistent store dir", PASS_S, str(chroma_dir))

except Exception as e:
    report("ChromaDB", FAIL_S, str(e)[:120])


# ══════════════════════════════════════════════════════════════
# SECTION 8 – SQLite
# ══════════════════════════════════════════════════════════════
section("8. SQLite – Structured DB (Memory Agent)")

try:
    import sqlite3

    db_path = PROJECT_ROOT / "data" / "chronic_chatbot.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS _validate_test (
            id  INTEGER PRIMARY KEY,
            val TEXT
        )
    """)
    cur.execute("INSERT INTO _validate_test (val) VALUES ('ok')")
    conn.commit()
    report("SQLite – connect + CREATE TABLE", PASS_S, f"path={db_path.name}")
    report("SQLite – INSERT", PASS_S)

    cur.execute("SELECT val FROM _validate_test WHERE val='ok'")
    row = cur.fetchone()
    if row and row[0] == "ok":
        report("SQLite – SELECT", PASS_S, f"value='{row[0]}'")
    else:
        report("SQLite – SELECT", FAIL_S, "row not found")

    cur.execute("DROP TABLE _validate_test")
    conn.commit()
    conn.close()
    report("SQLite – cleanup test table", PASS_S)

    # Check main app tables
    conn2 = sqlite3.connect(str(db_path))
    cur2 = conn2.cursor()
    cur2.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cur2.fetchall()]
    conn2.close()

    for tbl in ["doctors", "medications", "appointments", "user_profile"]:
        if tbl in tables:
            report(f"SQLite – table '{tbl}' exists", PASS_S)
        else:
            report(f"SQLite – table '{tbl}' exists", WARN_S,
                   "auto-created when app starts (init_db)")

except Exception as e:
    report("SQLite", FAIL_S, str(e)[:120])


# ══════════════════════════════════════════════════════════════
# SECTION 9 – Gmail SMTP (replaces SendGrid)
# ══════════════════════════════════════════════════════════════
section("9. Gmail SMTP – Email (Action Agent)")

if not GMAIL_SENDER_EMAIL or not GMAIL_APP_PASSWORD:
    report("Gmail SMTP – credentials", SKIP_S,
           "GMAIL_SENDER_EMAIL or GMAIL_APP_PASSWORD not set in .env")
    report("Gmail SMTP – how to get App Password", WARN_S,
           "myaccount.google.com → Security → 2-Step → App Passwords")
else:
    try:
        # Test STARTTLS connection to Gmail (no email actually sent)
        t0 = time.time()
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=10) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(GMAIL_SENDER_EMAIL, GMAIL_APP_PASSWORD)
        latency = round((time.time() - t0) * 1000)

        report("Gmail SMTP – STARTTLS connect", PASS_S, f"smtp.gmail.com:587  {latency}ms")
        report("Gmail SMTP – login()", PASS_S, f"auth OK for {GMAIL_SENDER_EMAIL}")
        report("Gmail SMTP – ready to send", PASS_S,
               "no email sent — this is a connection test only")

    except smtplib.SMTPAuthenticationError:
        report("Gmail SMTP – login()", FAIL_S,
               "Auth failed — are you using an App Password? (not your real Gmail password)")
        report("Gmail SMTP – fix", WARN_S,
               "myaccount.google.com → Security → App Passwords → Create")
    except smtplib.SMTPException as e:
        report("Gmail SMTP – connect", FAIL_S, str(e)[:120])
    except Exception as e:
        report("Gmail SMTP – connect", FAIL_S, str(e)[:120])


# ══════════════════════════════════════════════════════════════
# SECTION 10 – Google Calendar
# ══════════════════════════════════════════════════════════════
section("10. Google Calendar API (Action Agent – Calendar)")

creds_path = PROJECT_ROOT / os.environ.get(
    "GOOGLE_CALENDAR_CREDENTIALS_PATH",
    "credentials/google_calendar_credentials.json"
)
token_path = PROJECT_ROOT / os.environ.get(
    "GOOGLE_CALENDAR_TOKEN_PATH",
    "credentials/google_calendar_token.json"
)

if not creds_path.exists() and not token_path.exists():
    report("Google Calendar – credentials", WARN_S,
           "credentials.json not found — Calendar booking will be skipped")
    report("Google Calendar – fix", SKIP_S,
           "console.cloud.google.com → APIs → Calendar → OAuth Credentials")
else:
    try:
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
        from googleapiclient.discovery import build

        SCOPES = ["https://www.googleapis.com/auth/calendar"]
        creds = None

        if token_path.exists():
            creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)

        if creds and creds.valid:
            service = build("calendar", "v3", credentials=creds)
            cal_list = service.calendarList().list().execute()
            num_cals = len(cal_list.get("items", []))
            report("Google Calendar – auth token valid", PASS_S)
            report("Google Calendar – calendarList()", PASS_S, f"calendars={num_cals}")
        elif creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            report("Google Calendar – token refreshed", PASS_S)
        else:
            report("Google Calendar – token status", WARN_S,
                   "Token missing — run app once to trigger OAuth browser flow")
    except Exception as e:
        report("Google Calendar – connection", FAIL_S, str(e)[:120])


# ══════════════════════════════════════════════════════════════
# SECTION 11 – LangGraph Workflow Compilation
# ══════════════════════════════════════════════════════════════
section("11. LangGraph – Workflow Compilation Test")

if not GOOGLE_API_KEY:
    report("LangGraph graph compile", SKIP_S,
           "GOOGLE_API_KEY required (agents import LLM at module level)")
else:
    try:
        t0 = time.time()
        from chronic_chatbot.graph import build_graph
        app = build_graph()
        latency = round((time.time() - t0) * 1000)
        report("LangGraph – build_graph()", PASS_S, f"compiled in {latency}ms")

        nodes = list(app.get_graph().nodes.keys())
        report("LangGraph – nodes registered", PASS_S, f"{nodes}")
    except Exception as e:
        report("LangGraph – build_graph()", FAIL_S, str(e)[:120])


# ══════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════
section("SUMMARY")

total   = len(results)
passed  = sum(1 for _, s, _ in results if "PASS" in s)
failed  = sum(1 for _, s, _ in results if "FAIL" in s)
warned  = sum(1 for _, s, _ in results if "WARN" in s)
skipped = sum(1 for _, s, _ in results if "SKIP" in s)

print(f"\n  {BOLD}Total checks  : {total}{RESET}")
print(f"  {GREEN}Passed        : {passed}{RESET}")
print(f"  {RED}Failed        : {failed}{RESET}")
print(f"  {YELLOW}Warnings      : {warned}{RESET}")
print(f"  {CYAN}Skipped       : {skipped}{RESET}")

if failed == 0 and warned == 0:
    print(f"\n  {GREEN}{BOLD}🎉 All systems go! The chatbot is fully operational.{RESET}\n")
elif failed == 0:
    print(f"\n  {YELLOW}{BOLD}⚠️  Passed with warnings. Review items above.{RESET}\n")
else:
    print(f"\n  {RED}{BOLD}❌ {failed} check(s) failed.{RESET}\n")
    print(f"  {BOLD}Failed checks:{RESET}")
    for name, status, detail in results:
        if "FAIL" in status:
            print(f"    • {name}: {detail}")
    print()

print(f"  {DIM}Run with venv: cd backend && venv/bin/python tests/validate_apis.py{RESET}\n")

sys.exit(0 if failed == 0 else 1)
