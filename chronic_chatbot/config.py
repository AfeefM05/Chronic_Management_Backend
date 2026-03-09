"""
config.py – Centralised settings loaded from .env
All modules import from here; never read os.environ directly.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

# ── LLM ───────────────────────────────────────────────────────
GOOGLE_API_KEY: str = os.environ["GOOGLE_API_KEY"]

# All agents use the same lean model — cheap, fast, still capable.
# gemini-2.5-flash-lite is the most cost-efficient 2.5-series model.
MODEL: str              = "gemini-2.5-flash-lite"
ORCHESTRATOR_MODEL: str = MODEL
SUBAGENT_MODEL: str     = MODEL

# ── Tavily ────────────────────────────────────────────────────
TAVILY_API_KEY: str = os.environ["TAVILY_API_KEY"]

# ── Gmail SMTP (Python built-in smtplib — zero extra packages) ───
# Use a Gmail account + App Password (not your real password)
# How to get App Password:
#   Google Account → Security → 2-Step Verification → App Passwords
#   Select app: Mail | Select device: Other → name it "Chronic Chatbot"
GMAIL_SENDER_EMAIL: str = os.environ.get("GMAIL_SENDER_EMAIL", "")  # your@gmail.com
GMAIL_APP_PASSWORD: str  = os.environ.get("GMAIL_APP_PASSWORD", "")  # 16-char App Password

# ── Google Calendar ───────────────────────────────────────────
GOOGLE_CALENDAR_CREDENTIALS_PATH: str = os.environ.get(
    "GOOGLE_CALENDAR_CREDENTIALS_PATH", "credentials/google_calendar_credentials.json"
)
GOOGLE_CALENDAR_TOKEN_PATH: str = os.environ.get(
    "GOOGLE_CALENDAR_TOKEN_PATH", "credentials/google_calendar_token.json"
)

# ── Database ──────────────────────────────────────────────────
SQLITE_DB_PATH: str = os.environ.get("SQLITE_DB_PATH", "data/chronic_chatbot.db")
CHROMA_PERSIST_DIR: str = os.environ.get("CHROMA_PERSIST_DIR", "data/chroma_store")

# ── Graph Safety ──────────────────────────────────────────────
MAX_AGENT_ITERATIONS: int = int(os.environ.get("MAX_AGENT_ITERATIONS", "10"))

# ── Logging ───────────────────────────────────────────────────
LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO")
