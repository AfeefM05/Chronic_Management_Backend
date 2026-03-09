"""
agents/action.py
────────────────
Action Agent – The Doer.

Responsibilities:
  • Book appointments via Google Calendar API
  • Send email notifications to doctors via SendGrid
  • Return a plain confirmation string (never speaks to user directly)

Uses Gemini 1.5 Flash to parse the Orchestrator's natural-language
instruction into structured parameters.
"""

import json
import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime, timedelta

from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from chronic_chatbot.config import (
    GOOGLE_API_KEY,
    SUBAGENT_MODEL,
    GMAIL_SENDER_EMAIL,
    GMAIL_APP_PASSWORD,
    GOOGLE_CALENDAR_CREDENTIALS_PATH,
    GOOGLE_CALENDAR_TOKEN_PATH,
)
from chronic_chatbot.state import AgentState

logger = logging.getLogger(__name__)

llm = ChatGoogleGenerativeAI(
    model=SUBAGENT_MODEL,
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
)

# ── Instruction Parser ─────────────────────────────────────────
ACTION_PARSE_PROMPT = """
Parse this action instruction and return JSON with:
  - "action_type": "calendar" | "email" | "both"
  - "doctor_name": string
  - "doctor_email": string (if available, else empty string)
  - "appointment_date": ISO 8601 string or natural date string
  - "appointment_reason": string
  - "email_body": string (draft email body to the doctor, if needed)

Return JSON only.

Instruction: {instruction}
""".strip()


def parse_action_instruction(instruction: str) -> dict:
    prompt = ACTION_PARSE_PROMPT.format(instruction=instruction)
    response = llm.invoke([HumanMessage(content=prompt)])
    raw = response.content.strip()
    try:
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Action parser returned non-JSON; using defaults")
        return {
            "action_type": "calendar",
            "doctor_name": "Doctor",
            "doctor_email": "",
            "appointment_date": "",
            "appointment_reason": instruction,
            "email_body": "",
        }


# ── Google Calendar ────────────────────────────────────────────

def create_calendar_event(doctor_name: str, date_str: str, reason: str) -> str:
    """
    Create a Google Calendar event for the appointment.
    Requires credentials.json placed at GOOGLE_CALENDAR_CREDENTIALS_PATH.
    """
    try:
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from google.auth.transport.requests import Request
        from googleapiclient.discovery import build
        import os, pickle
        from pathlib import Path

        SCOPES = ["https://www.googleapis.com/auth/calendar"]
        creds = None

        token_path = Path(GOOGLE_CALENDAR_TOKEN_PATH)
        creds_path = Path(GOOGLE_CALENDAR_CREDENTIALS_PATH)

        if token_path.exists():
            creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not creds_path.exists():
                    return "❌ Google Calendar credentials file not found. Please add credentials/google_calendar_credentials.json"
                flow = InstalledAppFlow.from_client_secrets_file(str(creds_path), SCOPES)
                creds = flow.run_local_server(port=0)
            token_path.parent.mkdir(parents=True, exist_ok=True)
            token_path.write_text(creds.to_json())

        service = build("calendar", "v3", credentials=creds)

        # Parse date; default to tomorrow if parsing fails
        try:
            start_dt = datetime.fromisoformat(date_str)
        except ValueError:
            start_dt = datetime.now() + timedelta(days=1)
            start_dt = start_dt.replace(hour=10, minute=0, second=0, microsecond=0)

        end_dt = start_dt + timedelta(hours=1)

        event = {
            "summary": f"Appointment with {doctor_name}",
            "description": reason,
            "start": {"dateTime": start_dt.isoformat(), "timeZone": "Asia/Kolkata"},
            "end": {"dateTime": end_dt.isoformat(), "timeZone": "Asia/Kolkata"},
            "reminders": {
                "useDefault": False,
                "overrides": [
                    {"method": "email", "minutes": 24 * 60},
                    {"method": "popup", "minutes": 30},
                ],
            },
        }

        created_event = service.events().insert(calendarId="primary", body=event).execute()
        return f"✅ Calendar event created: {created_event.get('htmlLink')}"

    except ImportError:
        return "❌ Google Calendar libraries not installed. Run: pip install google-api-python-client google-auth-oauthlib"
    except Exception as e:
        logger.error(f"Calendar error: {e}")
        return f"❌ Calendar booking failed: {e}"


# ── Gmail SMTP Email (Python stdlib — zero extra packages) ────────

def send_email_to_doctor(doctor_email: str, doctor_name: str, subject: str, body: str) -> str:
    """
    Send an email to the doctor using Gmail SMTP + App Password.
    Uses Python's built-in smtplib — no pip package required.

    Setup (one-time, 2 minutes):
      1. Go to myaccount.google.com/security
      2. Enable 2-Step Verification (if not already on)
      3. Search "App Passwords" → Create one:
         App = Mail, Device = Other → name it "Chronic Chatbot"
      4. Copy the 16-character password → GMAIL_APP_PASSWORD in .env
    """
    if not GMAIL_SENDER_EMAIL or not GMAIL_APP_PASSWORD:
        return (
            "❌ Gmail not configured. Add GMAIL_SENDER_EMAIL and "
            "GMAIL_APP_PASSWORD to your .env file."
        )
    if not doctor_email:
        return "❌ Doctor email address not found in records."

    try:
        # Build the email message
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = f"Chronic Health Companion <{GMAIL_SENDER_EMAIL}>"
        msg["To"]      = doctor_email

        # Plain-text part
        text_part = MIMEText(body, "plain", "utf-8")

        # Simple HTML part so it looks clean in the doctor's inbox
        html_body = f"""
        <html><body style="font-family:Arial,sans-serif;font-size:14px;color:#333;">
          <div style="max-width:600px;margin:auto;padding:20px;border:1px solid #e0e0e0;border-radius:8px;">
            <h3 style="color:#2c7be5;">📅 Appointment Request</h3>
            <p>{body.replace(chr(10), '<br>')}</p>
            <hr style="border:none;border-top:1px solid #eee;">
            <p style="font-size:11px;color:#999;">Sent via Chronic Disease AI Companion</p>
          </div>
        </body></html>
        """
        html_part = MIMEText(html_body, "html", "utf-8")

        msg.attach(text_part)
        msg.attach(html_part)

        # Connect to Gmail SMTP via STARTTLS (port 587)
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=15) as server:
            server.ehlo()
            server.starttls()           # Upgrade to encrypted connection
            server.ehlo()
            server.login(GMAIL_SENDER_EMAIL, GMAIL_APP_PASSWORD)
            server.sendmail(
                from_addr=GMAIL_SENDER_EMAIL,
                to_addrs=[doctor_email],
                msg=msg.as_string(),
            )

        logger.info(f"✉️  Email sent to {doctor_email}")
        return f"✅ Email sent to Dr. {doctor_name} ({doctor_email})."

    except smtplib.SMTPAuthenticationError:
        return (
            "❌ Gmail authentication failed. "
            "Use an App Password (not your real Gmail password). "
            "Get one at: myaccount.google.com/apppasswords"
        )
    except smtplib.SMTPRecipientsRefused:
        return f"❌ Recipient address rejected: {doctor_email}"
    except smtplib.SMTPException as e:
        logger.error(f"SMTP error: {e}")
        return f"❌ SMTP error: {e}"
    except Exception as e:
        logger.error(f"Email error: {e}")
        return f"❌ Email failed: {e}"


# ── Main Node ─────────────────────────────────────────────────

def action_node(state: AgentState) -> dict:
    """
    LangGraph node – Action Agent.

    Parses the Orchestrator's instruction, executes calendar and/or
    email actions, and writes results to state.action_result.
    """
    logger.info("⚡ Action Agent activated")

    instruction = state["messages"][-1].content
    if "[Orchestrator → action]" in instruction:
        instruction = instruction.split("]", 1)[-1].strip()

    parsed = parse_action_instruction(instruction)
    action_type = parsed.get("action_type", "calendar")
    doctor_name = parsed.get("doctor_name", "Unknown Doctor")
    doctor_email = parsed.get("doctor_email", "")
    appointment_date = parsed.get("appointment_date", "")
    reason = parsed.get("appointment_reason", "Medical appointment")
    email_body = parsed.get(
        "email_body",
        f"Dear Dr. {doctor_name},\n\nI would like to request an appointment regarding {reason}.\n\nThank you.",
    )

    results = []

    if action_type in ("calendar", "both"):
        cal_result = create_calendar_event(doctor_name, appointment_date, reason)
        results.append(cal_result)

    if action_type in ("email", "both") and doctor_email:
        email_result = send_email_to_doctor(
            doctor_email=doctor_email,
            doctor_name=doctor_name,
            subject=f"Appointment Request – {reason}",
            body=email_body,
        )
        results.append(email_result)
    elif action_type in ("email", "both"):
        results.append("⚠️ Doctor email not found; skipping email.")

    combined_result = "\n".join(results)
    logger.info(f"⚡ Action Agent result: {combined_result[:100]}…")

    ai_msg = AIMessage(content=f"[Action Agent Result]\n{combined_result}")

    return {
        "messages": [ai_msg],
        "action_result": combined_result,
    }
