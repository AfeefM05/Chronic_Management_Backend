"""
agents/action.py
────────────────
Action Agent – The Doer.

Responsibilities:
  • Create / Update / Delete Google Calendar events
  • Send email notifications to doctors via Gmail SMTP
  • Return a plain confirmation string (never speaks to user directly)

Calendar functions are also importable as utilities by main.py's REST
endpoints so that booking an appointment via the UI also syncs to Google
Calendar automatically.
"""

import json
import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime, timedelta
from pathlib import Path

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
from chronic_chatbot.utils import safe_llm_invoke, safe_content, strip_agent_prefix

logger = logging.getLogger(__name__)

llm = ChatGoogleGenerativeAI(
    model=SUBAGENT_MODEL,
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
)


# ══════════════════════════════════════════════════════════════
# Google Calendar helpers
# ══════════════════════════════════════════════════════════════

SCOPES = ["https://www.googleapis.com/auth/calendar"]
TIMEZONE = "Asia/Kolkata"


def _get_calendar_service():
    """
    Build and return an authenticated Google Calendar service object.
    Automatically refreshes the token if expired.
    Raises RuntimeError with a user-friendly message on failure.
    """
    try:
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from google.auth.transport.requests import Request
        from googleapiclient.discovery import build
    except ImportError:
        raise RuntimeError(
            "Google Calendar libraries not installed. "
            "Run: pip install google-api-python-client google-auth-oauthlib"
        )

    # Resolve paths relative to the backend project root (not CWD)
    project_root = Path(__file__).resolve().parent.parent.parent
    token_path = project_root / GOOGLE_CALENDAR_TOKEN_PATH
    creds_path = project_root / GOOGLE_CALENDAR_CREDENTIALS_PATH

    creds = None

    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            from google.auth.transport.requests import Request
            creds.refresh(Request())
            token_path.write_text(creds.to_json())
        else:
            if not creds_path.exists():
                raise RuntimeError(
                    f"Google Calendar credentials file not found at {creds_path}. "
                    "Please add credentials/google_calendar_credentials.json"
                )
            flow = InstalledAppFlow.from_client_secrets_file(str(creds_path), SCOPES)
            creds = flow.run_local_server(port=0)
            token_path.parent.mkdir(parents=True, exist_ok=True)
            token_path.write_text(creds.to_json())

    return build("calendar", "v3", credentials=creds)


def _parse_datetime(date_str: str) -> datetime:
    """
    Try to parse an ISO 8601 datetime string.
    Falls back to tomorrow 10:00 AM if the string is empty or unparseable.
    """
    if not date_str:
        dt = datetime.now() + timedelta(days=1)
        return dt.replace(hour=10, minute=0, second=0, microsecond=0)
    try:
        return datetime.fromisoformat(date_str.replace("Z", "+00:00").replace("z", "+00:00"))
    except ValueError:
        # Try common human-readable formats
        for fmt in ("%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
            try:
                return datetime.strptime(date_str[:len(fmt)], fmt)
            except ValueError:
                continue
        # Absolute fallback
        dt = datetime.now() + timedelta(days=1)
        return dt.replace(hour=10, minute=0, second=0, microsecond=0)


def create_calendar_event(
    doctor_name: str,
    date_str: str,
    reason: str,
    location: str = "",
) -> tuple[str, str]:
    """
    Create a Google Calendar event for an appointment.

    Returns:
        (result_message, calendar_event_id)
        calendar_event_id is empty string on failure.
    """
    try:
        service = _get_calendar_service()

        start_dt = _parse_datetime(date_str)
        end_dt = start_dt + timedelta(hours=1)

        event_body = {
            "summary": f"Appointment with {doctor_name}",
            "description": reason,
            "location": location,
            "start": {"dateTime": start_dt.isoformat(), "timeZone": TIMEZONE},
            "end":   {"dateTime": end_dt.isoformat(),   "timeZone": TIMEZONE},
            "reminders": {
                "useDefault": False,
                "overrides": [
                    {"method": "email",  "minutes": 24 * 60},
                    {"method": "popup",  "minutes": 30},
                ],
            },
        }

        created = service.events().insert(calendarId="primary", body=event_body).execute()
        event_id = created.get("id", "")
        link = created.get("htmlLink", "")

        logger.info(f"📅 Calendar event created: {event_id}")
        return f"✅ Calendar event created. View: {link}", event_id

    except RuntimeError as e:
        return f"❌ {e}", ""
    except Exception as e:
        logger.error(f"Calendar create error: {e}", exc_info=True)
        return f"❌ Calendar booking failed: {e}", ""


def update_calendar_event(
    event_id: str,
    doctor_name: str,
    date_str: str,
    reason: str,
    location: str = "",
) -> str:
    """
    Update an existing Google Calendar event (reschedule or rename).
    If event_id is empty or event not found, creates a new one instead.
    """
    if not event_id:
        msg, _ = create_calendar_event(doctor_name, date_str, reason, location)
        return msg

    try:
        service = _get_calendar_service()

        # Fetch existing event first
        try:
            existing = service.events().get(calendarId="primary", eventId=event_id).execute()
        except Exception:
            # Event not found — create fresh
            msg, _ = create_calendar_event(doctor_name, date_str, reason, location)
            return msg + " (original event not found; created new)"

        start_dt = _parse_datetime(date_str)
        end_dt   = start_dt + timedelta(hours=1)

        existing.update({
            "summary":     f"Appointment with {doctor_name}",
            "description": reason,
            "location":    location,
            "start": {"dateTime": start_dt.isoformat(), "timeZone": TIMEZONE},
            "end":   {"dateTime": end_dt.isoformat(),   "timeZone": TIMEZONE},
        })

        updated = service.events().update(
            calendarId="primary",
            eventId=event_id,
            body=existing,
        ).execute()

        link = updated.get("htmlLink", "")
        logger.info(f"📅 Calendar event updated: {event_id}")
        return f"✅ Calendar event updated. View: {link}"

    except RuntimeError as e:
        return f"❌ {e}"
    except Exception as e:
        logger.error(f"Calendar update error: {e}", exc_info=True)
        return f"❌ Calendar update failed: {e}"


def delete_calendar_event(event_id: str) -> str:
    """
    Delete a Google Calendar event by its event ID.
    Silently succeeds if the event is already gone (idempotent).
    """
    if not event_id:
        return "⚠️ No calendar event ID — nothing to delete."

    try:
        service = _get_calendar_service()
        service.events().delete(calendarId="primary", eventId=event_id).execute()
        logger.info(f"📅 Calendar event deleted: {event_id}")
        return f"✅ Calendar event {event_id} deleted."

    except RuntimeError as e:
        return f"❌ {e}"
    except Exception as e:
        # 410 Gone = already deleted — treat as success
        if "410" in str(e) or "404" in str(e):
            return f"✅ Event already removed from calendar."
        logger.error(f"Calendar delete error: {e}", exc_info=True)
        return f"❌ Calendar delete failed: {e}"


# ══════════════════════════════════════════════════════════════
# Gmail SMTP Email
# ══════════════════════════════════════════════════════════════

def send_email_to_doctor(
    doctor_email: str,
    doctor_name: str,
    subject: str,
    body: str,
) -> str:
    """
    Send an email to the doctor using Gmail SMTP + App Password.
    Uses Python's built-in smtplib — no pip package required.
    """
    if not GMAIL_SENDER_EMAIL or not GMAIL_APP_PASSWORD:
        return (
            "❌ Gmail not configured. Add GMAIL_SENDER_EMAIL and "
            "GMAIL_APP_PASSWORD to your .env file."
        )
    if not doctor_email:
        return "❌ Doctor email address not found in records."

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = f"Chronic Health Companion <{GMAIL_SENDER_EMAIL}>"
        msg["To"]      = doctor_email

        text_part = MIMEText(body, "plain", "utf-8")
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

        with smtplib.SMTP("smtp.gmail.com", 587, timeout=15) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(GMAIL_SENDER_EMAIL, GMAIL_APP_PASSWORD)
            server.sendmail(GMAIL_SENDER_EMAIL, [doctor_email], msg.as_string())

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


# ══════════════════════════════════════════════════════════════
# Instruction Parser
# ══════════════════════════════════════════════════════════════

ACTION_PARSE_PROMPT = """
Parse this action instruction and return JSON with:
  - "action_type": "calendar" | "email" | "both"
  - "doctor_name": string
  - "doctor_email": string (if available, else empty string)
  - "appointment_date": ISO 8601 datetime string (e.g. "2026-03-15T10:00:00")
  - "appointment_reason": string
  - "email_body": string (draft email body to the doctor, if action includes email)
  - "calendar_event_id": string (existing Google Calendar event ID if rescheduling, else empty)

Return JSON only, no markdown.

Instruction: {instruction}
""".strip()


def parse_action_instruction(instruction: str) -> dict:
    _FALLBACK = {
        "action_type": "calendar",
        "doctor_name": "Doctor",
        "doctor_email": "",
        "appointment_date": "",
        "appointment_reason": instruction,
        "email_body": "",
        "calendar_event_id": "",
    }
    prompt = ACTION_PARSE_PROMPT.format(instruction=instruction)
    raw = safe_llm_invoke(
        llm,
        [HumanMessage(content=prompt)],
        fallback=json.dumps(_FALLBACK),
    )
    try:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        # Find JSON block in case of extra text
        brace_s = cleaned.find("{")
        brace_e = cleaned.rfind("}")
        if brace_s != -1 and brace_e > brace_s:
            result = json.loads(cleaned[brace_s:brace_e+1])
            if isinstance(result, dict):
                return {**_FALLBACK, **result}   # merge so no key is ever missing
    except (json.JSONDecodeError, ValueError):
        pass
    logger.warning("Action parser returned non-JSON; using defaults")
    return _FALLBACK


# ══════════════════════════════════════════════════════════════
# Main Node
# ══════════════════════════════════════════════════════════════

def action_node(state: AgentState) -> dict:
    """
    LangGraph node – Action Agent.

    Parses the Orchestrator's instruction, executes calendar and/or
    email actions, and writes results to state.action_result.
    """
    logger.info("⚡ Action Agent activated")

    instruction = state["messages"][-1]
    raw_content = safe_content(instruction, "") if not isinstance(instruction.content, str) \
        else str(instruction.content)
    instruction_text = strip_agent_prefix(raw_content, "action")

    parsed           = parse_action_instruction(instruction_text)
    action_type      = parsed.get("action_type", "calendar")
    doctor_name      = parsed.get("doctor_name", "Unknown Doctor")
    doctor_email     = parsed.get("doctor_email", "")
    appointment_date = parsed.get("appointment_date", "")
    reason           = parsed.get("appointment_reason", "Medical appointment")
    email_body       = parsed.get(
        "email_body",
        f"Dear Dr. {doctor_name},\n\nI would like to request an appointment regarding {reason}.\n\nThank you.",
    )
    existing_event_id = parsed.get("calendar_event_id", "")

    results = []

    if action_type in ("calendar", "both"):
        if existing_event_id:
            # Rescheduling an existing appointment
            cal_result = update_calendar_event(
                event_id=existing_event_id,
                doctor_name=doctor_name,
                date_str=appointment_date,
                reason=reason,
            )
            results.append(cal_result)
        else:
            # New appointment
            cal_result, _ = create_calendar_event(
                doctor_name=doctor_name,
                date_str=appointment_date,
                reason=reason,
            )
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
