"""
utils.py – Shared helpers for all agents

Provides:
  • safe_content(response) — extract string from LLM response, handles both
    plain-string and list-of-parts content (newer langchain-google-genai versions)
  • safe_llm_invoke(llm, messages, fallback) — LLM call with retry + network fallback
  • strip_agent_prefix(text) — remove "[Orchestrator → X]" prefixes from instructions
"""

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


# ── Content normaliser ────────────────────────────────────────────────────────
def safe_content(response: Any, fallback: str = "") -> str:
    """
    Extract a plain string from an LLM response.

    langchain-google-genai ≥ 0.3 sometimes returns `content` as a list of
    content-part dicts:  [{"type": "text", "text": "..."}, ...]
    Older versions return a plain string. Handle both.
    """
    if response is None:
        return fallback

    content = getattr(response, "content", response)

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                # {"type": "text", "text": "..."} or {"type": "text", "content": "..."}
                parts.append(part.get("text") or part.get("content") or str(part))
            else:
                parts.append(str(part))
        return " ".join(p for p in parts if p).strip()

    return str(content).strip() if content else fallback


# ── Safe LLM invoke with retry ────────────────────────────────────────────────
def safe_llm_invoke(
    llm: Any,
    messages: list,
    fallback: str = "I'm sorry, I'm having trouble connecting right now. Please try again.",
    retries: int = 2,
    delay: float = 1.5,
) -> str:
    """
    Invoke an LLM with automatic retry on transient network/API errors.
    Returns the response as a plain string (never raises).
    """
    last_error = None

    for attempt in range(1, retries + 1):
        try:
            response = llm.invoke(messages)
            result   = safe_content(response, fallback)
            if result:
                return result
        except Exception as e:
            last_error = e
            err_str    = str(e)

            # Permanent errors — don't retry
            if any(k in err_str for k in ("API_KEY", "invalid", "PERMISSION_DENIED", "400")):
                logger.error(f"LLM permanent error: {e}")
                return fallback

            # Transient — retry
            logger.warning(f"LLM attempt {attempt}/{retries} failed: {e}")
            if attempt < retries:
                time.sleep(delay)

    logger.error(f"LLM failed after {retries} retries: {last_error}")
    return fallback


# ── Prefix stripper ───────────────────────────────────────────────────────────
def strip_agent_prefix(text: str, agent_name: str) -> str:
    """
    Remove the routing prefix the Orchestrator adds, e.g.:
      "[Orchestrator → memory] WRITE symptom: headache"
      → "WRITE symptom: headache"
    """
    markers = [
        f"[Orchestrator → {agent_name}]",
        f"[Orchestrator -> {agent_name}]",
        f"[Orchestrator → {agent_name}]",   # different arrow char
    ]
    for marker in markers:
        if marker in text:
            return text.split("]", 1)[-1].strip()
    return text.strip()
