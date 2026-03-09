"""
agents/orchestrator.py
──────────────────────
Main Agent (Orchestrator) – The Planner & Synthesiser.
"""

import json
import logging
import re
from langchain_core.messages import AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from chronic_chatbot.config import GOOGLE_API_KEY, ORCHESTRATOR_MODEL
from chronic_chatbot.state import AgentState
from chronic_chatbot.utils import safe_content, safe_llm_invoke

logger = logging.getLogger(__name__)

llm = ChatGoogleGenerativeAI(
    model=ORCHESTRATOR_MODEL,
    google_api_key=GOOGLE_API_KEY,
    temperature=0.1,
)

# ── System Prompt ──────────────────────────────────────────────
ORCHESTRATOR_SYSTEM_PROMPT = """
You are a compassionate, intelligent AI health manager for a patient with chronic disease(s).

You have three specialist sub-agents:
  1. memory_agent    – Reads/WRITES symptoms, medications, appointments, doctors to the database.
  2. knowledge_agent – Searches the web for medical information.
  3. action_agent    – Books Google Calendar appointments and sends emails.

════════════════════════════════════════════
CRITICAL DECISION RULES:
════════════════════════════════════════════

[SYMPTOM REPORTING — user says they are EXPERIENCING a symptom]
  Triggers: "I have", "I feel", "I'm feeling", "I've been experiencing",
            "I noticed", "suffering from", "hurts", "pain", "dizzy", "fatigue", "headache"
  → Step 1: next_step=memory  |  instruction="WRITE symptom: <exact symptom text>"
  → Step 2: next_step=knowledge  |  instruction="Causes and advice for <symptom> in chronic patient"
  → Step 3: next_step=final_response  |  reply includes "I've logged this" + knowledge advice

[SYMPTOM HISTORY QUERY]
  Triggers: "have I had", "did I log", "my symptoms", "symptom history"
  → next_step=memory  |  instruction="READ symptoms: <query>"
  → next_step=final_response  |  report findings

[MEDICATION REPORTING — user mentions starting or taking a new medication]
  Triggers: "I started taking", "I take", "I'm on", "prescribed me", "my new medication"
  → next_step=memory  |  instruction="WRITE medication: name=<name>, dose=<dose>, frequency=<freq>"
  → next_step=final_response  |  confirm logged

[MEDICATION QUERY]
  Triggers: "my medications", "what am I taking", "medication list", "review my meds"
  → next_step=memory  |  instruction="READ medications"
  → next_step=final_response

[MEDICAL/KNOWLEDGE QUESTION]
  Triggers: "what is", "tell me about", "side effects of", "how to manage", "diet for"
  → next_step=knowledge  |  instruction="<the question>"
  → next_step=final_response

[APPOINTMENT BOOKING]
  Triggers: "book", "schedule", "appointment with", "remind me", "email doctor"
  → next_step=action  |  instruction="Book appointment with <doctor> on <date> for <reason>"
  → next_step=final_response

[DOCTOR LOOKUP]
  Triggers: "my doctors", "find a doctor"
  → next_step=memory  |  instruction="READ doctors"
  → next_step=final_response

[GENERAL CHAT]
  → next_step=final_response immediately

════════════════════════════════════════════
INSTRUCTION PREFIXES (use EXACTLY):
════════════════════════════════════════════
  memory writes : "WRITE symptom: ..."  |  "WRITE medication: ..."  |  "WRITE doctor: ..."
  memory reads  : "READ symptoms: ..."  |  "READ medications"  |  "READ doctors"
  knowledge     : plain natural-language question
  action        : "Book appointment with ..."

════════════════════════════════════════════
SEQUENCING RULES:
════════════════════════════════════════════
  - Call ONE agent per step. Wait for its result before deciding the next step.
  - If memory_context contains "logged" or "added" → write succeeded, do NOT ask the user again.
  - NEVER ask the user for more information if they have already provided it.
  - If they said "I have a headache" → LOG IT, give advice. Do not reply "What symptoms?".

Output ONLY valid JSON (no markdown, no extra text):
{
  "reasoning": "<one sentence: what intent detected>",
  "next_step": "<knowledge|memory|action|final_response>",
  "instruction_for_agent": "<structured instruction, or the full final reply>"
}
""".strip()


def _extract_json(raw: str) -> dict | None:
    """
    Attempt to extract a JSON object from the model's raw output.
    Handles: plain JSON, markdown-fenced JSON, prefix-then-JSON,
             and the model outputting a routing prefix instead of JSON.
    """
    # 1. Strip markdown fences
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        parts   = cleaned.split("```")
        cleaned = parts[1] if len(parts) > 1 else cleaned
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()

    # 2. Direct parse
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        pass

    # 3. Find first { … } block (handles prefix lines before the JSON)
    brace_start = cleaned.find("{")
    brace_end   = cleaned.rfind("}")
    if brace_start != -1 and brace_end > brace_start:
        try:
            return json.loads(cleaned[brace_start : brace_end + 1])
        except (json.JSONDecodeError, ValueError):
            pass

    # 4. Model skipped JSON entirely and output a routing prefix line
    #    e.g.  "[Orchestrator → knowledge] What is headache?"
    #    Use alternation (?:→|->) NOT a character class [→->] to avoid regex range error
    route_match = re.search(
        r"\[Orchestrator\s*(?:→|->)\s*(\w+)\]\s*(.+)",
        cleaned,
        re.DOTALL,
    )
    if route_match:
        return {
            "reasoning": "Model skipped JSON; parsed routing prefix.",
            "next_step": route_match.group(1).strip().lower(),
            "instruction_for_agent": route_match.group(2).strip(),
        }

    return None


def orchestrator_node(state: AgentState) -> dict:
    """LangGraph node – Orchestrator."""
    logger.info("🧠 Orchestrator activated")

    messages      = state["messages"]
    user_profile  = state.get("user_profile", {})
    knowledge_ctx = state.get("knowledge_context") or ""
    memory_ctx    = state.get("memory_context")    or ""
    action_result = state.get("action_result")     or ""

    context_block = (
        f"Current User Profile: {json.dumps(user_profile)}\n\n"
        f"Sub-agent results this turn:\n"
        f"  Memory   : {memory_ctx   or 'None yet'}\n"
        f"  Knowledge: {knowledge_ctx or 'None yet'}\n"
        f"  Action   : {action_result or 'None yet'}\n\n"
        f"NOTE: If Memory says 'logged' or 'added', write succeeded — move on."
    )

    system_msg = SystemMessage(content=ORCHESTRATOR_SYSTEM_PROMPT + "\n\n" + context_block)
    full_msgs  = [system_msg] + list(messages)

    # ── LLM call with safe fallback ───────────────────────────
    raw = safe_llm_invoke(
        llm,
        full_msgs,
        fallback='{"reasoning":"network error","next_step":"final_response",'
                 '"instruction_for_agent":"I\'m having trouble connecting right now. Please try again in a moment."}',
    )

    # ── Parse decision ────────────────────────────────────────
    decision = _extract_json(raw)

    if decision is None:
        logger.warning(f"Orchestrator total parse failure: {raw[:200]}")
        decision = {
            "reasoning": "Unparseable response — default to final answer.",
            "next_step": "final_response",
            "instruction_for_agent": raw,
        }

    next_step   = decision.get("next_step", "final_response")
    instruction = decision.get("instruction_for_agent", "")
    reasoning   = decision.get("reasoning", "")

    # Validate next_step
    if next_step not in ("knowledge", "memory", "action", "final_response"):
        logger.warning(f"Unknown next_step '{next_step}' — defaulting to final_response")
        next_step = "final_response"

    logger.info(f"🧠 Orchestrator → {next_step} | {reasoning[:80]}")

    ai_msg = AIMessage(
        content=(
            instruction
            if next_step == "final_response"
            else f"[Orchestrator → {next_step}] {instruction}"
        )
    )

    return {
        "messages":  [ai_msg],
        "next_step": next_step,
    }
