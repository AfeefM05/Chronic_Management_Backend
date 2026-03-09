"""
agents/orchestrator.py
──────────────────────
Main Agent (Orchestrator) – The Planner.

Responsibilities:
  • Receive and understand user input
  • Detect intents (symptom report / query / action request)
  • Decide which sub-agent(s) to delegate to via next_step
  • Synthesise sub-agent results into a final user-facing response

This agent uses Gemini 1.5 Pro (high-reasoning model) because
it makes the most critical decisions in the graph.
"""

import json
import logging
from langchain_core.messages import AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from chronic_chatbot.config import GOOGLE_API_KEY, ORCHESTRATOR_MODEL, MAX_AGENT_ITERATIONS
from chronic_chatbot.state import AgentState

logger = logging.getLogger(__name__)

# ── LLM Setup ─────────────────────────────────────────────────
llm = ChatGoogleGenerativeAI(
    model=ORCHESTRATOR_MODEL,
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2,  # Low temp for deterministic routing decisions
)

# ── System Prompt ─────────────────────────────────────────────
ORCHESTRATOR_SYSTEM_PROMPT = """
You are a compassionate and intelligent AI health manager for a patient with chronic disease(s).

You have access to three specialist agents:
  1. knowledge_agent  – Searches the web for reliable medical information.
  2. memory_agent     – Reads/writes symptom history, medication lists, and doctor details from the database.
  3. action_agent     – Books appointments (Google Calendar) and sends emails to doctors.

Your Job:
  - Analyse the user's LATEST message in the context of the full conversation.
  - Determine which agent(s) to call and in what sequence.
  - Once you have all the information you need, produce a final, empathetic response.

Decision Rules:
  - Symptom mentions / "last time" / history check → call memory_agent first
  - Medical questions / drug side effects / condition info → call knowledge_agent
  - "Book", "schedule", "remind", "email doctor" → call action_agent
  - If you have everything you need → set next_step to "final_response"

Output Format (JSON only, no markdown):
{
  "reasoning": "<brief internal thought>",
  "next_step": "<knowledge|memory|action|final_response>",
  "instruction_for_agent": "<what the sub-agent should do, or the final answer if next_step=final_response>"
}
""".strip()


def orchestrator_node(state: AgentState) -> dict:
    """
    LangGraph node – Orchestrator.

    Reads the current state, decides routing, and returns:
      - next_step: routing key for the conditional edge
      - messages:  the LLM's reasoning/response appended
      - (optionally) knowledge_context / memory_context / action_result reset
    """
    logger.info("🧠 Orchestrator activated")

    messages = state["messages"]
    user_profile = state.get("user_profile", {})
    knowledge_ctx = state.get("knowledge_context", "")
    memory_ctx = state.get("memory_context", "")
    action_result = state.get("action_result", "")

    # Build the context block injected into the system prompt
    context_block = f"""
Current User Profile:
{json.dumps(user_profile, indent=2)}

Sub-agent results available:
  Knowledge context : {knowledge_ctx or 'None yet'}
  Memory context    : {memory_ctx or 'None yet'}
  Action result     : {action_result or 'None yet'}
""".strip()

    system_msg = SystemMessage(content=ORCHESTRATOR_SYSTEM_PROMPT + "\n\n" + context_block)
    full_messages = [system_msg] + messages

    response = llm.invoke(full_messages)
    raw = response.content.strip()

    # ── Parse JSON decision ───────────────────────────────────
    try:
        # Strip potential markdown fences if model adds them
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        decision = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Orchestrator returned non-JSON; defaulting to final_response")
        decision = {
            "reasoning": "Parse error – defaulting to final answer.",
            "next_step": "final_response",
            "instruction_for_agent": raw,
        }

    next_step = decision.get("next_step", "final_response")
    instruction = decision.get("instruction_for_agent", "")

    logger.info(f"🧠 Orchestrator → next_step={next_step}")

    # Append the decision as an AI message so it persists in history
    ai_msg = AIMessage(
        content=instruction if next_step == "final_response" else f"[Orchestrator → {next_step}] {instruction}"
    )

    return {
        "messages": [ai_msg],
        "next_step": next_step,
        # Pass the instruction as the last message for sub-agents to read
    }
