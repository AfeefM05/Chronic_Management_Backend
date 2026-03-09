"""
agents/knowledge.py
───────────────────
Knowledge Agent – The Web Researcher.

Responsibilities:
  • Receive a medical topic / query from the Orchestrator
  • Use Tavily to perform a targeted, reliable web search
  • Summarise results into a concise medical context snippet
  • Return findings via state.knowledge_context

All errors (network, Tavily quota, LLM) are caught and result in
a soft graceful fallback — never propagates exceptions upward.
"""

import logging
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from chronic_chatbot.config import GOOGLE_API_KEY, SUBAGENT_MODEL, TAVILY_API_KEY
from chronic_chatbot.state import AgentState
from chronic_chatbot.utils import safe_content, safe_llm_invoke, strip_agent_prefix

logger = logging.getLogger(__name__)

# ── Clients ───────────────────────────────────────────────────
llm = ChatGoogleGenerativeAI(
    model=SUBAGENT_MODEL,
    google_api_key=GOOGLE_API_KEY,
    temperature=0.1,
)

# Tavily client — lazy-initialised so import errors don't crash startup
_tavily = None

def _get_tavily():
    global _tavily
    if _tavily is None:
        try:
            from tavily import TavilyClient
            _tavily = TavilyClient(api_key=TAVILY_API_KEY)
        except Exception as e:
            logger.error(f"Tavily init failed: {e}")
    return _tavily


KNOWLEDGE_SYSTEM_PROMPT = """
You are a medical research assistant. You have been given raw web search results.
Your task:
  1. Extract only medically relevant, factual information.
  2. Summarise in 3-5 concise bullet points, jargon-free.
  3. Always end with: "⚠️ This is general information. Always consult your doctor."
  4. Do NOT invent information not in the search results.

If no search results are available, give a brief, helpful answer from general medical knowledge
and add the disclaimer. Do not say you cannot help.

Output plain text only (no markdown headers, no JSON).
""".strip()


def _tavily_search(query: str) -> str:
    """
    Run a Tavily search and return a raw context string.
    Returns an empty string on any error (caller handles fallback).
    """
    client = _get_tavily()
    if client is None:
        return ""
    try:
        resp = client.search(
            query=query,
            search_depth="advanced",
            max_results=4,
            include_answer=True,
        )
        # Normalise: newer SDK returns object, older returns dict
        if isinstance(resp, dict):
            answer  = resp.get("answer") or ""
            results = resp.get("results") or []
        else:
            answer  = getattr(resp, "answer",  None) or ""
            results = getattr(resp, "results", None) or []

        parts = [str(answer)] if answer else []
        for r in results[:4]:
            if isinstance(r, dict):
                parts.append(f"Source: {r.get('url','')}\n{r.get('content','')}")
            else:
                parts.append(f"Source: {getattr(r,'url','')}\n{getattr(r,'content','')}")

        return "\n\n".join(parts)

    except Exception as e:
        logger.warning(f"Tavily search failed: {e}")
        return ""


def knowledge_node(state: AgentState) -> dict:
    """LangGraph node – Knowledge Agent."""
    logger.info("🔍 Knowledge Agent activated")

    # Extract query from last message
    raw_instruction = state["messages"][-1].content
    # safe_content handles list-of-parts responses
    if not isinstance(raw_instruction, str):
        raw_instruction = safe_content(type("R", (), {"content": raw_instruction})(), "")

    query = strip_agent_prefix(str(raw_instruction), "knowledge")
    if not query:
        query = "general chronic disease management advice"

    logger.info(f"🔍 Query: {query[:100]}")

    # ── Web search (best-effort) ──────────────────────────────
    raw_context = _tavily_search(query)

    if raw_context:
        user_content = f"Search query: {query}\n\nRaw search results:\n{raw_context[:4000]}"
    else:
        # No web results — ask LLM from general knowledge
        user_content = (
            f"Search query: {query}\n\n"
            "No web search results available. Answer from your general medical knowledge."
        )

    # ── LLM summarisation ─────────────────────────────────────
    system_msg = SystemMessage(content=KNOWLEDGE_SYSTEM_PROMPT)
    human_msg  = HumanMessage(content=user_content)

    summary = safe_llm_invoke(
        llm,
        [system_msg, human_msg],
        fallback=(
            "I couldn't retrieve medical information right now due to a connectivity issue. "
            "Please try again in a moment, or consult your healthcare provider directly. "
            "⚠️ Always consult your doctor for medical advice."
        ),
    )

    logger.info("🔍 Knowledge Agent done")

    ai_msg = AIMessage(content=f"[Knowledge Agent Result]\n{summary}")
    return {
        "messages":         [ai_msg],
        "knowledge_context": summary,
    }
