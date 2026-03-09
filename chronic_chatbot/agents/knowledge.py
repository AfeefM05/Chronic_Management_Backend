"""
agents/knowledge.py
───────────────────
Knowledge Agent – The Web Researcher.

Responsibilities:
  • Receive a medical topic / query from the Orchestrator.
  • Use Tavily to perform a targeted, reliable web search.
  • Summarise results into a "Medical Context" snippet.
  • Return findings to the Orchestrator via state.knowledge_context.

Uses Gemini 1.5 Flash (fast & cheap) because it is a sub-agent
that performs a focused, bounded task.
"""

import logging
from langchain_core.messages import AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import TavilyClient

from chronic_chatbot.config import GOOGLE_API_KEY, SUBAGENT_MODEL, TAVILY_API_KEY
from chronic_chatbot.state import AgentState

logger = logging.getLogger(__name__)

# ── Clients ───────────────────────────────────────────────────
llm = ChatGoogleGenerativeAI(
    model=SUBAGENT_MODEL,
    google_api_key=GOOGLE_API_KEY,
    temperature=0.1,
)

tavily = TavilyClient(api_key=TAVILY_API_KEY)

# ── System Prompt ─────────────────────────────────────────────
KNOWLEDGE_SYSTEM_PROMPT = """
You are a medical research assistant. You have been given raw web search results.
Your task is to:
  1. Extract only the medically relevant, factual information.
  2. Summarise it in 3-5 bullet points – clear, concise, and jargon-free.
  3. Always include a disclaimer: "This is general information. Consult your doctor."
  4. DO NOT make up or infer information not present in the search results.
Output plain text (no markdown headers).
""".strip()


def knowledge_node(state: AgentState) -> dict:
    """
    LangGraph node – Knowledge Agent.

    Reads the last orchestrator instruction from messages,
    searches the web, summarises findings, and writes to
    state.knowledge_context.
    """
    logger.info("🔍 Knowledge Agent activated")

    # The orchestrator's instruction is the last message
    instruction = state["messages"][-1].content
    # Strip the routing prefix if present
    if "[Orchestrator → knowledge]" in instruction:
        query = instruction.split("]", 1)[-1].strip()
    else:
        query = instruction

    logger.info(f"🔍 Searching for: {query}")

    # ── Tavily Search ─────────────────────────────────────────
    try:
        search_response = tavily.search(
            query=query,
            search_depth="advanced",
            max_results=5,
            include_answer=True,
        )
        # Normalise: newer Tavily SDK returns a SearchResponse object, older returns a dict
        if isinstance(search_response, dict):
            resp = search_response
        else:
            resp = {
                "answer":  getattr(search_response, "answer", "") or "",
                "results": getattr(search_response, "results", []) or [],
            }

        raw_context = (resp.get("answer") or "") + "\n\n"
        for r in resp.get("results", []):
            if isinstance(r, dict):
                raw_context += f"Source: {r.get('url', '')}\n{r.get('content', '')}\n\n"
            else:
                raw_context += f"Source: {getattr(r, 'url', '')}\n{getattr(r, 'content', '')}\n\n"
    except Exception as e:
        logger.error(f"Tavily search failed: {e}")
        raw_context = f"Search failed: {e}"

    # ── LLM Summarisation ─────────────────────────────────────
    system_msg = SystemMessage(content=KNOWLEDGE_SYSTEM_PROMPT)
    user_msg_content = f"Search query: {query}\n\nRaw results:\n{raw_context}"

    from langchain_core.messages import HumanMessage
    summary_response = llm.invoke([system_msg, HumanMessage(content=user_msg_content)])
    summary = summary_response.content.strip()

    logger.info("🔍 Knowledge Agent completed search and summary")

    ai_msg = AIMessage(content=f"[Knowledge Agent Result]\n{summary}")

    return {
        "messages": [ai_msg],
        "knowledge_context": summary,
    }
