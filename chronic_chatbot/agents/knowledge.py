"""
agents/knowledge.py
───────────────────
Knowledge Agent – MCP client for web search, then LLM summarisation.
"""

import logging
from pathlib import Path

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient

from chronic_chatbot.config import GOOGLE_API_KEY, SUBAGENT_MODEL
from chronic_chatbot.state import AgentState
from chronic_chatbot.utils import safe_content, safe_llm_invoke, strip_agent_prefix

logger = logging.getLogger(__name__)

_SERVER_SCRIPT = str(
    Path(__file__).resolve().parent.parent / "mcp_server" / "knowledge_server.py"
)

llm = ChatGoogleGenerativeAI(
    model=SUBAGENT_MODEL,
    google_api_key=GOOGLE_API_KEY,
    temperature=0.1,
)

KNOWLEDGE_SYSTEM_PROMPT = """
You are a medical research assistant with raw web search results.
Your task:
  1. Extract only medically relevant, factual information.
  2. Summarise in 3-5 concise bullet points, jargon-free.
  3. Always end with: "⚠️ This is general information. Always consult your doctor."
  4. Do NOT invent information not present in the search results.

If no search results, give a brief helpful answer from general medical knowledge + disclaimer.
Output plain text only (no markdown headers, no JSON).
""".strip()

MCP_CONFIG = {
    "knowledge_server": {
        "command": "python",
        "args": [_SERVER_SCRIPT],
        "transport": "stdio",
    }
}


async def knowledge_node(state: AgentState) -> dict:
    """LangGraph node – Knowledge Agent (MCP client)."""
    logger.info("🔍 Knowledge Agent activated")

    raw_instruction = state["messages"][-1].content
    if not isinstance(raw_instruction, str):
        raw_instruction = safe_content(
            type("R", (), {"content": raw_instruction})(), ""
        )
    query = strip_agent_prefix(str(raw_instruction), "knowledge") or \
        "general chronic disease management advice"

    logger.info(f"🔍 Query: {query[:100]}")

    # ── MCP: run web search ───────────────────────────────────
    raw_context = ""
    try:
        client = MultiServerMCPClient(MCP_CONFIG)
        mcp_tools = await client.get_tools()

        search_tool = next((t for t in mcp_tools if t.name == "tavily_search"), None)
        if search_tool:
            result = await search_tool.ainvoke({"query": query})
            raw_context = (
                str(result.content) if hasattr(result, "content") else str(result)
            )
        else:
            logger.warning("tavily_search tool not found in knowledge_server")
    except Exception as e:
        logger.warning(f"Knowledge MCP error: {e}")

    # ── LLM summarisation ─────────────────────────────────────
    user_content = (
        f"Search query: {query}\n\nRaw search results:\n{raw_context[:4000]}"
        if raw_context else
        f"Search query: {query}\n\nNo web results. Answer from general medical knowledge."
    )

    summary = safe_llm_invoke(
        llm,
        [SystemMessage(content=KNOWLEDGE_SYSTEM_PROMPT), HumanMessage(content=user_content)],
        fallback=(
            "I couldn't retrieve medical information right now. "
            "Please try again, or consult your healthcare provider. "
            "⚠️ Always consult your doctor for medical advice."
        ),
    )

    logger.info("🔍 Knowledge Agent done")

    return {
        "messages":          [AIMessage(content=f"[Knowledge Agent Result]\n{summary}")],
        "knowledge_context":  summary,
    }
