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
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from chronic_chatbot.config import GOOGLE_API_KEY, SUBAGENT_MODEL
from chronic_chatbot.state import AgentState
from chronic_chatbot.utils import safe_content, safe_llm_invoke, strip_agent_prefix

logger = logging.getLogger(__name__)

# ── Clients ───────────────────────────────────────────────────
llm = ChatGoogleGenerativeAI(
    model=SUBAGENT_MODEL,
    google_api_key=GOOGLE_API_KEY,
    temperature=0.1,
)

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

async def knowledge_node(state: AgentState) -> dict:
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
    
    client = MultiServerMCPClient({
        "knowledge_server": {
            "command": "python",
            "args": ["mcp_servers/knowledge_server.py"],
            "transport": "stdio"
        }
    })
    
    raw_context = ""

    async with client.session("knowledge_server") as session:
        # Load tools from the MCP server
        mcp_tools = await load_mcp_tools(session)
        
        search_tool = next((t for t in mcp_tools if t.name == "tavily_search"), None)
        
        if search_tool:
            raw_context = await search_tool.ainvoke({"query": query})
            if hasattr(raw_context, "content"):
                raw_context = raw_context.content
        else:
            logger.error("Search tool not found in MCP server")

    # ── Construct Payload ────────────────────────────────────────
    if raw_context:
        user_content = f"Search query: {query}\n\nRaw search results:\n{str(raw_context)[:4000]}"
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
