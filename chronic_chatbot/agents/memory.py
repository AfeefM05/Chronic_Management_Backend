"""
agents/memory.py
────────────────
Memory Agent – MCP client that delegates all DB operations to memory_server.py.

Uses MultiServerMCPClient.get_tools() (the recommended short-form usage).
"""

import logging
from pathlib import Path

from langchain_core.messages import AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient

from chronic_chatbot.config import GOOGLE_API_KEY, SUBAGENT_MODEL
from chronic_chatbot.state import AgentState
from chronic_chatbot.utils import safe_content, strip_agent_prefix

logger = logging.getLogger(__name__)

_SERVER_SCRIPT = str(
    Path(__file__).resolve().parent.parent / "mcp_server" / "memory_server.py"
)

llm = ChatGoogleGenerativeAI(
    model=SUBAGENT_MODEL,
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
)

_SYSTEM_PROMPT = (
    "You are a Memory Agent. Read and write data to the patient's database.\n"
    "READ tools : search_symptoms, query_medications, query_doctors, query_appointments\n"
    "WRITE tools: log_symptom_dual, add_medication, add_doctor, add_appointment_record\n"
    "Choose and call the correct tool(s) based on the instruction immediately."
)

MCP_CONFIG = {
    "memory_server": {
        "command": "python",
        "args": [_SERVER_SCRIPT],
        "transport": "stdio",
    }
}


async def memory_node(state: AgentState) -> dict:
    """LangGraph node – Memory Agent (MCP client)."""
    logger.info("🗄️  Memory Agent activated")

    raw_msg = state["messages"][-1]
    instruction = (
        safe_content(raw_msg, "") if not isinstance(raw_msg.content, str)
        else str(raw_msg.content)
    )
    instruction_text = strip_agent_prefix(instruction, "memory")

    results: list[str] = []

    try:
        client = MultiServerMCPClient(MCP_CONFIG)
        # get_tools() starts ephemeral sessions, loads all tools, then closes sessions
        mcp_tools = await client.get_tools()

        if not mcp_tools:
            logger.error("No tools loaded from memory_server")
            results.append("⚠️ Memory server returned no tools.")
        else:
            llm_with_tools = llm.bind_tools(mcp_tools)

            ai_msg = await llm_with_tools.ainvoke([
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": instruction_text},
            ])

            if ai_msg.tool_calls:
                tool_map = {t.name: t for t in mcp_tools}
                for tc in ai_msg.tool_calls:
                    tool = tool_map.get(tc["name"])
                    if tool:
                        logger.info(f"🗄️  MCP tool: {tc['name']} args={tc['args']}")
                        tool_result = await tool.ainvoke(tc["args"])
                        results.append(
                            str(tool_result.content)
                            if hasattr(tool_result, "content")
                            else str(tool_result)
                        )
                    else:
                        results.append(f"⚠️ Tool '{tc['name']}' not found in memory_server.")
            else:
                results.append("⚠️ Memory Agent: LLM generated no tool calls.")

    except Exception as e:
        logger.error(f"Memory Agent error: {e}", exc_info=True)
        results.append(f"⚠️ Memory Agent error: {e}")

    combined = "\n".join(results)
    logger.info(f"🗄️  Memory result: {combined[:120]}")

    return {
        "messages":      [AIMessage(content=f"[Memory Agent Result]\n{combined}")],
        "memory_context": combined,
    }