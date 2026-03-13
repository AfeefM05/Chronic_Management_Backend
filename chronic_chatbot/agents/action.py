"""
agents/action.py
────────────────
Action Agent – MCP client for calendar and email actions.
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
    Path(__file__).resolve().parent.parent / "mcp_server" / "action_server.py"
)

llm = ChatGoogleGenerativeAI(
    model=SUBAGENT_MODEL,
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
)

_SYSTEM_PROMPT = (
    "You are an Action Agent for a chronic disease management app.\n"
    "Execute calendar and email actions using the provided tools.\n"
    "Tools: create_calendar_event, update_calendar_event, delete_calendar_event, send_email_to_doctor\n"
    "Parse the instruction and call the correct tool(s) immediately. "
    "Use reasonable defaults for missing fields (e.g. tomorrow 10am if no time given)."
)

MCP_CONFIG = {
    "action_server": {
        "command": "python",
        "args": [_SERVER_SCRIPT],
        "transport": "stdio",
    }
}


async def action_node(state: AgentState) -> dict:
    """LangGraph node – Action Agent (MCP client)."""
    logger.info("⚡ Action Agent activated")

    raw_msg = state["messages"][-1]
    instruction = (
        safe_content(raw_msg, "") if not isinstance(raw_msg.content, str)
        else str(raw_msg.content)
    )
    instruction_text = strip_agent_prefix(instruction, "action")

    results: list[str] = []

    try:
        client = MultiServerMCPClient(MCP_CONFIG)
        mcp_tools = await client.get_tools()

        if not mcp_tools:
            results.append("⚠️ Action server returned no tools.")
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
                        logger.info(f"⚡ MCP tool: {tc['name']} args={tc['args']}")
                        tool_result = await tool.ainvoke(tc["args"])
                        results.append(
                            str(tool_result.content)
                            if hasattr(tool_result, "content")
                            else str(tool_result)
                        )
                    else:
                        results.append(f"⚠️ Tool '{tc['name']}' not found in action_server.")
            else:
                results.append(
                    "⚠️ Action Agent: LLM generated no tool calls. "
                    "Be more specific (e.g. 'Book appointment with Dr. X on March 20 at 10am')."
                )

    except Exception as e:
        logger.error(f"Action Agent error: {e}", exc_info=True)
        results.append(
            f"⚠️ Action failed: {e}. "
            "Ensure Google Calendar credentials are configured."
        )

    combined = "\n".join(results)
    logger.info(f"⚡ Action result: {combined[:120]}")

    return {
        "messages":     [AIMessage(content=f"[Action Agent Result]\n{combined}")],
        "action_result": combined,
    }