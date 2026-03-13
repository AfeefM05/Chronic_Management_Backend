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
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from chronic_chatbot.config import (
    GOOGLE_API_KEY,
    SUBAGENT_MODEL
)
from chronic_chatbot.state import AgentState
from chronic_chatbot.utils import safe_content, strip_agent_prefix

logger = logging.getLogger(__name__)

llm = ChatGoogleGenerativeAI(
    model=SUBAGENT_MODEL,
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
)

# ══════════════════════════════════════════════════════════════
# Main Node
# ══════════════════════════════════════════════════════════════

async def action_node(state: AgentState) -> dict:
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
    
    # 1. MCP client connection setup
    client= MultiServerMCPClient({
        "action_server": {
            "command": "python",
            "args": ["mcp_servers/action_server.py"],
        }
    })
    
    results = []
    
    async with client.session("action_server") as session:
        # 2. Load tools from the MCP server
        mcp_tools = await load_mcp_tools(session)
        
        # 3. Bind the tools to Gemini
        llm_with_tools = llm.bind_tools(mcp_tools)
        
        # 4. Ask Gemini which tools to use based on the Orchestrator's instruction
        ai_msg = await llm_with_tools.ainvoke([
            {"role": "system", "content": "You are an Action Agent. Use the provided tools to fulfill the user's request. Always execute the necessary tools."},
            {"role": "user", "content": instruction_text}
        ])

        # 5. Execute the tools Gemini decided to call
        if ai_msg.tool_calls:
            for tool_call in ai_msg.tool_calls:
                # Find the corresponding tool object
                tool = next(t for t in mcp_tools if t.name == tool_call["name"])
                
                # Execute the tool asynchronously
                logger.info(f"Executing MCP Tool: {tool.name}")
                tool_result = await tool.ainvoke(tool_call["args"])
                results.append(str(tool_result))
        else:
            results.append("⚠️ No recognizable calendar or email actions were extracted by the LLM.")

    # 6. Format the final output
    combined_result = "\n".join(results)
    final_msg = AIMessage(content=f"[Action Agent Result]\n{combined_result}")

    return {
        "messages": [final_msg],
        "action_result": combined_result,
    }