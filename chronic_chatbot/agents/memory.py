"""
agents/memory.py
────────────────
Memory Agent – The Librarian.

READ operations:
  • Semantic search of symptom history via ChromaDB (vector store)
  • Structured queries against SQLite (doctors, medications, appointments)

WRITE operations:
  • Log new symptoms to BOTH ChromaDB AND SQLite symptoms_log
  • Insert medications, appointments, doctors into SQLite

All errors are caught and returned as soft messages — never raises.
Uses shared utils for safe LLM calls and content extraction.

"""

import json
import logging
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from chronic_chatbot.config import (
    GOOGLE_API_KEY,
    SUBAGENT_MODEL,
)
from chronic_chatbot.state import AgentState
from chronic_chatbot.utils import safe_llm_invoke, safe_content, strip_agent_prefix

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════
# LLM — for free-form instruction parsing when prefix is absent
# ══════════════════════════════════════════════════════════════

llm = ChatGoogleGenerativeAI(
    model=SUBAGENT_MODEL,
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
)

# ══════════════════════════════════════════════════════════════
# Main Node
# ══════════════════════════════════════════════════════════════

async def memory_node(state: AgentState) -> dict:
    """LangGraph node – Memory Agent."""
    logger.info("🗄️  Memory Agent activated")

    # Extract instruction — safe_content handles list-of-parts responses
    raw_msg = state["messages"][-1]
    instruction = safe_content(raw_msg, "") if not isinstance(raw_msg.content, str) \
        else str(raw_msg.content)
    instruction_text = strip_agent_prefix(instruction, "memory")

    # 1. MCP client connection setup
    client = MultiServerMCPClient({
        "memory_server": {
            "command": "python",
            "args": ["mcp_servers/memory_server.py"],
            "transport": "stdio"
        }
    })
    
    results=[]
    
    async with client.session("memory_server") as session:
        # 2. Load tools from the MCP server
        mcp_tools = await load_mcp_tools(session)
        
        # 3. Bind the tools to Gemini
        llm_with_tools = llm.bind_tools(mcp_tools)
        
        system_prompt = (
            "You are a Memory Agent. Your job is to read and write data to the patient's database.\n"
            "Use the provided tools to fulfill the user's request. Always execute the necessary tools.\n"
            "If asked to 'read', use search_symptoms, query_medications, query_doctors, or query_appointments.\n"
            "If asked to 'write' or 'log', use log_symptom_dual, add_medication, add_doctor, or add_appointment_record."
        )
        
        # 4. Ask Gemini which tools to use based on the Orchestrator's instruction
        ai_msg = await llm_with_tools.ainvoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instruction_text}
        ])

        # 5. Execute the tools Gemini decided to call
        if ai_msg.tool_calls:
            for tool_call in ai_msg.tool_calls:
                # Find the corresponding tool object
                tool = next((t for t in mcp_tools if t.name == tool_call["name"]), None)
                
                if tool:
                    # Execute the tool asynchronously
                    logger.info(f"Executing MCP Tool: {tool.name}")
                    tool_result = await tool.ainvoke(tool_call["args"])
                    
                    # Extract string content from tool_result (handles Langchain's ToolMessage wrapper)
                    if hasattr(tool_result, 'content'):
                        results.append(str(tool_result.content))
                    else:
                        results.append(str(tool_result))
                else:
                    results.append(f"⚠️ Tool {tool_call['name']} not found.")
        else:
            results.append("⚠️ Memory Agent didn't recognize any database actions to take.")

# 6. Format the final output
    combined_result = "\n".join(results)
    logger.info(f"🗄️ Memory result: {combined_result[:120]}...")

    final_msg = AIMessage(content=f"[Memory Agent Result]\n{combined_result}")

    return {
        "messages": [final_msg],
        "memory_context": combined_result,
    }