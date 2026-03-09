"""
graph.py – LangGraph Workflow Builder

Assembles the four agent nodes into a cyclic state machine:
  
  [START] → orchestrator → router → knowledge_agent ─┐
                         ↑         memory_agent  ─────┤
                         └─────────action_agent  ─────┘
                         └──── (final_response) → [END]
"""

import logging
from langgraph.graph import StateGraph, END

from chronic_chatbot.state import AgentState
from chronic_chatbot.agents import (
    orchestrator_node,
    knowledge_node,
    memory_node,
    action_node,
)

logger = logging.getLogger(__name__)


def router(state: AgentState) -> str:
    """
    Conditional edge function.
    Reads state.next_step set by the Orchestrator and routes to
    the appropriate agent node or END.
    """
    step = state.get("next_step", "final_response")
    logger.debug(f"Router → {step}")

    routing_map = {
        "knowledge": "knowledge_agent",
        "memory": "memory_agent",
        "action": "action_agent",
        "final_response": END,
    }
    return routing_map.get(step, END)


def build_graph() -> "CompiledGraph":
    """
    Construct and compile the LangGraph workflow.

    Returns the compiled app ready for `app.invoke(state)` calls.
    """
    workflow = StateGraph(AgentState)

    # ── Register Nodes ─────────────────────────────────────────
    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("knowledge_agent", knowledge_node)
    workflow.add_node("memory_agent", memory_node)
    workflow.add_node("action_agent", action_node)

    # ── Entry Point ────────────────────────────────────────────
    workflow.set_entry_point("orchestrator")

    # ── Conditional Routing from Orchestrator ──────────────────
    workflow.add_conditional_edges(
        "orchestrator",
        router,
        {
            "knowledge_agent": "knowledge_agent",
            "memory_agent": "memory_agent",
            "action_agent": "action_agent",
            END: END,
        },
    )

    # ── Return Edges – Sub-agents always report back ───────────
    workflow.add_edge("knowledge_agent", "orchestrator")
    workflow.add_edge("memory_agent", "orchestrator")
    workflow.add_edge("action_agent", "orchestrator")

    app = workflow.compile()
    logger.info("✅ LangGraph workflow compiled successfully")
    return app


# Module-level singleton – import this in main.py
graph_app = build_graph()
