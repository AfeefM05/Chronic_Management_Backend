"""
state.py – Shared LangGraph State definition

This TypedDict is the "brain" that flows through every node.
Every agent reads from and writes to this state.
"""

from __future__ import annotations

import operator
from typing import Annotated, List, Optional
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    # ── Conversation History ───────────────────────────────────
    # Using operator.add so messages from each node are appended,
    # not replaced, preserving the full conversation thread.
    messages: Annotated[List[BaseMessage], operator.add]

    # ── User Context ──────────────────────────────────────────
    # Cached profile: conditions, medications, doctors, allergies.
    # Loaded at startup from the SQLite DB and refreshed as needed.
    user_profile: dict

    # ── Routing ───────────────────────────────────────────────
    # The Orchestrator sets this to tell the router which agent
    # to delegate to next.
    # Values: "knowledge" | "memory" | "action" | "final_response"
    next_step: str

    # ── Execution Plan ────────────────────────────────────────
    # The Orchestrator decomposes complex requests into a list of
    # sub-tasks. Each sub-agent pops a task when it runs.
    current_plan: List[str]

    # ── Intermediate Results ──────────────────────────────────
    # Sub-agents write their outputs here so the Orchestrator can
    # synthesise them into a coherent final answer.
    knowledge_context: Optional[str]   # Knowledge Agent findings
    memory_context: Optional[str]      # Memory Agent retrieval
    action_result: Optional[str]       # Action Agent confirmation
