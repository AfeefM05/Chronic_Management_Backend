"""
tests/test_graph.py
───────────────────
Unit & integration tests for the 4-agent LangGraph workflow.

Run with:  venv/bin/pytest tests/ -v
"""

import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage

from chronic_chatbot.state import AgentState


# ── Helpers ───────────────────────────────────────────────────

def make_state(**overrides) -> AgentState:
    """Return a minimal valid AgentState for testing."""
    base: AgentState = {
        "messages": [],
        "user_profile": {"condition": "Type 2 Diabetes"},
        "next_step": "",
        "current_plan": [],
        "knowledge_context": None,
        "memory_context": None,
        "action_result": None,
    }
    base.update(overrides)
    return base


# ── State Tests ───────────────────────────────────────────────

class TestAgentState:
    def test_state_has_required_keys(self):
        state = make_state()
        required = ["messages", "user_profile", "next_step", "current_plan"]
        for key in required:
            assert key in state

    def test_messages_append_behaviour(self):
        """operator.add should concatenate lists."""
        import operator
        a = [HumanMessage(content="Hello")]
        b = [AIMessage(content="Hi")]
        merged = operator.add(a, b)
        assert len(merged) == 2


# ── Router Tests ──────────────────────────────────────────────

class TestRouter:
    def test_routes_to_knowledge(self):
        from chronic_chatbot.graph import router
        state = make_state(next_step="knowledge")
        assert router(state) == "knowledge_agent"

    def test_routes_to_memory(self):
        from chronic_chatbot.graph import router
        state = make_state(next_step="memory")
        assert router(state) == "memory_agent"

    def test_routes_to_action(self):
        from chronic_chatbot.graph import router
        state = make_state(next_step="action")
        assert router(state) == "action_agent"

    def test_routes_to_end_on_final_response(self):
        from langgraph.graph import END
        from chronic_chatbot.graph import router
        state = make_state(next_step="final_response")
        assert router(state) == END

    def test_routes_to_end_on_unknown(self):
        from langgraph.graph import END
        from chronic_chatbot.graph import router
        state = make_state(next_step="nonexistent")
        assert router(state) == END


# ── Memory Agent Tests ────────────────────────────────────────

class TestMemoryAgent:
    def test_search_symptoms_returns_string(self):
        from chronic_chatbot.agents.memory import search_symptoms
        result = search_symptoms("dizziness")
        assert isinstance(result, str)

    def test_log_and_retrieve_symptom(self):
        from chronic_chatbot.agents.memory import log_symptom, search_symptoms
        log_result = log_symptom("Test symptom: mild headache after medication")
        assert "logged" in log_result.lower()
        search_result = search_symptoms("headache")
        assert isinstance(search_result, str)

    def test_query_doctors_empty(self):
        from chronic_chatbot.agents.memory import query_doctors
        result = query_doctors("NonExistentDoctor12345")
        assert isinstance(result, str)


# ── Integration Test (requires API keys) ─────────────────────

@pytest.mark.integration
class TestGraphIntegration:
    """
    Skipped in CI unless marked --integration.
    Requires real API keys in .env
    """

    def test_simple_question_completes(self):
        """End-to-end: a simple health question should return a string reply."""
        from chronic_chatbot.graph import graph_app

        state = make_state(messages=[HumanMessage(content="What is Type 2 Diabetes?")])
        result = graph_app.invoke(state)
        assert "messages" in result
        assert len(result["messages"]) > 1
        # Final message should contain text (not empty)
        assert result["messages"][-1].content.strip()
