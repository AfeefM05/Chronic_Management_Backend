"""
mcp_server/knowledge.py
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
from chronic_chatbot.config import TAVILY_API_KEY
from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

mcp = FastMCP("ChronicKnowledgeServer")

# Tavily client — lazy-initialised so import errors don't crash startup
_tavily = None

def _get_tavily():
    global _tavily
    if _tavily is None:
        try:
            from tavily import TavilyClient
            _tavily = TavilyClient(api_key=TAVILY_API_KEY)
        except Exception as e:
            logger.error(f"Tavily init failed: {e}")
    return _tavily

@mcp.tool()
def tavily_search(query: str) -> str:
    """
    Run a Tavily search and return a raw context string.
    Returns an empty string on any error (caller handles fallback).
    """
    client = _get_tavily()
    if client is None:
        return ""
    try:
        resp = client.search(
            query=query,
            search_depth="advanced",
            max_results=4,
            include_answer=True,
        )
        # Normalise: newer SDK returns object, older returns dict
        if isinstance(resp, dict):
            answer  = resp.get("answer") or ""
            results = resp.get("results") or []
        else:
            answer  = getattr(resp, "answer",  None) or ""
            results = getattr(resp, "results", None) or []

        parts = [str(answer)] if answer else []
        for r in results[:4]:
            if isinstance(r, dict):
                parts.append(f"Source: {r.get('url','')}\n{r.get('content','')}")
            else:
                parts.append(f"Source: {getattr(r,'url','')}\n{getattr(r,'content','')}")

        return "\n\n".join(parts)

    except Exception as e:
        logger.warning(f"Tavily search failed: {e}")
        return ""
    
if __name__ == "__main__":
    mcp.run(transport="stdio")