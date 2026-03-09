"""
main.py – FastAPI Application Entry Point

Exposes a single POST /chat endpoint that:
  1. Accepts a user message
  2. Runs it through the LangGraph 4-agent workflow
  3. Returns the assistant's final response
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

from chronic_chatbot.config import LOG_LEVEL
from chronic_chatbot.graph import graph_app

# ── Logging ───────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)


# ── App Lifespan ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Chronic Chatbot starting up…")
    yield
    logger.info("🛑 Chronic Chatbot shutting down…")


# ── FastAPI App ───────────────────────────────────────────────
app = FastAPI(
    title="Chronic Disease Chatbot API",
    description="4-Agent LangGraph system for chronic disease management",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Models ─────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    user_id: str = "default_user"


class ChatResponse(BaseModel):
    reply: str
    agent_path: list[str] = []


# ── In-memory session state (replace with Redis for production) 
SESSIONS: dict = {}


# ── Endpoints ─────────────────────────────────────────────────
@app.get("/health")
def health_check():
    return {"status": "ok", "service": "chronic_chatbot"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint.
    
    Runs the user message through the 4-agent LangGraph workflow
    and returns the final assistant response.
    """
    logger.info(f"💬 New message from {request.user_id}: {request.message[:80]}…")

    # Load or initialise session state
    session = SESSIONS.get(request.user_id, {
        "messages": [],
        "user_profile": {},
        "next_step": "",
        "current_plan": [],
        "knowledge_context": None,
        "memory_context": None,
        "action_result": None,
    })

    # Append the new human message
    session["messages"] = session["messages"] + [HumanMessage(content=request.message)]

    try:
        # Run the graph (synchronous compile, async wrapper for FastAPI)
        result_state = graph_app.invoke(session)
    except Exception as e:
        logger.error(f"Graph execution failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Agent error: {e}")

    # Persist updated session
    SESSIONS[request.user_id] = result_state

    # Extract the final AI response (last message in history)
    messages = result_state.get("messages", [])
    final_reply = "I'm sorry, I couldn't generate a response." 
    for msg in reversed(messages):
        # Skip internal routing messages; pick the clean final answer
        if hasattr(msg, "content") and not msg.content.startswith("["):
            final_reply = msg.content
            break
    if final_reply == "I'm sorry, I couldn't generate a response.":
        # Fallback: last message content regardless
        if messages:
            final_reply = messages[-1].content

    logger.info(f"✅ Reply generated for {request.user_id}")

    return ChatResponse(reply=final_reply)


# ── Dev Server Entry Point ────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("chronic_chatbot.main:app", host="0.0.0.0", port=8000, reload=True)
