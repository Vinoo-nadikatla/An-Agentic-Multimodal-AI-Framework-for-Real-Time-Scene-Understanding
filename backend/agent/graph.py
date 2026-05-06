"""
agent/graph.py
--------------
LangGraph state machine for the vision assistant.

Graph flow:
    conversation_node → tool_router
                            ├─ (tool call)  → tool_executor → conversation_node
                            └─ (no tool)    → END

State is a TypedDict kept in memory per WebSocket session.
Each session gets its own compiled graph instance via get_agent().
"""

from __future__ import annotations

import logging
import os
from typing import Annotated, Literal, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from agent.nodes import conversation_node, tool_executor_node
from agent.tools import get_tools

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    # Injected by the WebSocket handler before each invocation
    current_frame_b64: str | None


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

def tool_router(state: AgentState) -> Literal["tool_executor", "__end__"]:
    """Route to tool_executor if the last AI message has tool calls, else end."""
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tool_executor"
    return "__end__"


# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    tools = get_tools()

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.environ["GROQ_API_KEY"],
        streaming=True,
        temperature=0,
    ).bind_tools(tools)

    graph = StateGraph(AgentState)

    # Nodes
    graph.add_node("conversation", lambda s: conversation_node(s, llm))
    graph.add_node("tool_executor", lambda s: tool_executor_node(s, tools))

    # Edges
    graph.set_entry_point("conversation")
    graph.add_conditional_edges("conversation", tool_router, {
        "tool_executor": "tool_executor",
        "__end__": END,
    })
    graph.add_edge("tool_executor", "conversation")

    return graph.compile()


# ---------------------------------------------------------------------------
# Session registry  (one compiled graph per ws session — stateless graph,
# but we store message history per session in a plain dict)
# ---------------------------------------------------------------------------

_sessions: dict[str, dict] = {}


def get_session(session_id: str) -> dict:
    if session_id not in _sessions:
        _sessions[session_id] = {
            "graph": build_graph(),
            "messages": [],
        }
    return _sessions[session_id]


def drop_session(session_id: str) -> None:
    _sessions.pop(session_id, None)
