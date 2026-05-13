"""
agent/graph.py
LangGraph state machine with intent routing and error recovery.

Flow:
    intent_router (fast classify)
         ↓
    conversation_node (LLM)
         ↓ tool_router
    tool_executor (max 1 call)
         ↓
    conversation_node (synthesize answer)
         ↓
    END

Error recovery: any failure returns clean message, session stays alive.
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

MAX_TOOL_CALLS = 1  # Hard limit — prevents infinite loops


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    current_frame_b64: str | None
    intent: str           # injected by WebSocket handler
    tool_call_count: int  # tracks tool calls this turn


def tool_router(state: AgentState) -> Literal["tool_executor", "__end__"]:
    last = state["messages"][-1]
    intent = state.get("intent", "general")
    tool_calls_used = state.get("tool_call_count", 0)

    # General queries never need tools
    if intent == "general":
        return "__end__"

    if tool_calls_used >= MAX_TOOL_CALLS:
        return "__end__"

    if isinstance(last, AIMessage) and last.tool_calls:
        return "tool_executor"

    return "__end__"

def build_graph():
    tools = get_tools()
    _base_llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.environ["GROQ_API_KEY"],
        streaming=True,
        temperature=0,
    )
    plain_llm = _base_llm                  # no tools — for general / scene
    llm_with_tools = _base_llm.bind_tools(tools)  # vision tool available

    _TOOL_INTENTS = {"vision_describe", "vision_ocr"}

    def safe_conversation(s):
        try:
            intent = s.get("intent", "general")
            llm = llm_with_tools if intent in _TOOL_INTENTS else plain_llm
            return conversation_node(s, llm)
        except Exception as e:
            logger.error("conversation_node failed: %s", e)
            return {"messages": [AIMessage(content="Sorry, I had a temporary issue. Please try again.")]}
    def safe_tool_executor(s):
        try:
            result = tool_executor_node(s, tools)
            # Increment tool call counter
            count = s.get("tool_call_count", 0) + 1
            result["tool_call_count"] = count
            return result
        except Exception as e:
            logger.error("tool_executor failed: %s", e)
            return {
                "messages": [ToolMessage(
                    content="Tool failed. I'll answer based on what I know.",
                    tool_call_id="error"
                )],
                "tool_call_count": MAX_TOOL_CALLS  # prevent retry
            }

    graph = StateGraph(AgentState)
    graph.add_node("conversation", safe_conversation)
    graph.add_node("tool_executor", safe_tool_executor)
    graph.set_entry_point("conversation")
    graph.add_conditional_edges("conversation", tool_router, {
        "tool_executor": "tool_executor",
        "__end__": END,
    })
    graph.add_edge("tool_executor", "conversation")
    return graph.compile()


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