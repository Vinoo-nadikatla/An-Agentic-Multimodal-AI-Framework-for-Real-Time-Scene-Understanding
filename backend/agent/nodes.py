from __future__ import annotations
import logging
from typing import Sequence
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)

# SYSTEM_PROMPT = """You are VN AI, an intelligent assistant with access to a live camera feed.

# You have one tool: analyze_image_with_query(query)

# Use this tool whenever the user is asking about something that requires seeing - their environment, appearance, objects nearby, people, motion, colors, or anything physical around them.

# Do NOT use the tool for general knowledge, math, history, coding, definitions, or anything that does not require visual context.

# When you are unsure whether to use the tool, ask yourself: "Would seeing the camera help answer this?" If yes, use it.

# Always respond naturally and conversationally. Never mention tool names to the user. Never say you cannot see - you have a camera and can use it anytime."""
SYSTEM_PROMPT = """You are VN AI, a friendly multimodal assistant with a live camera.

Respond in the same language the user speaks. Be natural and conversational —
talk like a friend, not a textbook.

You have a camera tool available. Use it naturally when seeing the environment
would genuinely help answer the question — just like a person would look around
when asked about their surroundings."""

def conversation_node(state: dict, llm) -> dict:
    messages: list[BaseMessage] = list(state["messages"])
    full_messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
    try:
        ai_msg = llm.invoke(full_messages)
        return {"messages": [ai_msg]}
    except Exception as e:
        logger.error("LLM error: %s", e)
        return {"messages": [AIMessage(content="Sorry, I had trouble with that. Please try again.")]}


def tool_executor_node(state: dict, tools: Sequence[BaseTool]) -> dict:
    last_ai: AIMessage = state["messages"][-1]
    tool_map = {t.name: t for t in tools}

    import agent.tools as tool_module
    frame_b64 = state.get("current_frame_b64")
    tok = tool_module._current_frame_b64.set(frame_b64)

    results = []
    try:
        for tc in last_ai.tool_calls:
            tool = tool_map.get(tc["name"])
            try:
                output = tool.invoke(tc["args"]) if tool else f"Unknown tool: {tc['name']}"
            except Exception as e:
                logger.error("Tool %s failed: %s", tc["name"], e)
                output = f"Tool error: {e}"
            results.append(ToolMessage(content=str(output), tool_call_id=tc["id"]))
    finally:
        tool_module._current_frame_b64.reset(tok)

    return {"messages": results}
