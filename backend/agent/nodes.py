from __future__ import annotations
import logging
import re
from typing import Sequence
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are VN AI, a friendly multimodal assistant with a live camera.

You have a camera tool. Use it when the user asks about anything visual.

Reply in the same language the user is using. Be natural and conversational.
Never start responses with 'The image depicts' or 'The image shows'. Instead say 'I can see...' or 'Right now...' or 'In front of me...'.
Never use asterisk actions like *zooms in* or *uses camera*. Respond in plain conversational language only.
When describing what you see, speak naturally as if you are the one observing, not as if you are describing a photograph."""

_VISION_INSTRUCTION = (
    "The user is asking about what the camera sees. "
    "You MUST call the analyze_image_with_query tool now. "
    "Do not answer without calling the tool first."
)

_OCR_INSTRUCTION = (
    "The user wants to read text visible in the camera. "
    "You MUST call the read_text_from_camera tool now. "
    "Do not answer without calling the tool first."
)


def _direct_vision_fallback(state: dict, intent: str) -> dict:
    """Call the vision tool directly when the LLM fails to generate a valid tool call."""
    messages = state.get("messages", [])
    human_msgs = [m for m in messages if hasattr(m, "content") and isinstance(m.content, str)]
    user_query = human_msgs[-1].content if human_msgs else "Describe what you see"
    try:
        if intent == "vision_ocr":
            from agent.tools import read_text_from_camera
            result = read_text_from_camera.invoke({"query": user_query})
        else:
            from agent.tools import analyze_image_with_query
            result = analyze_image_with_query.invoke({"query": user_query})
        logger.info("Vision direct fallback succeeded for intent=%s", intent)
        return {"messages": [AIMessage(content=result)]}
    except Exception as fe:
        logger.error("Vision fallback failed: %s", fe)
        return {"messages": [AIMessage(content="I cannot access the camera right now. Please try again.")]}


_SYNTHESIS_INSTRUCTION = (
    "The camera tool has returned its result above. "
    "Answer the user's question in plain text using that result. "
    "Do NOT call any tool again."
)


def conversation_node(state: dict, llm) -> dict:
    messages = list(state["messages"])
    intent = state.get("intent", "general")

    # Detect synthesis phase: tool already ran and returned a ToolMessage
    in_synthesis = bool(messages) and isinstance(messages[-1], ToolMessage)
    if in_synthesis:
        logger.info("Synthesis pass — tool result length: %d, preview: %.100s",
                    len(messages[-1].content), messages[-1].content)

    if intent == "scene":
        try:
            from services.scene_memory import get_scene
            scene = get_scene()
            people = scene.get("people", 0)
            motion = scene.get("motion", False)
            scene_msg = SystemMessage(content=(
                f"Current scene: {people} {'person' if people == 1 else 'people'} detected, "
                f"motion {'detected' if motion else 'not detected'}. "
                f"Answer directly from this. Do not use any tool."
            ))
            full_messages = [SystemMessage(content=SYSTEM_PROMPT), scene_msg] + messages
        except Exception:
            full_messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
    elif in_synthesis:
        # Second pass: synthesize from the tool result — never re-inject the tool-call instruction
        full_messages = [SystemMessage(content=SYSTEM_PROMPT),
                         SystemMessage(content=_SYNTHESIS_INSTRUCTION)] + messages
    elif intent == "vision_describe":
        full_messages = [SystemMessage(content=SYSTEM_PROMPT),
                         SystemMessage(content=_VISION_INSTRUCTION)] + messages
    elif intent == "vision_ocr":
        full_messages = [SystemMessage(content=SYSTEM_PROMPT),
                         SystemMessage(content=_OCR_INSTRUCTION)] + messages
    else:
        full_messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

    try:
        ai_msg = llm.invoke(full_messages)
        logger.info("LLM response — content length: %d, has tool_calls: %s",
                    len(str(ai_msg.content or "")), bool(getattr(ai_msg, "tool_calls", None)))
        if ai_msg.content:
            logger.info("LLM response preview: %.100s", str(ai_msg.content))
        elif in_synthesis:
            # LLM returned empty on synthesis — surface the raw tool result instead
            logger.warning("Synthesis returned empty content — returning tool result directly")
            return {"messages": [AIMessage(content=messages[-1].content)]}

        if isinstance(ai_msg.content, str):
            # Clean leaked tool syntax
            ai_msg.content = re.sub(r'<function=\w+>.*?</function>', '', ai_msg.content, flags=re.DOTALL)
            ai_msg.content = ai_msg.content.strip()
        return {"messages": [ai_msg]}
    except Exception as e:
        logger.error("LLM error: %s", e)
        # Synthesis failure — return the raw tool result rather than a generic error
        if in_synthesis and isinstance(messages[-1], ToolMessage):
            logger.info("Synthesis LLM failed — returning tool result directly")
            return {"messages": [AIMessage(content=messages[-1].content)]}
        if intent in ("vision_describe", "vision_ocr"):
            return _direct_vision_fallback(state, intent)
        return {"messages": [AIMessage(content="Sorry, something went wrong. Please try again.")]}


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
                logger.info("Tool result length: %d, preview: %.100s", len(str(output)), str(output))
            except Exception as e:
                logger.error("Tool %s failed: %s", tc["name"], e)
                output = "I could not complete that visual analysis. Please try again."
            results.append(ToolMessage(content=str(output), tool_call_id=tc["id"]))
    finally:
        tool_module._current_frame_b64.reset(tok)

    return {"messages": results}