from __future__ import annotations
import logging
import re
from typing import Sequence
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are VN AI Safety Monitor, an AI-powered industrial safety assistant with a live camera.

You monitor worker safety in real-time. You can:
- Check if workers are wearing PPE (helmets, vests, gloves, masks)
- Detect unsafe behavior or movements
- Count workers present in the zone
- Identify safety violations
- Answer questions about what is happening in the monitored area

Always be specific and actionable in your safety assessments.
Respond in the same language the user speaks.

CRITICAL: After the camera tool returns results, immediately give your safety assessment as plain text. Never call any tool a second time."""

def conversation_node(state: dict, llm) -> dict:
    messages = list(state["messages"])
    intent = state.get("intent", "general")

    # Scene queries — inject real data and answer without tools
    if intent == "scene":
        try:
            from services.scene_memory import get_scene
            from services.activity_log import get_summary
            scene = get_scene()
            people = scene.get("people", 0)
            motion = scene.get("motion", False)
            log_summary = get_summary(minutes=60)
            context = (
                f"Current scene: {people} {'worker' if people == 1 else 'workers'} detected, "
                f"motion {'active' if motion else 'not detected'}.\n"
                f"{log_summary}\n"
                f"Answer the user's safety question using this data. Do not use any tool."
            )
            full_messages = [SystemMessage(content=SYSTEM_PROMPT),
                             SystemMessage(content=context)] + messages
        except Exception:
            full_messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
    else:
        full_messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

    try:
        ai_msg = llm.invoke(full_messages)
        if isinstance(ai_msg.content, str):
            # Clean leaked tool syntax and action narration
            ai_msg.content = re.sub(r'<function=\w+>.*?</function>', '', ai_msg.content, flags=re.DOTALL)
            ai_msg.content = re.sub(r'\*[^*]+\*', '', ai_msg.content)
            ai_msg.content = ai_msg.content.strip()
        return {"messages": [ai_msg]}
    except Exception as e:
        logger.error("LLM error: %s", e)
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
            except Exception as e:
                logger.error("Tool %s failed: %s", tc["name"], e)
                output = "I could not complete that visual analysis. Please try again."
            results.append(ToolMessage(content=str(output), tool_call_id=tc["id"]))
    finally:
        tool_module._current_frame_b64.reset(tok)

    return {"messages": results}