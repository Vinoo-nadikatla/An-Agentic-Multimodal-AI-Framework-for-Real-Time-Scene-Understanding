from __future__ import annotations
import logging
import re
from typing import Sequence
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)

_TEMPORAL_RE = re.compile(
    r"\b(happened|history|last|past|ago|earlier|recently|timeline|log|record|minute|hour|today)\b",
    re.IGNORECASE,
)


def _detect_language(text: str) -> str:
    """Detect if text is Telugu, Hindi, or English based on Unicode script ranges."""
    telugu_chars = sum(1 for c in text if "ఀ" <= c <= "౿")
    hindi_chars  = sum(1 for c in text if "ऀ" <= c <= "ॿ")
    if telugu_chars > 0:
        return "Telugu"
    if hindi_chars > 0:
        return "Hindi"
    return "English"

SYSTEM_PROMPT = """You are VN AI Safety Monitor, an AI-powered industrial safety assistant with a live camera.

You monitor worker safety in real-time. You can:
- Check if workers are wearing PPE (helmets, vests, gloves, masks)
- Detect unsafe behavior or movements
- Count workers present in the zone
- Identify safety violations
- Answer questions about what is happening in the monitored area

Always be specific and actionable in your safety assessments.
Respond in the same language the user speaks.
You may answer simple factual questions like current date, time, or weather briefly, then redirect to safety monitoring. Never refuse to respond — always give some answer.

CRITICAL : 
- After the camera tool returns results, immediately give your safety assessment as plain text. Never call any tool a second time.
- Any person you see in the camera IS A WORKER in the monitored area
- NEVER say "this appears to be a domestic setting" or "no workers present"
- NEVER say the person does not need PPE because of the environment
- If you see a person without a helmet, that IS a safety violation
- You monitor for PPE compliance — helmets are required for all workers
- The monitored area may look like any room — treat all persons as workers
- Always respond as an industrial safety AI, not a general assistant
"""

def conversation_node(state: dict, llm) -> dict:
    from datetime import datetime as _dt
    messages = list(state["messages"])
    intent = state.get("intent", "general")

    # Always inject current date/time and detected language for every intent
    now_str = _dt.now().strftime("%A, %d %B %Y, %I:%M %p")
    human_msgs = [m for m in messages if isinstance(m, HumanMessage)]
    last_human = human_msgs[-1].content if human_msgs else ""
    user_language = _detect_language(last_human)

    if user_language == "Hindi":
        language_instruction = (
            "\n\nIMPORTANT: The user is speaking Hindi. "
            "You MUST respond in Hindi (Devanagari script). Do not respond in English."
        )
    elif user_language == "Telugu":
        language_instruction = (
            "\n\nIMPORTANT: The user is speaking Telugu. "
            "You MUST respond in Telugu (Telugu script). Do not respond in English."
        )
    else:
        language_instruction = ""

    dynamic_system = SYSTEM_PROMPT + f"\n\nCurrent date and time: {now_str}" + language_instruction
    # Reports are always formal English documents — no language override
    report_system  = SYSTEM_PROMPT + f"\n\nCurrent date and time: {now_str}"

    # Report generation — build structured safety report from live data
    if intent == "report":
        try:
            from services.activity_log import get_summary
            from services.ppe_detector import get_ppe_status
            ppe = get_ppe_status()
            activity = get_summary(minutes=480)
            worker_lines = "\n".join(
                f"  - {w['label']}: {', '.join(w['violations']) if w['violations'] else 'Compliant'}"
                + (f" (Duration: {w['violation_duration']})" if w.get("violation_duration") else "")
                for w in ppe.get("workers", [])
            ) or "  No workers currently detected"
            report_context = (
                f"Generate a professional industrial safety monitoring report.\n\n"
                f"Current date/time: {now_str}\n\n"
                f"LIVE PPE STATUS:\n"
                f"- Total workers: {ppe['total_workers']}\n"
                f"- Compliant: {ppe['compliant']}\n"
                f"- Violations: {ppe['violations']}\n"
                f"- Helmet compliance: {ppe['helmet_compliance']}%\n"
                f"- Overall compliance: {ppe['overall_compliance']}%\n\n"
                f"Workers:\n{worker_lines}\n\n"
                f"ACTIVITY LOG:\n{activity}\n\n"
                f"IMPORTANT: You MUST format the report with EXACTLY these 5 numbered sections:\n\n"
                f"1. EXECUTIVE SUMMARY\n"
                f"(2-3 sentences about overall safety status)\n\n"
                f"2. COMPLIANCE SUMMARY\n"
                f"(bullet points with exact percentages)\n\n"
                f"3. VIOLATIONS DETECTED\n"
                f"(list each violation with worker name, what is missing, and duration if available)\n\n"
                f"4. ACTIVITY TIMELINE\n"
                f"(chronological list from the activity log)\n\n"
                f"5. RECOMMENDATIONS\n"
                f"(numbered list of specific actionable steps)\n\n"
                f"Do not add any other sections. Do not use conversational language. This is a formal safety report."
            )
            full_messages = [SystemMessage(content=report_system),
                             HumanMessage(content=report_context)]
        except Exception:
            full_messages = [SystemMessage(content=report_system)] + messages

    # Scene queries — inject real data and answer without tools
    elif intent == "scene":
        scene_context = ""
        try:
            from services.scene_memory import get_scene
            from services.activity_log import get_summary
            from services.ppe_detector import get_ppe_status
            ppe   = get_ppe_status()
            scene = get_scene()

            last_user_text = messages[-1].content if messages else ""
            if _TEMPORAL_RE.search(last_user_text):
                # Temporal query — use activity log
                activity = get_summary(minutes=60)
                scene_context = (
                    f"Activity Log (last 60 minutes):\n{activity}\n\n"
                    f"Current PPE Status:\n"
                    f"- Workers: {ppe['total_workers']}\n"
                    f"- Helmet compliance: {ppe['helmet_compliance']}%\n"
                    f"- Violations: {ppe['violations']}\n"
                )
            else:
                # Live query — use current detection data
                worker_lines = "\n".join(
                    f"  - {w['label']}: {', '.join(w['violations']) if w['violations'] else 'Compliant'}"
                    + (f" ({w['violation_duration']})" if w.get("violation_duration") else "")
                    for w in ppe.get("workers", [])
                ) or "  No workers detected"
                scene_context = (
                    f"Current Safety Status:\n"
                    f"- Workers detected: {ppe['total_workers']}\n"
                    f"- Compliant: {ppe['compliant']}\n"
                    f"- Violations: {ppe['violations']}\n"
                    f"- Helmet compliance: {ppe['helmet_compliance']}%\n"
                    f"- Motion: {'Yes' if scene.get('motion') else 'No'}\n\n"
                    f"Per-worker status:\n{worker_lines}\n"
                )
        except Exception as e:
            logger.warning("Scene data unavailable: %s", e)

        if not scene_context or not scene_context.strip():
            return {"messages": [AIMessage(content="System is initializing, please wait a moment and try again.")]}

        scene_system = SystemMessage(content=(
            dynamic_system
            + "\n\nCRITICAL: Answer ONLY using the data provided below. "
            "Do NOT call any tool or function. Do NOT use the camera. "
            "Do NOT guess, assume, or add details not in the data. "
            "If data says 0 workers, say 0 workers. "
            "Do not mention gloves, goggles, or any PPE not tracked by the system.\n\n"
            + scene_context
        ))
        full_messages = [scene_system] + messages
    else:
        full_messages = [SystemMessage(content=dynamic_system)] + messages

    try:
        ai_msg = llm.invoke(full_messages)
        if intent == "scene":
            logger.info("Scene response content length: %d, preview: %.80s",
                        len(ai_msg.content or ""), ai_msg.content or "")
        if isinstance(ai_msg.content, str):
            # Remove leaked tool-call XML syntax
            ai_msg.content = re.sub(r'<function=\w+>.*?</function>', '', ai_msg.content, flags=re.DOTALL)
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