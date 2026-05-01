"""
agent/tools.py
--------------
LangChain tools for the vision agent.

Only ONE tool is exposed: analyze_image_with_query.

Frame source priority:
  1. Browser-streamed frame injected via _current_frame_b64 context-var
     (set by tool_executor_node from state["current_frame_b64"])
  2. Server-side camera_stream (if running — local/Pi deployments)

This means the same tool works for both deployment modes transparently.
"""

from __future__ import annotations

import base64
import contextvars
import logging
import os

import cv2
import numpy as np
from groq import Groq
from langchain_core.tools import BaseTool, tool

logger = logging.getLogger(__name__)

# Context-var set by tool_executor_node before each tool call
_current_frame_b64: contextvars.ContextVar[str | None] = \
    contextvars.ContextVar("_current_frame_b64", default=None)

VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_frame_b64(size: int = 512) -> str:
    """
    Return a base64-encoded JPEG frame from the best available source.
    Raises RuntimeError if no frame is available.
    """
    # Priority 1: browser-streamed frame (already base64-encoded JPEG)
    frame_b64 = _current_frame_b64.get()
    if frame_b64:
        # Decode → resize → re-encode at target size for consistency
        try:
            img_bytes = base64.b64decode(frame_b64)
            arr = np.frombuffer(img_bytes, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is not None:
                frame = cv2.resize(frame, (size, size))
                _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                return base64.b64encode(buf).decode("utf-8")
        except Exception as e:
            logger.warning("Failed to decode browser frame: %s", e)

    # Priority 2: server-side OpenCV camera
    try:
        from services.camera_stream import camera_stream
        frame = camera_stream.get_frame()
        if frame is not None:
            frame = cv2.resize(frame, (size, size))
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return base64.b64encode(buf).decode("utf-8")
    except Exception:
        pass  # Camera not available on this deployment

    raise RuntimeError(
        "No camera frame available. "
        "Ensure the browser camera is active or the server-side camera is connected."
    )


# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------

@tool
def analyze_image_with_query(query: str) -> str:
    """
    Use this tool ONLY when the user's question requires seeing the live camera feed.

    Examples: object identification, counting people, describing surroundings,
    reading visible text, colour questions, motion detection confirmation.

    Do NOT use for general knowledge, math, history, or follow-up questions
    about a scene already described in this conversation.

    Args:
        query: The specific visual question to answer about the current camera view.

    Returns:
        A natural-language answer based on what the camera sees.
    """
    logger.info("analyze_image_with_query: %s", query)

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return "Error: GROQ_API_KEY not configured."

    try:
        img_b64 = _get_frame_b64()
    except RuntimeError as e:
        return str(e)

    client = Groq(api_key=api_key)
    try:
        resp = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{img_b64}"
                    }},
                ],
            }],
            temperature=0,
            max_tokens=512,
        )
        return resp.choices[0].message.content
    except Exception as e:
        logger.error("Vision API error: %s", e)
        raise RuntimeError(f"Vision analysis failed: {e}") from e


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def get_tools() -> list[BaseTool]:
    return [analyze_image_with_query]
