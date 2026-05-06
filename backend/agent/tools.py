"""
agent/tools.py
Vision tool using Google Gemini 1.5 Flash.
Frame source priority:
  1. Browser-streamed frame via _current_frame_b64 context-var
  2. Server-side OpenCV camera_stream
"""
from __future__ import annotations
import base64
import contextvars
import logging
import os
import cv2
import numpy as np
from langchain_core.tools import BaseTool, tool

logger = logging.getLogger(__name__)

_current_frame_b64: contextvars.ContextVar[str | None] = \
    contextvars.ContextVar("_current_frame_b64", default=None)


def _get_frame_b64(size: int = 512) -> str:
    frame_b64 = _current_frame_b64.get()
    if frame_b64:
        try:
            arr = np.frombuffer(base64.b64decode(frame_b64), dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is not None:
                frame = cv2.resize(frame, (size, size))
                _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                return base64.b64encode(buf).decode()
        except Exception as e:
            logger.warning("Browser frame decode failed: %s", e)

    try:
        from services.camera_stream import camera_stream
        frame = camera_stream.get_frame()
        if frame is not None:
            frame = cv2.resize(frame, (size, size))
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return base64.b64encode(buf).decode()
    except Exception:
        pass

    raise RuntimeError(
        "No camera frame available. "
        "Make sure your browser camera is active or the server camera is connected."
    )


@tool
def analyze_image_with_query(query: str) -> str:
    """
    Captures a live frame from the camera and analyzes it using AI vision.

    Call this tool when the user wants to know anything about:
    - What is physically present in their environment
    - Their own appearance or what they are wearing
    - Objects, colors, text, or people visible through the camera
    - What the room or background looks like
    - Motion, activity, or anything happening around them

    Do not call this tool for:
    - General knowledge questions
    - Math, history, science, definitions
    - Questions about the past or hypothetical situations
    - Follow-up questions about something already described this conversation

    This tool makes one API call per invocation. Call it only once per query.

    Args:
        query: The specific visual question to answer.
    Returns:
        Natural language answer based on what the camera currently sees.
    """
    logger.info("Gemini vision called: %s", query)

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return "Error: GEMINI_API_KEY not set. Please add it to your .env file."

    try:
        img_b64 = _get_frame_b64()
    except RuntimeError as e:
        return str(e)

    try:
        import google.generativeai as genai
        import PIL.Image
        import io

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        img = PIL.Image.open(io.BytesIO(base64.b64decode(img_b64)))
        response = model.generate_content([query, img])
        return response.text

    except Exception as e:
        logger.error("Gemini vision error: %s", e)
        return f"Vision analysis failed: {e}"


def get_tools() -> list[BaseTool]:
    return [analyze_image_with_query]
