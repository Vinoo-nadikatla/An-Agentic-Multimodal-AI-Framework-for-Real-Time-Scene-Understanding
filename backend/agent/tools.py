"""
agent/tools.py
Vision tool using Groq llama-4-scout.
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
from groq import Groq
from langchain_core.tools import BaseTool, tool

logger = logging.getLogger(__name__)

_current_frame_b64: contextvars.ContextVar[str | None] = \
    contextvars.ContextVar("_current_frame_b64", default=None)

# EasyOCR reader singleton — loaded once, reused every call
_ocr_reader = None

def _get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        _ocr_reader = easyocr.Reader(['en'], gpu=False)
    return _ocr_reader

VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"


def _get_frame_b64(size: int = 320) -> str:
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
    """Capture a live camera frame and answer a visual question about it.

    Args:
        query: The visual question to answer.
    Returns:
        Natural language answer based on what the camera currently sees.
    """
    logger.info("Vision tool called: %s", query)

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return "Error: GROQ_API_KEY not set."

    try:
        img_b64 = _get_frame_b64()
    except RuntimeError as e:
        return str(e)

    try:
        client = Groq(api_key=api_key)
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
        return f"Vision analysis failed: {e}"


@tool
def read_text_from_camera(query: str) -> str:
    """Read text, numbers, or symbols visible in the camera view using OCR.

    Args:
        query: What text the user wants to read.
    Returns:
        Text extracted from the camera image.
    """
    logger.info("OCR tool called: %s", query)
    try:
        img_b64 = _get_frame_b64()
    except RuntimeError as e:
        return str(e)

    try:
        import numpy as np

        img_bytes = base64.b64decode(img_b64)
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        reader = _get_ocr_reader()
        results = reader.readtext(frame)

        if not results:
            return "No readable text found in the camera view."

        texts = [text for (_, text, conf) in results if conf > 0.25]
        if not texts:
            return "Text detected but could not be read clearly."

        return "Text visible: " + " | ".join(texts)

    except Exception as e:
        logger.error("OCR error: %s", e)
        return f"OCR failed: {e}"


def get_tools() -> list[BaseTool]:
    return [analyze_image_with_query, read_text_from_camera]