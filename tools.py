# # import cv2
# # import base64
# # from dotenv import load_dotenv
# # from camera_stream import camera_manager
# # import base64
# # import cv2

# # load_dotenv()

# # def capture_image() -> str:
# #     """
# #     Captures one frame from the default webcam, resizes it,
# #     encodes it as Base64 JPEG (raw string) and returns it.
# #     """
# #     for idx in range(4):
# #         cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
# #         if cap.isOpened():
# #             for _ in range(10):  # Warm up
# #                 cap.read()
# #             ret, frame = cap.read()
# #             cap.release()
# #             if not ret:
# #                 continue
# #             cv2.imwrite("sample.jpg", frame)  
# #             ret, buf = cv2.imencode('.jpg', frame)
# #             if ret:
# #                 return base64.b64encode(buf).decode('utf-8')
# #     raise RuntimeError("Could not open any webcam (tried indices 0-3)")


# # from groq import Groq

# # # def analyze_image_with_query(query: str) -> str:
# # #     """
# # #     Expects a string with 'query'.
# # #     Captures the image and sends the query and the image to
# # #     to Groq's vision chat API and returns the analysis.
# # #     """

# # from langchain_core.tools import tool

# # @tool
# # def analyze_image_with_query(query: str) -> str:
# #     """
# #     Capture an image from the webcam and answer a question about it.

# #     Use this tool when the user asks something that requires
# #     seeing the user or their surroundings.

# #     Example questions:
# #     - Do I have a beard?
# #     - What am I holding?
# #     - What is behind me?
# #     """
# #     img_b64 = capture_image()
# #     model = "meta-llama/llama-4-scout-17b-16e-instruct"
# #     #model = "meta-llama/llama-4-maverick-17b-instruct"
# #     if not query or not img_b64:
# #         return "Error: both 'query' and 'image' fields required."

# #     client=Groq()  
# #     messages=[
# #         {
# #             "role": "user",
# #             "content": [
# #                 {
# #                     "type": "text", 
# #                     "text": query
# #                 },
# #                 {
# #                     "type": "image_url",
# #                     "image_url": {
# #                         "url": f"data:image/jpeg;base64,{img_b64}",
# #                     },
# #                 },
# #             ],
# #         }]
# #     chat_completion=client.chat.completions.create(
# #         messages=messages,
# #         model=model
# #     )

# #     return chat_completion.choices[0].message.content

# # # query ="what did the person holding in her hand?"
# # # print(analyze_image_with_query(query))

# import cv2
# import base64
# import os
# from groq import Groq
# from langchain_core.tools import tool


# from camera_stream import camera_stream

# def capture_image():

#     frame = camera_stream.get_frame()

#     if frame is None:
#         raise RuntimeError("Camera frame unavailable")

#     frame = cv2.resize(frame,(512,512))

#     _, buffer = cv2.imencode(".jpg", frame)

#     return base64.b64encode(buffer).decode("utf-8")

# @tool
# def analyze_image_with_query(query: str) -> str:
#     """
#     Capture webcam image and analyze it
#     """

#     img_b64 = capture_image()

#     client = Groq(api_key=os.getenv("GROQ_API_KEY"))

#     completion = client.chat.completions.create(

#         model="meta-llama/llama-3.2-11b-vision-instruct",

#         messages=[
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "text", "text": query},
#                     {
#                         "type": "image_url",
#                         "image_url": {
#                             "url": f"data:image/jpeg;base64,{img_b64}"
#                         }
#                     }
#                 ]
#             }
#         ],

#         temperature=0
#     )

#     return completion.choices[0].message.content

#last editing
"""
tools.py
--------
LangChain tools available to the agent.
Only ONE tool is registered: analyze_image_with_query.

The agent must never hallucinate additional tools.
The system prompt in ai_agent.py enforces this at the LLM level.
"""

import cv2
import base64
import os
import logging
from groq import Groq
from langchain_core.tools import tool
from camera_stream import camera_stream

logger = logging.getLogger(__name__)

# ── Groq vision model identifier ──────────────────────────────────────────────
# llama-3.2-11b-vision-preview was decommissioned April 2025.
# Current replacement: meta-llama/llama-4-scout-17b-16e-instruct
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"


def _capture_image_b64(size: int = 512) -> str:
    """
    Grab the latest camera frame, resize it, and return as a base64 JPEG string.
    Raises RuntimeError if no frame is available.
    """
    frame = camera_stream.get_frame()

    if frame is None:
        raise RuntimeError(
            "Camera frame unavailable. Make sure camera_stream.start() was called."
        )

    frame = cv2.resize(frame, (size, size))
    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buffer).decode("utf-8")


@tool
def analyze_image_with_query(query: str) -> str:
    """
    Use this tool ONLY when the user's question requires seeing the live camera feed.
    Examples: object identification, color questions, counting people,
    describing the environment, reading text in view.

    Do NOT use this tool for general knowledge questions, math, definitions,
    history, or anything that does not require visual context.

    Args:
        query: The specific visual question to answer about the current camera view.

    Returns:
        A natural language description answering the visual query.
    """
    logger.info("analyze_image_with_query called with query: %s", query)

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "Error: GROQ_API_KEY not set."

    try:
        img_b64 = _capture_image_b64()
    except RuntimeError as e:
        return f"Error capturing image: {e}"

    client = Groq(api_key=api_key)

    try:
        completion = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": query},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_b64}"
                            },
                        },
                    ],
                }
            ],
            temperature=0,
            max_tokens=512,
        )
        return completion.choices[0].message.content

    except Exception as e:
        logger.error("Vision API call failed: %s", e)
        # Raise — don't return an error string.
        # If we return a string, the agent retries the tool repeatedly.
        # Raising causes LangGraph to surface it cleanly and stop.
        raise RuntimeError(f"Vision analysis failed: {e}") from e
