# # # from langchain_groq import ChatGroq
# # # from langgraph.prebuilt import create_react_agent
# # # from dotenv import load_dotenv
# # # from tools import analyze_image_with_query


# # # load_dotenv()

# # # system_prompt = """
# # # You are an intelligent multimodal assistant.

# # # You have access to a tool called `analyze_image_with_query`
# # # which captures an image from the webcam and analyzes it.

# # # Follow these rules when deciding to use the tool:

# # # 1. Use the tool ONLY if the question requires visual information.
# # # 2. Do NOT use the tool for general knowledge questions.
# # # 3. If the question asks about the user's appearance, environment, or objects nearby,
# # #    then call the tool.
# # # 4. Examples where the tool SHOULD be used:
# # #    - "Do I have a beard?"
# # #    - "What is behind me?"
# # #    - "Am I holding a phone?"
# # # 5. Examples where the tool SHOULD NOT be used:
# # #    - "What is Python?"
# # #    - "Who is the president of India?"

# # # If you use the tool, explain the result naturally and conversationally.
# # # """



# # # llm = ChatGroq(
# # #     model="llama-3.3-70b-versatile",
# # #     temperature=0.7
# # # )

# # # agent = create_react_agent(
# # #     model=llm,
# # #     tools=[analyze_image_with_query],
# # #     prompt=system_prompt
    
# # # )

# # # # def ask_agent(user_query: str):
# # # #     response = agent.invoke(
# # # #         {"messages":[{"role":"user","content":user_query}]}
# # # #     )
# # # #     return response["messages"][-1].content


# # # # print(ask_agent(user_query="Do I have beard?"))

# # #/////////////////////////////////////////////////////////


# # # from langchain_groq import ChatGroq
# # # from langgraph.prebuilt import create_react_agent

# # # from tools import analyze_image_with_query
# # # from scene_memory import get_scene
# # # from dotenv import load_dotenv
# # # import os

# # # load_dotenv()

# # # llm = ChatGroq(
# # #     model="llama-3.3-70b-versatile",
# # #     temperature=0.7,
# # #     groq_api_key=os.getenv("GROQ_API_KEY")
# # # )

# # # agent=create_react_agent(
# # #     model=llm,
# # #     tools=[analyze_image_with_query]
# # # )


# # # def ask_agent(user_query):

# # #     scene=get_scene()

# # #     context=f"""
# # # Environment Status
# # # People detected: {scene['people']}
# # # Motion detected: {scene['motion']}
# # # """

# # #     full_query=context+"\nUser question: "+user_query

# # #     response=agent.invoke(
# # #         {"messages":[{"role":"user","content":full_query}]}
# # #     )

# # #     return response["messages"][-1].content

# # from langchain_groq import ChatGroq
# # from langgraph.prebuilt import create_react_agent
# # from dotenv import load_dotenv
# # import os

# # from tools import analyze_image_with_query
# # from scene_memory import get_scene

# # load_dotenv()
# # tools = [analyze_image_with_query]

# # llm = ChatGroq(
# #     model="llama-3.1-8b-instant",
# #     temperature=0.2,
# #     groq_api_key=os.getenv("GROQ_API_KEY")
# # )

# # system_prompt =""" ## ROLE
# # You are a witty, multimodal AI Agent. You can "see" through a webcam and "hear" through audio.

# # ## CAPABILITIES
# # 1. **Environment Context**: You receive real-time data on 'People detected' and 'Motion'.
# # 2. **Vision Tool**: You can call `analyze_image_with_query` to get a deep visual analysis of the current frame.

# # ## TOOL CALLING LOGIC (STRICT)
# # Before responding, follow these steps:
# # 1. **Check Context**: Look at the "Environment information" provided in the prompt.
# #    - If the user asks "How many people?" and the context says `People detected: 2`, answer directly. DO NOT use the vision tool.
# # 2. **Evaluate Tool Necessity**: Only call `analyze_image_with_query` if the question requires details NOT in the context (e.g., colors, specific actions, text on a screen, facial features).
# # 3. **Internal Reasoning**: If you aren't sure, think: "Can I answer this using only general knowledge or the provided scene memory?" 
# #    - General Knowledge (e.g., "Who is the PM?") -> Answer immediately.
# #    - Scene Memory (e.g., "Is anyone there?") -> Answer using context.
# #    - Visual Detail (e.g., "What is the color of my shirt?") -> CALL TOOL.

# # ## PERSONALITY
# # Snappy, human-like, and clever. If you use a tool, don't say "I am calling a tool"; just describe what you see naturally.
# #     """

# # agent = create_react_agent(
# #     model=llm,
# #     tools=tools,
# #     prompt=system_prompt
# # )

# # def ask_agent(user_query):

# #     scene = get_scene()

# #     context = f"""
# #                 Environment information:
# #                 People detected: {scene['people']}
# #                 Motion detected: {scene['motion']}
# #                 """

# #     full_query = context + "\nUser question: " + user_query

# #     response = agent.invoke(
# #         {"messages":[{"role":"user","content":full_query}]},
# #         config={"recursion_limit":50}
# #     )

# #     return response["messages"][-1].content
# #//////////////////////////////////////////////////////////////////////////

# # """
# # ai_agent.py
# # -----------
# # Two-stage pipeline replacing the LangGraph ReAct loop.

# # WHY WE DROPPED THE REACT LOOP:
# #   LangGraph's ReAct loop caused cascading retries, malformed tool JSON (400),
# #   and recursion_limit exhaustion with Groq's rate limits.

# # THE NEW APPROACH — Simple 2-stage router:
# #   Stage 1 (CLASSIFIER): Fast 8b model → YES/NO, does this need vision?
# #   Stage 2a (VISION PATH): Call vision tool once → 70b model synthesizes answer
# #   Stage 2b (TEXT PATH): 70b model answers directly from knowledge + scene data

# #   Max 2 LLM calls per response. Zero looping. Zero recursion limit issues.
# # """

# # import os
# # import logging
# # from groq import Groq
# # from dotenv import load_dotenv

# # from tools import analyze_image_with_query
# # from scene_memory import get_scene

# # load_dotenv()
# # logger = logging.getLogger(__name__)

# # CLASSIFIER_MODEL = "llama-3.1-8b-instant"    # fast classifier — YES/NO only
# # ANSWERER_MODEL   = "llama-3.3-70b-versatile"  # strong synthesizer — final answer

# # def _get_client() -> Groq:
# #     api_key = os.getenv("GROQ_API_KEY")
# #     if not api_key:
# #         raise ValueError("GROQ_API_KEY not set.")
# #     return Groq(api_key=api_key)


# # # ── Stage 1: Classifier ───────────────────────────────────────────────────────

# # CLASSIFIER_PROMPT = """You are a query classifier. Decide if a question needs a live camera image to answer.
# # Reply with exactly one word: YES or NO.

# # Reply YES if the question asks about:
# # - what something looks like (color, shape, size, style)
# # - what a person is wearing, doing, or looks like  
# # - objects, people, or text visible in the room
# # - the environment, background, or surroundings
# # - whether something is present or absent visually

# # Reply NO if the question is about:
# # - general knowledge, facts, history, science, math
# # - definitions, explanations, jokes, greetings, opinions
# # - anything that doesn't require seeing the physical environment

# # Examples:
# # Q: describe my surroundings → YES
# # Q: what color is my shirt → YES
# # Q: do I have a beard → YES
# # Q: how many doors are in the room → YES
# # Q: what am I holding → YES
# # Q: who is the prime minister of india → NO
# # Q: what is python → NO
# # Q: tell me a joke → NO
# # Q: hi → NO

# # Reply with only YES or NO."""


# # def _needs_vision(query: str, scene: dict) -> bool:
# #     """Returns True if query requires the camera. Defaults to False on error."""
# #     # Scene data already answers these — skip vision
# #     scene_keywords = ["motion", "movement", "moving", "how many people", "anyone in the room"]
# #     if any(kw in query.lower() for kw in scene_keywords):
# #         return False

# #     try:
# #         client = _get_client()
# #         response = client.chat.completions.create(
# #             model=CLASSIFIER_MODEL,
# #             messages=[
# #                 {"role": "system", "content": CLASSIFIER_PROMPT},
# #                 {"role": "user", "content": f"Q: {query}"},
# #             ],
# #             temperature=0,
# #             max_tokens=5,
# #         )
# #         answer = response.choices[0].message.content.strip().upper()
# #         needs = answer.startswith("YES")
# #         logger.info("Classifier: '%s' → %s", query, "YES" if needs else "NO")
# #         return needs
# #     except Exception as e:
# #         logger.warning("Classifier failed, defaulting to text path: %s", e)
# #         return False


# # # ── Stage 2: Synthesizer ──────────────────────────────────────────────────────

# # DORA_SYSTEM = """You are Dora — a witty, warm, and perceptive AI assistant.
# # Be conversational and direct. Never robotic. Keep answers concise unless detail is asked for.
# # When you have visual information, describe it naturally as if you can see it yourself.
# # Never mention "tools", "functions", or "camera analysis" — just respond as Dora."""


# # def _synthesize(user_query: str, scene: dict, visual_context: str | None = None) -> str:
# #     """Generate the final answer using 70b model."""
# #     motion_str = "Yes" if scene["motion"] else "No"
# #     scene_block = f"[Scene] People: {scene['people']}, Motion: {motion_str}\n"

# #     if visual_context:
# #         user_content = (
# #             f"{scene_block}"
# #             f"[Camera sees]: {visual_context}\n\n"
# #             f"User: {user_query}\n"
# #             f"Answer using the camera info above. Be natural."
# #         )
# #     else:
# #         user_content = f"{scene_block}\nUser: {user_query}"

# #     try:
# #         client = _get_client()
# #         response = client.chat.completions.create(
# #             model=ANSWERER_MODEL,
# #             messages=[
# #                 {"role": "system", "content": DORA_SYSTEM},
# #                 {"role": "user", "content": user_content},
# #             ],
# #             temperature=0.7,
# #             max_tokens=512,
# #         )
# #         return response.choices[0].message.content.strip()
# #     except Exception as e:
# #         logger.error("Synthesizer failed: %s", e)
# #         return f"Sorry, I hit an error: {e}"


# # # ── Public API ────────────────────────────────────────────────────────────────

# # def ask_agent(user_query: str) -> str:
# #     """
# #     Two-stage pipeline:
# #       1. Classify: needs vision?
# #       2a. YES → call vision tool once → synthesize with visual context
# #       2b. NO  → synthesize from knowledge + scene data only
# #     """
# #     scene = get_scene()

# #     if _needs_vision(user_query, scene):
# #         logger.info("Vision path: %s", user_query)
# #         try:
# #             vision_query = (
# #                 f"Answer this about what you see: {user_query}. "
# #                 "Include all relevant visual details."
# #             )
# #             visual_result = analyze_image_with_query.invoke({"query": vision_query})
# #             logger.info("Vision result preview: %s", str(visual_result)[:120])
# #         except Exception as e:
# #             logger.error("Vision tool failed: %s", e)
# #             visual_result = None
# #         return _synthesize(user_query, scene, visual_context=visual_result)
# #     else:
# #         logger.info("Text path: %s", user_query)
# #         return _synthesize(user_query, scene, visual_context=None)
    
    
# """
# ai_agent.py
# -----------
# Two-stage pipeline: Classifier → Vision or Text path.
# Max 2 LLM calls per response. Zero looping. Zero recursion issues.
# """

# import os
# import logging
# from groq import Groq
# from dotenv import load_dotenv

# from tools import analyze_image_with_query
# from scene_memory import get_scene

# load_dotenv()
# logger = logging.getLogger(__name__)

# CLASSIFIER_MODEL = "llama-3.1-8b-instant"    # fast — YES/NO only
# ANSWERER_MODEL   = "llama-3.3-70b-versatile"  # strong — final answer


# def _get_client() -> Groq:
#     api_key = os.getenv("GROQ_API_KEY")
#     if not api_key:
#         raise ValueError("GROQ_API_KEY not set.")
#     return Groq(api_key=api_key)


# # ── Stage 1: Classifier ───────────────────────────────────────────────────────

# CLASSIFIER_PROMPT = """You are a query classifier. Decide if a question needs a live camera image to answer.
# Reply with exactly one word: YES or NO.

# Reply YES if the question asks about:
# - what something looks like (color, shape, size, style)
# - what a person is wearing, doing, or looks like
# - objects, people, or text visible in the room
# - the environment, background, or surroundings
# - whether something is present or absent visually

# Reply NO if the question is about:
# - general knowledge, facts, history, science, math
# - definitions, explanations, jokes, greetings, opinions
# - anything that doesn't require seeing the physical environment

# Examples:
# Q: describe my surroundings → YES
# Q: what color is my shirt → YES
# Q: do I have a beard → YES
# Q: how many doors are in the room → YES
# Q: what am I holding → YES
# Q: who is the prime minister of india → NO
# Q: what is python → NO
# Q: tell me a joke → NO
# Q: hi → NO

# Reply with only YES or NO."""


# def _needs_vision(query: str, scene: dict) -> bool:
#     """Returns True if query requires the camera. Defaults to False on error."""
#     scene_keywords = ["motion", "movement", "moving", "how many people", "anyone in the room"]
#     if any(kw in query.lower() for kw in scene_keywords):
#         return False

#     try:
#         client = _get_client()
#         response = client.chat.completions.create(
#             model=CLASSIFIER_MODEL,
#             messages=[
#                 {"role": "system", "content": CLASSIFIER_PROMPT},
#                 {"role": "user", "content": f"Q: {query}"},
#             ],
#             temperature=0,
#             max_tokens=5,
#         )
#         answer = response.choices[0].message.content.strip().upper()
#         needs = answer.startswith("YES")
#         logger.info("Classifier: '%s' → %s", query, "YES" if needs else "NO")
#         return needs
#     except Exception as e:
#         logger.warning("Classifier failed, defaulting to text path: %s", e)
#         return False


# # ── Stage 2: Synthesizer ──────────────────────────────────────────────────────

# DORA_SYSTEM = """You are Dora — a witty, warm, and perceptive AI assistant.
# Be conversational and direct. Never robotic. Keep answers concise unless detail is asked for.
# When you have visual information, describe it naturally as if you can see it yourself.
# Never mention "tools", "functions", or "camera analysis" — just respond as Dora.
# IMPORTANT: Never mention people count, motion detection, or any scene statistics
# unless the user specifically asked about them. Just answer the question naturally."""


# def _synthesize(user_query: str, scene: dict, visual_context: str | None = None) -> str:
#     """Generate the final answer using 70b model."""

#     if visual_context:
#         # Vision path: camera result drives the answer
#         user_content = (
#             f"[Camera sees]: {visual_context}\n\n"
#             f"User asked: {user_query}\n"
#             f"Answer naturally using the camera info above."
#         )
#     else:
#         # Text path: pure knowledge — no scene data injected so it never leaks
#         # Only inject scene if the user explicitly asked about presence/motion
#         query_lower = user_query.lower()
#         asked_about_scene = any(
#             kw in query_lower
#             for kw in ["how many people", "anyone", "someone", "motion", "movement", "moving"]
#         )
#         if asked_about_scene:
#             motion_str = "Yes" if scene["motion"] else "No"
#             scene_note = f"[Scene: {scene['people']} person(s) visible, motion={motion_str}]\n"
#             user_content = f"{scene_note}User: {user_query}"
#         else:
#             user_content = f"User: {user_query}"

#     try:
#         client = _get_client()
#         response = client.chat.completions.create(
#             model=ANSWERER_MODEL,
#             messages=[
#                 {"role": "system", "content": DORA_SYSTEM},
#                 {"role": "user", "content": user_content},
#             ],
#             temperature=0.7,
#             max_tokens=512,
#         )
#         return response.choices[0].message.content.strip()
#     except Exception as e:
#         logger.error("Synthesizer failed: %s", e)
#         return f"Sorry, I hit an error: {e}"


# # ── Public API ────────────────────────────────────────────────────────────────

# def ask_agent(user_query: str) -> str:
#     """
#     Two-stage pipeline:
#       1. Classify: needs vision?
#       2a. YES → call vision tool once → synthesize with visual context
#       2b. NO  → synthesize from knowledge only (scene data only if relevant)
#     """
#     scene = get_scene()

#     if _needs_vision(user_query, scene):
#         logger.info("Vision path: %s", user_query)
#         try:
#             vision_query = (
#                 f"Answer this about what you see: {user_query}. "
#                 "Include all relevant visual details."
#             )
#             visual_result = analyze_image_with_query.invoke({"query": vision_query})
#             logger.info("Vision result preview: %s", str(visual_result)[:120])
#         except Exception as e:
#             logger.error("Vision tool failed: %s", e)
#             visual_result = None
#         return _synthesize(user_query, scene, visual_context=visual_result)
#     else:
#         logger.info("Text path: %s", user_query)
#         return _synthesize(user_query, scene, visual_context=None)


###########################################################################final########################
"""
ai_agent.py
-----------
Two-stage pipeline: Classifier → Vision or Text path.
Max 2 LLM calls per response. Zero looping. Zero recursion issues.
"""

import os
import logging
from groq import Groq
from dotenv import load_dotenv

from tools import analyze_image_with_query
from scene_memory import get_scene

load_dotenv()
logger = logging.getLogger(__name__)

CLASSIFIER_MODEL = "llama-3.1-8b-instant"    # fast — YES/NO only
ANSWERER_MODEL   = "llama-3.3-70b-versatile"  # strong — final answer


def _get_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set.")
    return Groq(api_key=api_key)


# ── Stage 1: Classifier ───────────────────────────────────────────────────────

CLASSIFIER_PROMPT = """You are a query classifier. Decide if a question needs a live camera image to answer.
Reply with exactly one word: YES or NO.

Reply YES if the question asks about:
- what something looks like (color, shape, size, style)
- what a person is wearing, doing, or looks like
- objects, people, or text visible in the room
- the environment, background, or surroundings
- whether something is present or absent visually
- reading something visible (time on a clock/phone, text on a sign, number on a board)
- anything physically present in the scene that requires looking

IMPORTANT — these are YES even though they sound like general questions:
Q: what time is it on the phone → YES  (needs to read the screen)
Q: what time does the clock show → YES  (needs to read the clock)
Q: what is written on the board → YES  (needs to read text)
Q: what does the screen say → YES  (needs to read the screen)
Q: what number is on the jersey → YES  (needs to read a number)
Q: what brand is that → YES  (needs to read a label)
Q: is the TV on → YES  (needs to see the TV)
Q: what color is the light → YES  (needs to see the light)

Reply NO if the question is about:
- general knowledge, facts, history, science, math
- definitions, explanations, jokes, greetings, opinions
- anything that doesn't require seeing the physical environment

More examples:
Q: describe my surroundings → YES
Q: what color is my shirt → YES
Q: do I have a beard → YES
Q: how many doors are in the room → YES
Q: what am I holding → YES
Q: who is the prime minister of india → NO
Q: what is python → NO
Q: tell me a joke → NO
Q: hi → NO
Q: what is 2+2 → NO
Q: what year was python created → NO

Reply with only YES or NO."""

# Keyword pre-checks: force YES before even calling the classifier LLM
# These phrases almost always mean the user wants to READ something from the camera
FORCE_VISION_PHRASES = [
    "time on", "time does", "what time", "clock say", "clock show",
    "written on", "what does it say", "read the", "what is on the screen",
    "screen say", "screen show", "what brand", "what number", "label say",
    "sign say", "board say", "calendar say",
]

# Keyword pre-checks: force NO — scene data is enough, skip vision
FORCE_TEXT_PHRASES = [
    "how many people", "anyone in the room", "someone in the room",
    "motion", "movement", "moving",
]


def _needs_vision(query: str, scene: dict) -> bool:
    """
    Returns True if query requires the camera. 
    Uses keyword pre-checks first, then falls back to LLM classifier.
    Defaults to False on any error.
    """
    q = query.lower()

    # Fast path: scene data is sufficient
    if any(kw in q for kw in FORCE_TEXT_PHRASES):
        logger.info("Classifier (keyword): '%s' → NO (scene data sufficient)", query)
        return False

    # Fast path: clearly needs vision (reading something from camera)
    if any(kw in q for kw in FORCE_VISION_PHRASES):
        logger.info("Classifier (keyword): '%s' → YES (read/look phrase detected)", query)
        return True

    # LLM classifier for everything else
    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=CLASSIFIER_MODEL,
            messages=[
                {"role": "system", "content": CLASSIFIER_PROMPT},
                {"role": "user", "content": f"Q: {query}"},
            ],
            temperature=0,
            max_tokens=5,
        )
        answer = response.choices[0].message.content.strip().upper()
        needs = answer.startswith("YES")
        logger.info("Classifier (LLM): '%s' → %s", query, "YES" if needs else "NO")
        return needs
    except Exception as e:
        logger.warning("Classifier failed, defaulting to text path: %s", e)
        return False


# ── Stage 2: Synthesizer ──────────────────────────────────────────────────────

PROMPT_SYSTEM = """You are AI AGENT — a witty, warm, and perceptive AI assistant.
Be conversational and direct. Never robotic. Keep answers concise unless detail is asked for.
When you have visual information, describe it naturally as if you can see it yourself.
Never mention "tools", "functions", or "camera analysis" — just respond as Dora.
Never mention people count, motion detection, or scene statistics unless the user specifically asked about them."""


def _synthesize(user_query: str, scene: dict, visual_context: str | None = None) -> str:
    """Generate the final answer using 70b model."""

    if visual_context:
        user_content = (
            f"[Camera sees]: {visual_context}\n\n"
            f"User asked: {user_query}\n"
            f"Answer naturally using the camera info above."
        )
    else:
        query_lower = user_query.lower()
        asked_about_scene = any(
            kw in query_lower
            for kw in ["how many people", "anyone", "someone", "motion", "movement", "moving"]
        )
        if asked_about_scene:
            motion_str = "Yes" if scene["motion"] else "No"
            scene_note = f"[Scene: {scene['people']} person(s) visible, motion={motion_str}]\n"
            user_content = f"{scene_note}User: {user_query}"
        else:
            user_content = f"User: {user_query}"

    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=ANSWERER_MODEL,
            messages=[
                {"role": "system", "content": PROMPT_SYSTEM},
                {"role": "user", "content": user_content},
            ],
            temperature=0.7,
            max_tokens=512,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error("Synthesizer failed: %s", e)
        return f"Sorry, I hit an error: {e}"


# ── Public API ────────────────────────────────────────────────────────────────

def ask_agent(user_query: str) -> str:
    """
    Two-stage pipeline:
      1. Classify: needs vision?
      2a. YES → call vision tool once → synthesize with visual context
      2b. NO  → synthesize from knowledge only
    """
    scene = get_scene()

    if _needs_vision(user_query, scene):
        logger.info("Vision path: %s", user_query)
        try:
            vision_query = (
                f"Answer this about what you see: {user_query}. "
                "Include all relevant visual details."
            )
            visual_result = analyze_image_with_query.invoke({"query": vision_query})
            logger.info("Vision result preview: %s", str(visual_result)[:120])
        except Exception as e:
            logger.error("Vision tool failed: %s", e)
            visual_result = None
        return _synthesize(user_query, scene, visual_context=visual_result)
    else:
        logger.info("Text path: %s", user_query)
        return _synthesize(user_query, scene, visual_context=None)

# """
# ai_agent.py
# -----------
# LangGraph ReAct agent with TWO tools and hard one-call enforcement.

# THE CORE PROBLEM:
#   The 70b model ignores prompt instructions like "call each tool at most once".
#   Under LangGraph's ReAct loop, when a tool returns a result the model keeps
#   rephrasing and retrying — burning rate limits and hitting recursion limits.

# THE SOLUTION — code-level enforcement, not prompt-level:
#   Each tool is wrapped in a stateful callable that raises an exception
#   after the first call within a single agent turn. The agent physically
#   cannot call the tool a second time — it hits an exception and must
#   synthesize its answer from the first result.

#   This is the only reliable approach. Prompts are suggestions to LLMs.
#   Code is law.

# TWO TOOLS:
#   Tool 1: get_scene_summary()         → zero API cost, presence/motion
#   Tool 2: analyze_image_with_query()  → one vision API call, visual detail
# """

# import os
# import logging
# from functools import wraps
# from langchain_groq import ChatGroq
# from langchain_core.tools import tool
# from langgraph.prebuilt import create_react_agent
# from dotenv import load_dotenv

# from scene_memory import get_scene
# import cv2
# import base64
# from groq import Groq
# from camera_stream import camera_stream

# load_dotenv()
# logger = logging.getLogger(__name__)

# VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"


# # ── One-call enforcement ───────────────────────────────────────────────────────
# # This class wraps a tool function and raises after the first call.
# # Reset it at the start of each ask_agent() call.

# class _OnceGuard:
#     """Allows a function to be called exactly once. Raises on subsequent calls."""

#     def __init__(self, name: str):
#         self.name = name
#         self._called = False

#     def reset(self):
#         self._called = False

#     def check(self):
#         if self._called:
#             raise RuntimeError(
#                 f"Tool '{self.name}' was already called this turn. "
#                 "You have the result — use it to answer now. Do not call this tool again."
#             )
#         self._called = True


# _scene_guard  = _OnceGuard("get_scene_summary")
# _vision_guard = _OnceGuard("analyze_image_with_query")


# # ── Tool 1 — Scene summary (zero API cost) ────────────────────────────────────

# @tool
# def get_scene_summary() -> str:
#     """
#     Returns real-time scene data from local sensors: people count and motion status.
#     INSTANT — zero API cost.

#     Use for: how many people, is anyone present, is there movement/motion.
#     Do NOT use for: appearance, clothing, objects, reading text — use analyze_image_with_query.
#     """
#     _scene_guard.check()

#     scene = get_scene()
#     motion_str = "Yes" if scene["motion"] else "No"
#     people = scene["people"]

#     logger.info("get_scene_summary called → people=%d, motion=%s", people, motion_str)

#     if people == 0:
#         presence = "No people are currently visible in the camera view."
#     elif people == 1:
#         presence = "There is 1 person currently visible in the camera view."
#     else:
#         presence = f"There are {people} people currently visible in the camera view."

#     motion = (
#         "Movement is currently detected in the scene."
#         if scene["motion"]
#         else "No movement is currently detected in the scene."
#     )
#     return f"{presence} {motion}"


# # ── Tool 2 — Vision analysis (one API call) ───────────────────────────────────

# def _capture_image_b64(size: int = 512) -> str:
#     frame = camera_stream.get_frame()
#     if frame is None:
#         raise RuntimeError("Camera frame unavailable.")
#     frame = cv2.resize(frame, (size, size))
#     _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
#     return base64.b64encode(buffer).decode("utf-8")


# @tool
# def analyze_image_with_query(query: str) -> str:
#     """
#     Captures the live webcam frame and answers a visual question about it.
#     Makes ONE vision API call — takes ~1-2 seconds.

#     Use for: appearance, clothing, hair, objects, surroundings, reading text/time,
#     emotions, counting items, describing the scene in detail.

#     Do NOT use for: presence/motion questions — use get_scene_summary instead.
#     Do NOT use for: general knowledge questions — answer directly.
#     """
#     _vision_guard.check()

#     logger.info("analyze_image_with_query called with query: %s", query)

#     api_key = os.getenv("GROQ_API_KEY")
#     if not api_key:
#         raise RuntimeError("GROQ_API_KEY not set.")

#     try:
#         img_b64 = _capture_image_b64()
#     except RuntimeError as e:
#         raise RuntimeError(f"Camera error: {e}") from e

#     client = Groq(api_key=api_key)

#     try:
#         completion = client.chat.completions.create(
#             model=VISION_MODEL,
#             messages=[{
#                 "role": "user",
#                 "content": [
#                     {"type": "text", "text": query},
#                     {"type": "image_url", "image_url": {
#                         "url": f"data:image/jpeg;base64,{img_b64}"
#                     }},
#                 ],
#             }],
#             temperature=0,
#             max_tokens=512,
#         )
#         return completion.choices[0].message.content

#     except Exception as e:
#         logger.error("Vision API call failed: %s", e)
#         raise RuntimeError(f"Vision analysis failed: {e}") from e


# # ── LLM ───────────────────────────────────────────────────────────────────────

# llm = ChatGroq(
#     model="llama-3.3-70b-versatile",
#     temperature=0.7,
#     groq_api_key=os.getenv("GROQ_API_KEY"),
# )

# # ── System Prompt ──────────────────────────────────────────────────────────────

# SYSTEM_PROMPT = """You are Dora — a witty, warm, and perceptive AI assistant with a live webcam and scene sensors.

# You have exactly TWO tools:

#   1. get_scene_summary()
#      - Instant, zero cost
#      - Use ONLY for: people count, presence, motion detection

#   2. analyze_image_with_query(query)
#      - Vision API call, ~1-2 seconds
#      - Use for: appearance, clothing, objects, emotions, surroundings, reading text

# Use NO tool for general knowledge, facts, math, greetings — answer directly.

# IMPORTANT: Each tool can only be called ONCE per response.
# After calling a tool and receiving its result, you MUST answer the user immediately.
# The system will raise an error if you try to call a tool a second time.

# Be witty, warm, and concise. Never robotic. Never mention tools."""

# # ── Agent ──────────────────────────────────────────────────────────────────────

# agent = create_react_agent(
#     model=llm,
#     tools=[get_scene_summary, analyze_image_with_query],
#     prompt=SYSTEM_PROMPT,
# )


# # ── Public API ────────────────────────────────────────────────────────────────

# def ask_agent(user_query: str) -> str:
#     """
#     Send a query to the two-tool ReAct agent.
#     Guards are reset before each call — each turn gets exactly one tool use per tool.
#     """
#     logger.info("ask_agent called: %s", user_query)

#     # Reset both guards — fresh slate for this turn
#     _scene_guard.reset()
#     _vision_guard.reset()

#     try:
#         response = agent.invoke(
#             {"messages": [{"role": "user", "content": user_query}]},
#             config={"recursion_limit": 15},
#         )
#         return response["messages"][-1].content

#     except Exception as e:
#         logger.error("Agent invocation failed: %s", e)
#         return f"Sorry, I ran into an error: {e}"