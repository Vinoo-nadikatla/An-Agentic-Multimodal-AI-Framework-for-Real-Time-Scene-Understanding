"""
agent/intent_router.py
Zero-cost regex intent classification.
No API calls. No tokens. No latency.
"""
from __future__ import annotations
import logging
import re

logger = logging.getLogger(__name__)

_SCENE_RE = re.compile(
    r"how.many|count.*people|number.of.people|kitne.log|entha.mandi|"
    r"is.there.*anyone|are.there.*people|are.there.*worker|anyone.in.the.room|"
    r"any.workers|workers.present|workers.in.the|how.many.workers|"
    r"motion|movement|moving|suspicious|halchal|"
    r"what.happened|activity.log|last.hour|last.30|"
    r"timeline|history|report|summary.of|"
    r"ఎంత.మంది|ఎవరైనా|కదలిక",
    re.IGNORECASE,
)

_OCR_RE = re.compile(
    r"what.time|what.does.it.say|what.is.written|"
    r"read.the|read.this|what.number|clock.show|watch.show|"
    r"label.say|sign.say|board.say|"
    r"సమయం|గడియారం|రాసింది",
    re.IGNORECASE,
)

_VISION_RE = re.compile(
    r"what.do.you.see|describe|surroundings|background|"
    r"what.am.i|what.are.you|what.is.in.the|who.is.in|"
    r"wearing|holding|doing|color|colour|"
    r"helmet|hard.hat|vest|gloves|mask|"
    r"wearing.*ppe|check.*ppe|ppe.*on|ppe.*violation|ppe.*complian|"
    r"safety.violation|safety.check|unsafe|danger|hazard|violation|"
    r"near.machine|near.equipment|"
    r"చూడు|వివరించు|ఏమి.కనిపిస్తోంది|"
    r"kya.dikh|batao|dekho",
    re.IGNORECASE,
)


def classify(query: str) -> str:
    """
    Returns: general | scene | vision_describe | vision_ocr
    Zero API cost. Zero tokens. Zero latency.
    Falls back to general for anything unknown.
    """
    q = query.strip()

    if _SCENE_RE.search(q):
        logger.info("Intent: scene for: %.50s", q)
        return "scene"

    if _OCR_RE.search(q):
        logger.info("Intent: vision_ocr for: %.50s", q)
        return "vision_ocr"

    if _VISION_RE.search(q):
        logger.info("Intent: vision_describe for: %.50s", q)
        return "vision_describe"

    logger.info("Intent: general for: %.50s", q)
    return "general"