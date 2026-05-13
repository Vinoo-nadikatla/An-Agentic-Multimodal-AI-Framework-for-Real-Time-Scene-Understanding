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
    r"is.there.*anyone|are.there.*people|anyone.in.the.room|"
    r"motion|movement|moving|halchal|"
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
    r"am.i.wearing|do.i.have|do.i.look|i.*look.like|"
    r"what.*wearing|what.*color|what.*colour|"
    r"shirt|tshirt|t-shirt|dress|outfit|clothes|"
    r"suits|look.good|look.nice|fits.her|fits.him|"
    r"women.*wearing|man.*wearing|person.*wearing|"
    r"my.appearance|my.outfit|my.expression|"
    r"color.of|colour.of|holding|doing|"
    r"see.right.now|look.right.now|"
    r"suspicious|behaviour|behavior|"
    r"what.*see|do.you.see|can.you.see|"
    r"activity|what.*happen|going.on|"
    r"చూడు|వివరించు|ఏమి.కనిపిస్తోంది|నా.వెనక|"
    r"kya.dikh|batao|rang|dekho",
    re.IGNORECASE,
)

# Queries containing "suspicious" always need the camera, even if they also
# match scene terms like "movement" or "motion".
_SUSPICIOUS_RE = re.compile(r"\bsuspicious\b", re.IGNORECASE)


def classify(query: str) -> str:
    """
    Returns: general | scene | vision_describe | vision_ocr
    Zero API cost. Zero tokens. Zero latency.
    Falls back to general for anything unknown.
    """
    q = query.strip()

    # suspicious beats all other matches — always needs a live camera look
    if _SUSPICIOUS_RE.search(q):
        logger.info("Intent: vision_describe (suspicious) for: %.50s", q)
        return "vision_describe"

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