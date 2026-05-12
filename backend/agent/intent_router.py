"""
agent/intent_router.py
Zero-cost regex intent classification.
No API calls. No tokens. No latency.
"""
from __future__ import annotations
import logging
import re

logger = logging.getLogger(__name__)

_REPORT_RE = re.compile(
    r"\b(generate|create|make|produce|write|show)"
    r".*\b(report|summary|daily|log|record)\b"
    r"|\b(safety report|daily report|shift report|incident report)\b",
    re.IGNORECASE,
)

_SCENE_RE = re.compile(
    r"how.many|count.*people|number.of.people|kitne.log|entha.mandi|"
    r"is.there.*anyone|are.there.*people|are.there.*worker|anyone.in.the.room|"
    r"any.workers|workers.present|workers.in.the|how.many.workers|"
    r"motion|movement|moving|suspicious|halchal|"
    r"what.happened|activity.log|last.hour|last.30|"
    r"timeline|history|summary.of|"
    # Hindi scene keywords (कोई omitted — matches vision_describe test case)
    r"कितने|मजदूर|कार्यकर्ता|गतिविधि|"
    # Telugu scene keywords (dot used as separator — \b unreliable with combining marks)
    r"ఎంత.మంది|కార్మికులు|ఎవరైనా|కదలిక",
    re.IGNORECASE | re.UNICODE,
)

_GENERAL_KNOWLEDGE_RE = re.compile(
    r"^what is (ppe|personal protective|osha|safety regulation|hard hat|helmet definition)"
    r"|^(define|explain|tell me about) (ppe|safety|osha)"
    r"|^why (are|is|do) (helmet|ppe|safety)",
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
    r"\b(helmet|hardhat|hard.hat|ppe|vest|glove|compliance"
    r"|wearing|dressed|equipped|protective|hazard|violation"
    r"|describe|look|see|show|visible|appear|check|inspect"
    r"|safety.hazard|unsafe|danger)\b"
    r"|what.do.you.see|surroundings|background"
    r"|what.am.i|what.are.you|what.is.in.the|who.is.in"
    r"|holding|doing|color|colour"
    r"|wearing.*ppe|check.*ppe|ppe.*on|ppe.*violation|ppe.*complian"
    r"|safety.violation|safety.check"
    r"|near.machine|near.equipment"
    # Telugu safety keywords
    r"|హెల్మెట్|వెస్ట్|భద్రత|ధరించారా"
    # Hindi safety keywords
    r"|हेलमेट|वेस्ट|सुरक्षा|पहने|पीपीई"
    r"|చూడు|వివరించు|ఏమి.కనిపిస్తోంది"
    r"|kya.dikh|batao|dekho",
    re.IGNORECASE | re.UNICODE,
)


def classify(query: str) -> str:
    """
    Returns: general | scene | vision_describe | vision_ocr | report
    Zero API cost. Zero tokens. Zero latency.
    Falls back to general for anything unknown.
    """
    q = query.strip()

    if _REPORT_RE.search(q):
        logger.info("Intent: report for: %.50s", q)
        return "report"

    if _GENERAL_KNOWLEDGE_RE.search(q):
        logger.info("Intent: general (knowledge) for: %.50s", q)
        return "general"

    if _SCENE_RE.search(q):
        logger.info("Intent: scene for: %.50s", q)
        return "scene"

    if _VISION_RE.search(q):
        logger.info("Intent: vision_describe for: %.50s", q)
        return "vision_describe"

    if _OCR_RE.search(q):
        logger.info("Intent: vision_ocr for: %.50s", q)
        return "vision_ocr"

    logger.info("Intent: general for: %.50s", q)
    return "general"