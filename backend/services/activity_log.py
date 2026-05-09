"""
services/activity_log.py
Thread-safe activity log for safety monitoring.
Records timestamped events triggered by scene changes.
"""
import threading
import time
from datetime import datetime
from collections import deque

_lock = threading.Lock()
_log: deque = deque(maxlen=200)


def add_entry(people: int, motion: bool, description: str = "") -> None:
    entry = {
        "timestamp": time.time(),
        "time_str": datetime.now().strftime("%I:%M %p"),
        "date_str": datetime.now().strftime("%Y-%m-%d"),
        "people": people,
        "motion": motion,
        "description": description,
    }
    with _lock:
        _log.append(entry)


def get_recent(minutes: int = 60) -> list:
    cutoff = time.time() - (minutes * 60)
    with _lock:
        return [e for e in _log if e["timestamp"] >= cutoff]


def get_summary(minutes: int = 60) -> str:
    entries = get_recent(minutes)
    if not entries:
        return "No activity recorded in this period."
    total = len(entries)
    with_people = [e for e in entries if e["people"] > 0]
    with_motion = [e for e in entries if e["motion"]]
    lines = [f"Activity summary — last {minutes} minutes:"]
    if with_people:
        lines.append(f"- Workers detected: {len(with_people)} of {total} snapshots")
        lines.append(f"- First seen: {with_people[0]['time_str']}, Last seen: {with_people[-1]['time_str']}")
    else:
        lines.append("- No workers detected in this period")
    if with_motion:
        lines.append(f"- Movement detected in {len(with_motion)} snapshots")
    descriptions = [e["description"] for e in entries if e["description"]]
    if descriptions:
        lines.append("Safety observations:")
        for d in descriptions[-5:]:
            lines.append(f"  • {d}")
    return "\n".join(lines)


def get_all() -> list:
    with _lock:
        return list(_log)
