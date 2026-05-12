"""
services/activity_log.py
Thread-safe activity log for safety monitoring.
Records timestamped events triggered by scene changes.
"""
import threading
from collections import deque
from datetime import datetime, timedelta

_lock = threading.Lock()
_log: deque = deque(maxlen=200)


def add_entry(people: int, motion: bool, description: str = "", ppe_workers: list = None) -> None:
    now = datetime.now()
    entry = {
        "time": now,
        "time_str": now.strftime("%I:%M %p"),
        "people": people,
        "motion": motion,
        "description": description,
        "ppe_workers": ppe_workers or [],
    }
    with _lock:
        _log.append(entry)


def get_recent(minutes: int = 60) -> list:
    cutoff = datetime.now() - timedelta(minutes=minutes)
    with _lock:
        return [e for e in _log if e["time"] >= cutoff]


def get_summary(minutes: int = 60) -> str:
    cutoff = datetime.now() - timedelta(minutes=minutes)
    with _lock:
        recent = [e for e in _log if e["time"] >= cutoff]

    if not recent:
        return "No activity recorded in the last hour."

    lines = [f"Activity summary — last {minutes} minutes:"]
    lines.append(f"- Snapshots recorded: {len(recent)}")

    with_people = [e for e in recent if e["people"] > 0]
    if with_people:
        lines.append(f"- Workers detected: {len(with_people)} of {len(recent)} snapshots")
        lines.append(f"- First seen: {with_people[0]['time_str']}, Last seen: {with_people[-1]['time_str']}")

    motion_count = sum(1 for e in recent if e["motion"])
    lines.append(f"- Movement detected in {motion_count} snapshots")

    # PPE violation timeline — only emit a line when a worker's state changes
    ppe_events = [e for e in recent if e.get("ppe_workers")]
    if ppe_events:
        lines.append("\nPPE Timeline:")
        seen_states: dict = {}
        for e in ppe_events:
            for w in e["ppe_workers"]:
                wid = w["label"]
                state = tuple(sorted(w.get("violations", [])))
                if seen_states.get(wid) != state:
                    seen_states[wid] = state
                    if state:
                        lines.append(f"  {e['time_str']} - {wid}: {', '.join(state)}")
                    else:
                        lines.append(f"  {e['time_str']} - {wid}: Compliant")

    descs = [e["description"] for e in recent if e.get("description")]
    if descs:
        lines.append("\nSafety observations:")
        for d in set(descs):
            lines.append(f"  - {d}")

    return "\n".join(lines)


def get_all() -> list:
    with _lock:
        return list(_log)
