"""
services/perception_loop.py
---------------------------
Background thread: reads camera frames, runs motion + person detection,
writes results to scene_memory.

Only started when server-side camera is available (local / Pi deployments).
Cloud deployments skip this entirely — the browser sends frames directly.
"""

import logging
import threading
import time

logger = logging.getLogger(__name__)

_thread: threading.Thread | None = None
_running = False


def _loop() -> None:
    from services.motion_detector import detect_motion
    from services.object_detector import detect_people
    from services.scene_memory import update_scene

    logger.info("Perception loop started.")
    while _running:
        try:
            motion = detect_motion()
            people = detect_people()
            update_scene(people=people, motion=motion)
        except Exception as e:
            logger.error("Perception loop error: %s", e)
        time.sleep(0.5)  # 2 Hz — sufficient for occupancy detection

    logger.info("Perception loop stopped.")


def start_perception_loop() -> None:
    global _thread, _running
    if _running:
        return
    _running = True
    _thread = threading.Thread(target=_loop, daemon=True, name="PerceptionLoop")
    _thread.start()


def stop_perception_loop() -> None:
    global _running
    _running = False
    if _thread:
        _thread.join(timeout=3.0)
