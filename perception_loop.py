# import threading
# import time

# from motion_detector import detect_motion
# from object_detector import detect_people
# from scene_memory import update_scene


# def perception_loop():

#     while True:

#         people=detect_people()

#         motion=detect_motion()

#         update_scene(people,motion)

#         time.sleep(0.5)


# def start_perception():

#     thread=threading.Thread(target=perception_loop,daemon=True)

#     thread.start()

"""
perception_loop.py
------------------
Background thread that continuously updates scene memory.
Runs at ~2 Hz (every 0.5s) — fast enough for awareness, slow enough
to not saturate CPU with YOLO inference.
"""

import threading
import time
import logging

from motion_detector import detect_motion
from object_detector import detect_people
from scene_memory import update_scene

logger = logging.getLogger(__name__)

PERCEPTION_INTERVAL = 0.5  # seconds between perception cycles


def _perception_loop() -> None:
    logger.info("Perception loop started.")

    while True:
        try:
            people = detect_people()
            motion = detect_motion()
            update_scene(people, motion)
        except Exception as e:
            logger.error("Perception loop error: %s", e)

        time.sleep(PERCEPTION_INTERVAL)


def start_perception() -> None:
    """Start the background perception thread. Call once at startup."""
    thread = threading.Thread(target=_perception_loop, daemon=True, name="PerceptionLoop")
    thread.start()
    logger.info("Perception thread launched.")
