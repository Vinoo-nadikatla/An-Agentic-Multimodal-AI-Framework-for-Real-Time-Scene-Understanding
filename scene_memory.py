# import time

# scene_state={

#     "people":0,
#     "motion":False,
#     "last_update":None
# }

# def update_scene(people,motion):

#     scene_state["people"]=people
#     scene_state["motion"]=motion
#     scene_state["last_update"] = time.time()

# def get_scene():

#     return scene_state

"""
scene_memory.py
---------------
Thread-safe shared memory for scene state.
Written by the perception thread, read by the agent thread.
"""

import time
import threading

_lock = threading.Lock()

_scene_state: dict = {
    "people": 0,
    "motion": False,
    "last_update": None,
}


def update_scene(people: int, motion: bool) -> None:
    """Called by perception loop — update latest scene state."""
    with _lock:
        _scene_state["people"] = people
        _scene_state["motion"] = motion
        _scene_state["last_update"] = time.time()


def get_scene() -> dict:
    """Called by agent — returns a snapshot copy of the scene state."""
    with _lock:
        return dict(_scene_state)