# import cv2
# import numpy as np
# from camera_stream import camera_stream

# previous_frame=None

# def detect_motion():

#     global previous_frame

#     frame=camera_stream.get_frame()

#     if frame is None:
#         return False

#     gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#     gray=cv2.GaussianBlur(gray,(21,21),0)

#     if previous_frame is None:
#         previous_frame=gray
#         return False

#     frame_delta=cv2.absdiff(previous_frame,gray)

#     thresh=cv2.threshold(frame_delta,25,255,cv2.THRESH_BINARY)[1]

#     motion_pixels=np.sum(thresh)

#     previous_frame=gray

#     return motion_pixels>50000

"""
motion_detector.py
------------------
Frame-differencing motion detector.
Compares consecutive greyscale frames to detect pixel-level movement.
"""

import cv2
import numpy as np
from camera_stream import camera_stream

# Module-level state — previous frame retained between calls
_previous_frame = None

# Tune this threshold for your environment.
# Higher = less sensitive (ignores minor lighting changes).
# Lower  = more sensitive (detects subtle movement).
MOTION_PIXEL_THRESHOLD = 50_000


def detect_motion() -> bool:
    """
    Returns True if significant motion is detected between the last two frames.
    """
    global _previous_frame

    frame = camera_stream.get_frame()
    if frame is None:
        return False

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if _previous_frame is None:
        _previous_frame = gray
        return False

    delta = cv2.absdiff(_previous_frame, gray)
    _, thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)

    motion_pixels = int(np.sum(thresh))
    _previous_frame = gray

    return motion_pixels > MOTION_PIXEL_THRESHOLD
