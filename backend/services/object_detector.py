# # from ultralytics import YOLO
# # from services.camera_stream import camera_stream

# # model=YOLO("yolov8n.pt")

# # def detect_people():

# #     frame=camera_stream.get_frame()

# #     if frame is None:
# #         return 0

# #     results=model(frame)

# #     count=0

# #     for r in results:

# #         for box in r.boxes:

# #             cls=int(box.cls[0])

# #             if cls==0:
# #                 count+=1

# #     return count

# from ultralytics import YOLO
# from services.camera_stream import camera_stream

# model = YOLO("yolov8n.pt")

# def detect_people():

#     frame = camera_stream.get_frame()

#     if frame is None:
#         return 0

#     results = model(frame, verbose=False)

#     count = 0

#     for r in results:

#         for box in r.boxes:

#             cls = int(box.cls[0])

#             if cls == 0:
#                 count += 1

#     return count

"""
object_detector.py
------------------
YOLOv8-based person detector.
YOLO class 0 = 'person' in the COCO dataset.
"""

import logging
from ultralytics import YOLO
from services.camera_stream import camera_stream

logger = logging.getLogger(__name__)

# Load model once at import time — avoids re-loading on every call
_model = YOLO("yolov8n.pt")

# Confidence threshold — adjust as needed
CONFIDENCE_THRESHOLD = 0.4


def detect_people() -> int:
    """
    Returns the count of people detected in the current camera frame.
    Returns 0 if the frame is unavailable or no people are found.
    """
    frame = camera_stream.get_frame()
    if frame is None:
        return 0

    try:
        results = _model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)
    except Exception as e:
        logger.error("YOLO inference failed: %s", e)
        return 0

    count = sum(
        1
        for r in results
        for box in r.boxes
        if int(box.cls[0]) == 0  # class 0 = person
    )

    return count
