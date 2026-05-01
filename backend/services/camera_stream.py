# # import cv2
# # import threading

# # class CameraStream:
# #     def __init__(self, src=0):
# #         self.cap = cv2.VideoCapture(src)
# #         self.frame = None
# #         self.running = True

# #         thread = threading.Thread(target=self.update, daemon=True)
# #         thread.start()

# #     def update(self):
# #         while self.running:
# #             ret, frame = self.cap.read()
# #             if ret:
# #                 self.frame = frame

# #     def get_frame(self):
# #         return self.frame

# #     def stop(self):
# #         self.running = False
# #         self.cap.release()


# # camera = CameraStream()

# import cv2
# import threading

# class CameraManager:
#     def __init__(self, index=0):
#         self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
#         self.frame = None
#         self.running = False
#         self.lock = threading.Lock()

#     def start(self):
#         if self.running:
#             return
#         self.running = True
#         thread = threading.Thread(target=self.update, daemon=True)
#         thread.start()

#     def update(self):
#         while self.running:
#             ret, frame = self.cap.read()
#             if ret:
#                 with self.lock:
#                     self.frame = frame

#     def get_frame(self):
#         with self.lock:
#             return self.frame

#     def stop(self):
#         self.running = False
#         if self.cap:
#             self.cap.release()


# camera_manager = CameraManager()

# import cv2
# import threading
# import time

# class CameraStream:

#     def __init__(self,index=0):
#         self.cap=cv2.VideoCapture(index,cv2.CAP_DSHOW)
#         self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
#         self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

#         self.frame=None
#         self.running=False
#         self.lock=threading.Lock()

#     def start(self):

#         if self.running:
#             return

#         self.running=True
#         thread=threading.Thread(target=self.update,daemon=True)
#         thread.start()

#     def update(self):

#         while self.running:

#             ret,frame=self.cap.read()

#             if ret:
#                 with self.lock:
#                     self.frame=frame
#             time.sleep(0.01)
            
#     def get_frame(self):

#         with self.lock:
#             return self.frame

#     def stop(self):

#         self.running=False
#         self.cap.release()


# camera_stream=CameraStream()

"""
camera_stream.py
----------------
Thread-safe continuous webcam capture.
The stream runs in a background daemon thread and never blocks the main thread.
Auto-reconnects if the camera is lost.
"""

import cv2
import threading
import time
import logging

logger = logging.getLogger(__name__)


class CameraStream:
    """
    Non-blocking webcam reader with auto-reconnect.

    Usage:
        stream = CameraStream(index=0)
        stream.start()
        frame = stream.get_frame()   # latest BGR frame or None
        stream.stop()
    """

    def __init__(self, index: int = 0, width: int = 640, height: int = 480):
        self.index = index
        self.width = width
        self.height = height

        self.cap: cv2.VideoCapture | None = None
        self.frame = None
        self.running = False
        self.lock = threading.Lock()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self.running:
            return

        self.cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # minimise latency

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {self.index}")

        self.running = True
        self._thread = threading.Thread(target=self._update, daemon=True, name="CameraStream")
        self._thread.start()
        logger.info("CameraStream started on index %d", self.index)

    def stop(self) -> None:
        self.running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()
        logger.info("CameraStream stopped.")

    # ------------------------------------------------------------------
    # Internal loop
    # ------------------------------------------------------------------

    def _update(self) -> None:
        consecutive_failures = 0

        while self.running:
            if self.cap is None or not self.cap.isOpened():
                logger.warning("Camera lost — attempting reconnect…")
                self._reconnect()
                continue

            ret, frame = self.cap.read()

            if ret:
                consecutive_failures = 0
                with self.lock:
                    self.frame = frame
            else:
                consecutive_failures += 1
                logger.warning("Frame read failed (%d consecutive)", consecutive_failures)

                if consecutive_failures >= 10:
                    logger.error("Too many failures — reconnecting camera.")
                    self._reconnect()
                    consecutive_failures = 0

            time.sleep(0.01)  # ~100 fps cap; actual rate limited by camera

    def _reconnect(self) -> None:
        if self.cap:
            self.cap.release()
        time.sleep(1.0)
        self.cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_frame(self):
        """Return the most recent BGR frame, or None if not yet available."""
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    @property
    def is_ready(self) -> bool:
        with self.lock:
            return self.frame is not None


# Module-level singleton
camera_stream = CameraStream()
def get_recent_frames(self, count: int = 5, interval_ms: int = 400) -> list:
        """Capture multiple frames over time for video understanding."""
        import time
        frames = []
        for _ in range(count):
            frame = self.get_frame()
            if frame is not None:
                frames.append(frame)
            time.sleep(interval_ms / 1000)
        return frames

# Module-level singleton
camera_stream = CameraStream()   
