"""
services/perception_loop.py
---------------------------
Two-thread pipeline:
  CameraReader  — reads frames at ~15 fps, updates raw cache immediately, queues
                  every 8th frame for detection. Scene/motion logged at ~2 Hz.
  DetectionThread — consumes queued frames, runs YOLO, updates annotated cache.

Camera display is never blocked by YOLO because they run in separate threads.
"""

import logging
import queue
import threading
import time

logger = logging.getLogger(__name__)

_thread_cam: threading.Thread | None = None
_thread_det: threading.Thread | None = None
_running = False
_frame_queue: queue.Queue = queue.Queue(maxsize=2)


def _camera_reader_loop() -> None:
    from services.motion_detector import detect_motion
    from services.object_detector import detect_people
    from services.scene_memory import update_scene
    from services.activity_log import add_entry
    from services.ppe_detector import ppe_detector, get_ppe_status, update_raw_frame
    from services.camera_stream import camera_stream

    logger.info("Camera reader thread started.")
    prev_people: int | None = None
    prev_motion: bool | None = None
    frame_count = 0
    scene_tick = 0

    while _running:
        frame = camera_stream.get_frame()
        if frame is None:
            time.sleep(0.01)
            continue

        # Raw cache updated every frame — fast path, no detection
        update_raw_frame(frame)

        frame_count += 1
        # Queue every 8th frame for the detection thread
        if frame_count % 8 == 0:
            try:
                _frame_queue.put_nowait(frame)
            except queue.Full:
                pass  # Detection busy — drop frame, never block camera

        # Heartbeat log every 30 seconds (~450 frames at 15 fps)
        if frame_count % 450 == 0:
            logger.info("CameraReader alive: frame %d, queue size: %d", frame_count, _frame_queue.qsize())

        # Scene/motion/activity throttled to ~2 Hz (every 8 frames at 15 fps)
        scene_tick += 1
        if scene_tick % 8 == 0:
            try:
                motion = detect_motion()
                people = detect_people()
                update_scene(people=people, motion=motion)
                if people != prev_people or motion != prev_motion:
                    ppe_status = get_ppe_status() if ppe_detector.is_ready() else {}
                    ppe_workers = ppe_status.get("workers", [])
                    add_entry(people=people, motion=motion, ppe_workers=ppe_workers)
                    prev_people = people
                    prev_motion = motion
            except Exception as e:
                logger.error("Scene update error: %s", e)

        time.sleep(0.067)  # ~15 fps

    logger.info("Camera reader thread stopped.")


def _detection_loop() -> None:
    from services.ppe_detector import update_cached_frame

    logger.info("Detection thread started.")
    while _running:
        try:
            frame = _frame_queue.get(timeout=1.0)
            update_cached_frame(frame)
            logger.debug("Detection complete — annotated frame updated")
        except queue.Empty:
            continue
        except Exception as e:
            logger.error("Detection error: %s", e)

    logger.info("Detection thread stopped.")


def start_perception_loop() -> None:
    global _thread_cam, _thread_det, _running
    if _running:
        return
    _running = True
    _thread_cam = threading.Thread(target=_camera_reader_loop, daemon=True, name="CameraReader")
    _thread_det = threading.Thread(target=_detection_loop, daemon=True, name="DetectionThread")
    _thread_cam.start()
    _thread_det.start()
    logger.info("Perception loop started (camera + detection threads)")


def stop_perception_loop() -> None:
    global _running
    _running = False
    if _thread_cam:
        _thread_cam.join(timeout=3.0)
    if _thread_det:
        _thread_det.join(timeout=3.0)
