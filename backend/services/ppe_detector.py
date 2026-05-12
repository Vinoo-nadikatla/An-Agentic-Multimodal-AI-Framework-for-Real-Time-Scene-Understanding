"""
ppe_detector.py
Real-time PPE detection — helmet only via YOLOv8.
Model loading runs in a background thread — never blocks startup.
"""
import base64
import logging
import threading
from datetime import datetime

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# keremberke/yolov8n-hard-hat-detection class IDs
HARDHAT_CLASSES    = {0}   # 0 = Hardhat
NO_HARDHAT_CLASSES = {1}   # 1 = NO-Hardhat

COLOR_COMPLIANT = (0, 200, 0)   # Green
COLOR_VIOLATION = (0, 0, 220)   # Red

_lock = threading.Lock()
_latest_result: dict = {
    "workers": [],
    "total_workers": 0,
    "compliant": 0,
    "violations": 0,
    "helmet_compliance": 0,
    "overall_compliance": 0,
    "annotated_frame": None,
}

# Violation duration tracking
_violation_start_times: dict = {}   # worker_label -> datetime when violation started

# TTS announcement throttling
_last_announcement_time: dict = {}  # worker_label -> last announcement datetime
ANNOUNCEMENT_INTERVAL = 30          # seconds between repeat announcements

# Centroid-based worker re-identification
_tracked_workers: dict = {}         # track_id -> {cx, cy, label, last_seen}
_next_worker_id: int = 1
MAX_CENTROID_DISTANCE = 200         # pixels


def _assign_worker_id(cx: int, cy: int, box_area: int) -> tuple[int, str]:
    """Match detection to any tracked worker using Y-weighted centroid distance.
    Active threshold: 200px. Inactive threshold: 300px (1.5×, not 3× — avoids cross-person confusion).
    Y-axis is weighted 1.5× because workers at different depths differ sharply in Y."""
    global _next_worker_id
    now = datetime.now()

    best_id = None
    best_dist = float("inf")

    for track_id, track in _tracked_workers.items():
        dx = cx - track["cx"]
        dy = cy - track["cy"]
        # Y weighted more heavily — people at different depths/distances have very different Y
        dist = (dx ** 2 + (dy * 1.5) ** 2) ** 0.5
        threshold = MAX_CENTROID_DISTANCE if track.get("active", True) else MAX_CENTROID_DISTANCE * 1.5
        if dist < threshold and dist < best_dist:
            best_dist = dist
            best_id = track_id

    if best_id is not None:
        _tracked_workers[best_id]["cx"] = cx
        _tracked_workers[best_id]["cy"] = cy
        _tracked_workers[best_id]["last_seen"] = now
        _tracked_workers[best_id]["active"] = True
        return best_id, _tracked_workers[best_id]["label"]

    # Truly new worker — never seen before
    label = f"Worker {chr(64 + _next_worker_id)}"
    _tracked_workers[_next_worker_id] = {
        "cx": cx, "cy": cy,
        "label": label,
        "last_seen": now,
        "active": True,
    }
    _next_worker_id += 1
    return _next_worker_id - 1, label


def _cleanup_lost_workers() -> None:
    """Mark workers inactive if not seen recently. Labels are NEVER deleted — they persist for the session."""
    now = datetime.now()
    for track in _tracked_workers.values():
        track["active"] = (now - track["last_seen"]).total_seconds() <= 5


class PPEDetector:
    def __init__(self):
        self.model = None
        self.class_names = {}
        self._ready = False
        t = threading.Thread(target=self._load, daemon=True, name="PPEModelLoader")
        t.start()

    def _load(self):
        try:
            import warnings
            warnings.filterwarnings("ignore")
            from ultralytics import YOLO

            ppe_model_path = (
                r"C:\Users\vinoo\.cache\huggingface\hub"
                r"\models--keremberke--yolov8n-hard-hat-detection"
                r"\snapshots\287bafa2feb311ee45d21f9e9b33315ff6ff955d\best.pt"
            )

            import os
            if os.path.exists(ppe_model_path):
                self.model = YOLO(ppe_model_path)
                logger.info("PPE helmet model loaded: %s", self.model.names)
            else:
                self.model = YOLO("yolov8n.pt")
                logger.info("PPE: using YOLOv8n fallback")

            self.class_names = self.model.names
            self._ready = True
        except Exception as e:
            logger.error("PPE model load failed: %s", e)

    def is_ready(self) -> bool:
        return self._ready and self.model is not None

    def detect(self, frame: np.ndarray) -> dict:
        if not self.is_ready() or frame is None:
            return dict(_latest_result)

        try:
            _cleanup_lost_workers()
            results   = self.model(frame, verbose=False)[0]
            annotated = frame.copy()

            workers = []
            for box in results.boxes:
                conf = float(box.conf[0])
                if conf < 0.40:
                    continue

                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                box_area = (x2 - x1) * (y2 - y1)
                frame_area = frame.shape[0] * frame.shape[1]
                if box_area / frame_area < 0.02:
                    continue

                track_id, worker_label = _assign_worker_id(cx, cy, box_area)

                has_helmet = cls_id in HARDHAT_CLASSES
                status     = "compliant" if has_helmet else "violation"
                color      = COLOR_COMPLIANT if has_helmet else COLOR_VIOLATION
                violations = [] if has_helmet else ["No Helmet"]

                # Violation duration tracking
                if status == "violation":
                    if worker_label not in _violation_start_times:
                        _violation_start_times[worker_label] = datetime.now()
                    elapsed = int((datetime.now() - _violation_start_times[worker_label]).total_seconds())
                    duration_str = f"{elapsed // 3600:02d}:{(elapsed % 3600) // 60:02d}:{elapsed % 60:02d}"
                else:
                    _violation_start_times.pop(worker_label, None)
                    duration_str = None

                worker = {
                    "id": track_id,
                    "box": (x1, y1, x2, y2),
                    "has_helmet": has_helmet,
                    "status": status,
                    "violations": violations,
                    "label": worker_label,
                    "violation_duration": duration_str,
                    "active": True,
                }
                workers.append(worker)

                label_text = worker_label + (" - OK" if has_helmet else " - No Helmet")
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.rectangle(annotated, (x1, y1 - 22), (x2, y1), color, -1)
                cv2.putText(annotated, label_text, (x1 + 3, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            # Only count workers visible in this frame (active=True was set in _assign_worker_id)
            workers = [w for w in workers if w.get("active", True)]
            total         = len(workers)
            compliant_cnt = sum(w["status"] == "compliant" for w in workers)
            helmet_ok     = sum(w["has_helmet"] for w in workers)

            annotated_small = cv2.resize(annotated, (480, 270), interpolation=cv2.INTER_LINEAR)
            _, buf = cv2.imencode(".jpg", annotated_small, [cv2.IMWRITE_JPEG_QUALITY, 55])
            ann_b64 = base64.b64encode(buf).decode()

            result = {
                "workers": workers,
                "total_workers": total,
                "compliant": compliant_cnt,
                "violations": total - compliant_cnt,
                "helmet_compliance":  round(helmet_ok / total * 100) if total > 0 else 0,
                "overall_compliance": round(compliant_cnt / total * 100) if total > 0 else 0,
                "annotated_frame": ann_b64,
            }
            with _lock:
                _latest_result.update(result)
            return result

        except Exception as e:
            logger.error("PPE detection error: %s", e)
            return dict(_latest_result)


ppe_detector = PPEDetector()

# Two-tier frame cache — raw updates every frame, annotated only every 8th (YOLO is slow)
_cached_raw_frame: str | None = None
_cached_annotated_frame: str | None = None
_cached_annotated_bytes: bytes | None = None   # raw JPEG bytes for MJPEG streaming
_cache_lock = threading.Lock()


def update_raw_frame(frame: np.ndarray) -> None:
    """Encode and cache the raw camera frame — no detection, runs every loop tick."""
    global _cached_raw_frame, _cached_annotated_bytes
    small = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)
    _, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])
    frame_bytes = buf.tobytes()
    with _cache_lock:
        _cached_raw_frame = base64.b64encode(frame_bytes).decode()
        if _cached_annotated_bytes is None:
            _cached_annotated_bytes = frame_bytes


def update_cached_frame(frame: np.ndarray) -> dict:
    """Run YOLO detection, cache annotated frame (b64 + raw bytes for MJPEG)."""
    global _cached_annotated_frame, _cached_annotated_bytes
    result = ppe_detector.detect(frame)
    ann_b64 = result.get("annotated_frame")
    if ann_b64:
        frame_bytes = base64.b64decode(ann_b64)
        with _cache_lock:
            _cached_annotated_frame = ann_b64
            _cached_annotated_bytes = frame_bytes
    return result


def get_cached_frame() -> str | None:
    """Return annotated frame if available, fall back to raw frame."""
    with _cache_lock:
        return _cached_annotated_frame or _cached_raw_frame


def get_cached_annotated_bytes() -> bytes | None:
    """Return raw JPEG bytes of latest frame — used by MJPEG streaming endpoint."""
    with _cache_lock:
        return _cached_annotated_bytes


def get_ppe_status() -> dict:
    with _lock:
        r = dict(_latest_result)
        r.pop("annotated_frame", None)
    r["model_ready"] = ppe_detector.is_ready()
    return r


def get_annotated_frame() -> str | None:
    with _lock:
        return _latest_result.get("annotated_frame")


def get_pending_announcements() -> list[str]:
    """Returns TTS announcement strings for workers missing a helmet for more than 30 seconds."""
    announcements = []
    now = datetime.now()

    with _lock:
        workers = list(_latest_result.get("workers", []))

    for worker in workers:
        label = worker["label"]
        if worker["status"] != "violation":
            _last_announcement_time.pop(label, None)
            continue

        start = _violation_start_times.get(label)
        if not start or (now - start).total_seconds() < 30:
            continue

        last = _last_announcement_time.get(label)
        if last and (now - last).total_seconds() < ANNOUNCEMENT_INTERVAL:
            continue

        announcements.append(
            f"Warning. {label} is not wearing a helmet. Please wear PPE immediately."
        )
        _last_announcement_time[label] = now

    return announcements
