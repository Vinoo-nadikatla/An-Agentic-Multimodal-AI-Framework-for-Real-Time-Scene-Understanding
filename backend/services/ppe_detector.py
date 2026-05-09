"""
ppe_detector.py
Real-time PPE (Personal Protective Equipment) detection.
Uses a pre-trained YOLOv8 model for safety equipment detection.
Model loading runs in a background thread — never blocks startup.
"""
import base64
import logging
import threading

import cv2
import numpy as np

logger = logging.getLogger(__name__)

HELMET_CLASSES    = {"helmet", "hard hat", "hardhat"}
NO_HELMET_CLASSES = {"no helmet", "no hard hat", "no-helmet", "no hardhat"}
VEST_CLASSES      = {"safety vest", "vest", "hi-vis", "high-vis"}
NO_VEST_CLASSES   = {"no vest", "no safety vest", "no-vest"}
PERSON_CLASSES    = {"person", "worker"}

COLOR_COMPLIANT = (0, 200, 0)    # Green
COLOR_PARTIAL   = (0, 165, 255)  # Orange
COLOR_VIOLATION = (0, 0, 220)    # Red

_lock = threading.Lock()
_latest_result: dict = {
    "workers": [],
    "total_workers": 0,
    "compliant": 0,
    "violations": 0,
    "helmet_compliance": 0.0,
    "vest_compliance": 0.0,
    "overall_compliance": 0.0,
    "annotated_frame": None,
}


class PPEDetector:
    def __init__(self):
        self.model = None
        self.class_names = {}
        self._model_ready = threading.Event()
        # Load model in background so it never blocks server startup
        t = threading.Thread(target=self._load_model, daemon=True, name="PPEModelLoader")
        t.start()

    def _load_model(self):
        try:
            from huggingface_hub import hf_hub_download
            from ultralytics import YOLO
            logger.info("PPE: downloading weights from HuggingFace hub...")
            local_path = hf_hub_download(
                repo_id="keremberke/yolov8n-hard-hat-detection",
                filename="best.pt",
            )
            logger.info("PPE: weights at %s", local_path)
            model = YOLO(local_path)
            model.overrides["conf"] = 0.3
            model.overrides["iou"] = 0.45
            self.model = model
            self.class_names = model.names
            logger.info("PPE model ready. Classes: %s", self.class_names)
        except Exception as e:
            logger.warning("PPE: specialized model failed (%s), trying yolov8n.pt", e)
            try:
                from ultralytics import YOLO
                self.model = YOLO("yolov8n.pt")
                self.class_names = self.model.names
                logger.info("PPE: fallback model ready (COCO classes). Classes: %s", self.class_names)
            except Exception as e2:
                logger.error("PPE: model load failed entirely: %s", e2)
        finally:
            self._model_ready.set()

    def is_ready(self) -> bool:
        return self._model_ready.is_set() and self.model is not None

    def detect(self, frame: np.ndarray) -> dict:
        if not self.is_ready() or frame is None:
            return dict(_latest_result)

        try:
            results = self.model(frame, verbose=False)[0]
            annotated = frame.copy()

            persons, helmets, no_helmets, vests, no_vests = [], [], [], [], []

            for box in results.boxes:
                cls_id    = int(box.cls[0])
                cls_name  = self.class_names.get(cls_id, "").lower()
                conf      = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                det = {"box": (x1, y1, x2, y2), "conf": conf}

                if any(c in cls_name for c in PERSON_CLASSES):
                    persons.append(det)
                elif any(c in cls_name for c in HELMET_CLASSES):
                    helmets.append(det)
                elif any(c in cls_name for c in NO_HELMET_CLASSES):
                    no_helmets.append(det)
                elif any(c in cls_name for c in VEST_CLASSES):
                    vests.append(det)
                elif any(c in cls_name for c in NO_VEST_CLASSES):
                    no_vests.append(det)

            workers = []
            for i, person in enumerate(persons):
                px1, py1, px2, py2 = person["box"]
                pcx = (px1 + px2) // 2

                has_helmet     = any(abs((h["box"][0] + h["box"][2]) // 2 - pcx) < 100 for h in helmets)
                missing_helmet = any(abs((h["box"][0] + h["box"][2]) // 2 - pcx) < 100 for h in no_helmets)
                has_vest       = any(abs((v["box"][0] + v["box"][2]) // 2 - pcx) < 120 for v in vests)
                missing_vest   = any(abs((v["box"][0] + v["box"][2]) // 2 - pcx) < 120 for v in no_vests)

                violations = []
                if missing_helmet: violations.append("No Helmet")
                if missing_vest:   violations.append("No Vest")

                if not violations:       status, color = "compliant", COLOR_COMPLIANT
                elif len(violations) == 1: status, color = "partial",   COLOR_PARTIAL
                else:                    status, color = "violation",  COLOR_VIOLATION

                worker = {
                    "id": i + 1,
                    "box": person["box"],
                    "has_helmet": has_helmet,
                    "has_vest": has_vest,
                    "status": status,
                    "violations": violations,
                    "label": f"Worker {chr(65 + i)}",
                }
                workers.append(worker)

                # Annotate frame
                cv2.rectangle(annotated, (px1, py1), (px2, py2), color, 2)
                status_text = (f"Worker {chr(65 + i)}" +
                               (" - Compliant" if not violations else f" - {', '.join(violations)}"))
                cv2.rectangle(annotated, (px1, py1 - 25), (px2, py1), color, -1)
                cv2.putText(annotated, status_text, (px1 + 4, py1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            total         = len(workers)
            compliant_cnt = sum(1 for w in workers if w["status"] == "compliant")
            helmet_ok     = sum(1 for w in workers if w["has_helmet"])
            vest_ok       = sum(1 for w in workers if w["has_vest"])

            _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            ann_b64 = base64.b64encode(buf).decode()

            result = {
                "workers": workers,
                "total_workers": total,
                "compliant": compliant_cnt,
                "violations": total - compliant_cnt,
                "helmet_compliance":  round(helmet_ok / total * 100) if total > 0 else 100,
                "vest_compliance":    round(vest_ok   / total * 100) if total > 0 else 100,
                "overall_compliance": round(compliant_cnt / total * 100) if total > 0 else 100,
                "annotated_frame": ann_b64,
            }
            with _lock:
                _latest_result.update(result)
            return result

        except Exception as e:
            logger.error("PPE detection error: %s", e)
            return dict(_latest_result)


ppe_detector = PPEDetector()


def get_ppe_status() -> dict:
    with _lock:
        r = dict(_latest_result)
        r.pop("annotated_frame", None)
    r["model_ready"] = ppe_detector.is_ready()
    return r


def get_annotated_frame() -> str | None:
    with _lock:
        return _latest_result.get("annotated_frame")
