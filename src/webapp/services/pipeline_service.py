"""Singleton wrapper for AlpacaVisionPipeline -- shared across requests."""
import base64
import logging
import threading
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

ROOT          = Path(__file__).parent.parent.parent.parent  # project root
DETECTOR_PATH = ROOT / "models" / "detector" / "best_clean.pt"          # honest, leakage-free
EYE_CLF_PATH  = ROOT / "models" / "classifier" / "eyes_b2_honest" / "best.pt"

_pipeline = None
_pipeline_error: str | None = None   # stored so the route can return it as JSON
_pipeline_lock  = threading.Lock()   # prevent concurrent init in request threads

CLASS_COLORS = {
    0: (50, 205, 50),
    1: (0, 140, 255),
    2: (255, 60, 60),
    3: (200, 0, 220),
    4: (100, 0, 220),
}
CLASS_LABELS = {0: "alpaca", 1: "head", 2: "eye", 3: "leg front", 4: "leg rear"}


def get_pipeline():
    """Return the singleton pipeline, initialising it on first call."""
    global _pipeline, _pipeline_error
    # Fast path -- already loaded
    if _pipeline is not None:
        return _pipeline

    with _pipeline_lock:
        # Double-checked inside lock
        if _pipeline is not None:
            return _pipeline

        if not DETECTOR_PATH.exists():
            _pipeline_error = f"Detector no encontrado: {DETECTOR_PATH}"
            log.error(_pipeline_error)
            return None

        from src.models.pipeline import AlpacaVisionPipeline

        for device in ("cuda", "cpu"):
            try:
                _pipeline = AlpacaVisionPipeline(
                    detector_path=DETECTOR_PATH,
                    eye_classifier_path=EYE_CLF_PATH if EYE_CLF_PATH.exists() else None,
                    device=device,
                )
                log.info("Pipeline cargado en %s", device)
                _pipeline_error = None
                return _pipeline
            except Exception as exc:
                log.warning("No se pudo cargar pipeline en %s: %s", device, exc)
                _pipeline = None
                _pipeline_error = str(exc)

        log.error("Pipeline no disponible: %s", _pipeline_error)
        return None


def pipeline_ok() -> bool:
    return DETECTOR_PATH.exists()


def draw_boxes(img_bgr: np.ndarray, detections: list) -> np.ndarray:
    out = img_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cls_id = det["class_id"]
        color  = CLASS_COLORS.get(cls_id, (180, 180, 180))
        label  = f"{CLASS_LABELS.get(cls_id, det['class_name'])} {det['confidence']:.0%}"
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def img_to_b64(img_bgr: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return base64.b64encode(buf).decode()
