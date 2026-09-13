"""AlpacaVision AI -- Wrapper del detector YOLOv11."""

from pathlib import Path
from typing import Union

import numpy as np


class AlpacaDetector:
    """Wrapper sobre Ultralytics YOLO para detección de regiones anatómicas."""

    CLASS_NAMES = {
        0: "alpaca",
        1: "alpaca_head",
        2: "alpaca_eye",
        3: "alpaca_leg_front",
        4: "alpaca_leg_rear",
        5: "alpaca_body",
    }

    def __init__(self, model_path: Union[str, Path], device: str = "cpu"):
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("Instalar: pip install ultralytics")

        self.model = YOLO(str(model_path))
        self.device = device
        self.model_path = Path(model_path)

    def predict(
        self,
        image: Union[str, Path, np.ndarray],
        confidence: float = 0.5,
        iou: float = 0.45,
    ) -> list[dict]:
        """Detecta regiones anatómicas en una imagen."""
        results = self.model(image, conf=confidence, iou=iou, device=self.device, verbose=False)[0]
        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            detections.append({
                "class_id":   cls_id,
                "class_name": self.CLASS_NAMES.get(cls_id, f"class_{cls_id}"),
                "confidence": float(box.conf[0]),
                "bbox":       [int(v) for v in box.xyxy[0].tolist()],
            })
        return detections

    def predict_batch(self, images: list, **kwargs) -> list[list[dict]]:
        return [self.predict(img, **kwargs) for img in images]
