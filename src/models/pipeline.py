"""AlpacaVision AI -- Pipeline completo: detector + clasificadores."""

from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
from PIL import Image

from src.models.detector import AlpacaDetector
from src.models.classifier import AnomalyClassifier

# detector classes feeding each classifier
EYE_DETECTOR_CLASS  = 2  # alpaca_eye
LEG_DETECTOR_CLASSES = (3, 4)  # alpaca_leg_front, alpaca_leg_rear


class AlpacaVisionPipeline:

    def __init__(
        self,
        detector_path: Union[str, Path],
        eye_classifier_path: Optional[Union[str, Path]] = None,
        leg_classifier_path: Optional[Union[str, Path]] = None,
        device: str = "cpu",
        detector_conf: float = 0.5,
        classifier_conf: float = 0.3,
    ):
        self.detector = AlpacaDetector(detector_path, device=device)
        self.eye_clf = (
            AnomalyClassifier(eye_classifier_path, task="eyes", device=device)
            if eye_classifier_path and Path(eye_classifier_path).exists()
            else None
        )
        self.leg_clf = (
            AnomalyClassifier(leg_classifier_path, task="legs", device=device)
            if leg_classifier_path and Path(leg_classifier_path).exists()
            else None
        )
        self.detector_conf = detector_conf
        self.classifier_conf = classifier_conf

    def predict(self, image_input: Union[str, Path, np.ndarray]) -> dict:
        """Process an image and return the full diagnostic report."""
        if isinstance(image_input, (str, Path)):
            img_bgr = cv2.imread(str(image_input))
        else:
            img_bgr = image_input

        h, w = img_bgr.shape[:2]
        detections = self.detector.predict(img_bgr, confidence=self.detector_conf)

        eye_results = []
        leg_results = []

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            crop = img_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            if det["class_id"] == EYE_DETECTOR_CLASS and self.eye_clf:
                clf_result = self.eye_clf.predict(crop)
                eye_results.append({**det, "classification": clf_result})

            elif det["class_id"] in LEG_DETECTOR_CLASSES and self.leg_clf:
                clf_result = self.leg_clf.predict(crop)
                leg_results.append({**det, "classification": clf_result})

            elif det["class_id"] == 0 and self.eye_clf:
                # Stage 1: solo detecta cuerpo completo -- extracción heurística anatómica
                bh, bw = y2 - y1, x2 - x1
                MIN_SIZE = 32
                # Ojo izquierdo: esquina superior izquierda (5-30% altura, 5-35% ancho)
                eye_l = img_bgr[
                    max(0, y1 + int(bh * 0.05)):min(img_bgr.shape[0], y1 + int(bh * 0.30)),
                    max(0, x1 + int(bw * 0.05)):min(img_bgr.shape[1], x1 + int(bw * 0.35)),
                ]
                # Ojo derecho: esquina superior derecha (5-30% altura, 65-95% ancho)
                eye_r = img_bgr[
                    max(0, y1 + int(bh * 0.05)):min(img_bgr.shape[0], y1 + int(bh * 0.30)),
                    max(0, x1 + int(bw * 0.65)):min(img_bgr.shape[1], x2 - int(bw * 0.05)),
                ]
                for i, eye_crop in enumerate([eye_l, eye_r]):
                    if eye_crop.shape[0] >= MIN_SIZE and eye_crop.shape[1] >= MIN_SIZE:
                        clf_result = self.eye_clf.predict(eye_crop)
                        pseudo_det = {
                            **det,
                            "class_id": EYE_DETECTOR_CLASS,
                            "class_name": f"eye_heuristic_{'L' if i == 0 else 'R'}",
                            "source": "heuristic",
                        }
                        eye_results.append({**pseudo_det, "classification": clf_result})

        has_anomaly = any(
            r["classification"].get("is_anomaly", r["classification"]["class_id"] == 0)
            for r in (eye_results + leg_results)
            if "classification" in r
        )

        summary = self._build_summary(eye_results, leg_results)

        return {
            "detections":  detections,
            "eye_results": eye_results,
            "leg_results": leg_results,
            "has_anomaly": has_anomaly,
            "summary":     summary,
        }

    def _build_summary(self, eye_results: list, leg_results: list) -> str:
        lines = []
        if not eye_results and not leg_results:
            return "No se detectaron regiones anatómicas relevantes."

        for i, r in enumerate(eye_results, 1):
            clf = r.get("classification", {})
            lines.append(
                f"Ojo {i}: {clf.get('class_name', 'N/A')} "
                f"(confianza: {clf.get('confidence', 0):.1%})"
            )
        for i, r in enumerate(leg_results, 1):
            clf = r.get("classification", {})
            lines.append(
                f"Extremidad {i} ({r['class_name']}): {clf.get('class_name', 'N/A')} "
                f"(confianza: {clf.get('confidence', 0):.1%})"
            )
        return "\n".join(lines)
