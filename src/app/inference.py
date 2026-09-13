"""AlpacaVision AI -- Utilidades de inferencia reutilizables."""

from pathlib import Path
from typing import Union

import cv2
import numpy as np
from PIL import Image

# Paleta de colores por clase detector (BGR)
CLASS_COLORS = {
    0: (0, 200, 0),    # alpaca -- verde
    1: (200, 100, 0),  # alpaca_head -- naranja
    2: (0, 100, 255),  # alpaca_eye -- azul
    3: (255, 0, 150),  # alpaca_leg_front -- violeta
    4: (255, 0, 80),   # alpaca_leg_rear -- rosa
    5: (0, 200, 200),  # alpaca_body -- cian
}


def draw_detections(image: np.ndarray, detections: list) -> np.ndarray:
    """Dibuja bounding boxes y etiquetas sobre la imagen."""
    out = image.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cls_id = det["class_id"]
        color = CLASS_COLORS.get(cls_id, (128, 128, 128))
        label = f"{det['class_name']} {det['confidence']:.2f}"

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(out, label, (x1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return out


def load_image(source: Union[str, Path, np.ndarray, Image.Image]) -> np.ndarray:
    """Carga una imagen a formato BGR numpy."""
    if isinstance(source, (str, Path)):
        img = cv2.imread(str(source))
        if img is None:
            raise FileNotFoundError(f"No se pudo cargar: {source}")
        return img
    if isinstance(source, Image.Image):
        return cv2.cvtColor(np.array(source.convert("RGB")), cv2.COLOR_RGB2BGR)
    return source  # ya es numpy BGR


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)


def bgr_to_pil(image: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
