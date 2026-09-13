"""Tests de los modelos (sin GPU, con mocks)."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch


def test_pipeline_import():
    """Verifica que el pipeline se puede importar sin errores."""
    from src.models.pipeline import AlpacaVisionPipeline
    from src.models.detector import AlpacaDetector
    from src.models.classifier import AnomalyClassifier


def test_draw_detections():
    from src.app.inference import draw_detections
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = [
        {"class_id": 0, "class_name": "alpaca", "confidence": 0.95, "bbox": [10, 10, 200, 300]},
        {"class_id": 2, "class_name": "alpaca_eye", "confidence": 0.80, "bbox": [50, 50, 100, 100]},
    ]
    result = draw_detections(img, detections)
    assert result.shape == img.shape


def test_pil_bgr_conversion():
    from PIL import Image
    from src.app.inference import pil_to_bgr, bgr_to_pil
    pil = Image.new("RGB", (100, 100), color=(255, 0, 0))
    bgr = pil_to_bgr(pil)
    assert bgr.shape == (100, 100, 3)
    restored = bgr_to_pil(bgr)
    assert restored.size == (100, 100)
