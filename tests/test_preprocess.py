"""Tests de preprocesamiento."""

import numpy as np
import pytest
from src.data.preprocess import compute_md5, enhance_altitude_image, letterbox


def test_letterbox_square():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    result = letterbox(img, target=640)
    assert result.shape == (640, 640, 3)


def test_letterbox_small():
    img = np.ones((100, 100, 3), dtype=np.uint8) * 128
    result = letterbox(img, target=640)
    assert result.shape == (640, 640, 3)


def test_enhance_altitude_image():
    img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    result = enhance_altitude_image(img)
    assert result.shape == img.shape
    assert result.dtype == np.uint8


def test_compute_md5(tmp_path):
    f = tmp_path / "test.txt"
    f.write_bytes(b"alpacavision")
    h1 = compute_md5(f)
    h2 = compute_md5(f)
    assert h1 == h2
    assert len(h1) == 32
