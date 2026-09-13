"""Tests de métricas."""

import numpy as np
import pytest


def test_evaluate_classifier_binary():
    from src.evaluation.metrics import evaluate_classifier

    y_true = [0, 0, 1, 1, 0, 1]
    y_pred = [0, 0, 1, 0, 0, 1]
    y_proba = [[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.6, 0.4], [0.7, 0.3], [0.1, 0.9]]
    classes = ["normal", "anomaly"]

    metrics = evaluate_classifier(y_true, y_pred, y_proba, classes)
    assert "accuracy" in metrics
    assert "f1_macro" in metrics
    assert metrics["accuracy"] == pytest.approx(5 / 6, rel=1e-3)


def test_evaluate_classifier_multiclass():
    from src.evaluation.metrics import evaluate_classifier

    y_true  = [0, 1, 2, 0, 1, 2]
    y_pred  = [0, 1, 2, 0, 2, 1]
    y_proba = [[0.8, 0.1, 0.1]] * 6
    classes = ["a", "b", "c"]

    metrics = evaluate_classifier(y_true, y_pred, y_proba, classes)
    assert metrics["accuracy"] == pytest.approx(4 / 6, rel=1e-3)
