"""Tests del generador de reportes veterinarios (Groq)."""

import os
import pytest


@pytest.fixture
def sample_pipeline_result():
    return {
        "has_anomaly": True,
        "eye_results": [{
            "class_name": "alpaca_eye",
            "confidence": 0.91,
            "bbox": [120, 80, 200, 140],
            "classification": {
                "class_id": 1,
                "class_name": "eye_cataract",
                "confidence": 0.94,
            },
        }],
        "leg_results": [{
            "class_name": "alpaca_leg_front",
            "confidence": 0.87,
            "bbox": [300, 400, 420, 600],
            "classification": {
                "class_id": 0,
                "class_name": "leg_normal",
                "confidence": 0.89,
            },
        }],
    }


@pytest.fixture
def normal_pipeline_result():
    return {
        "has_anomaly": False,
        "eye_results": [{
            "class_name": "alpaca_eye",
            "confidence": 0.88,
            "bbox": [120, 80, 200, 140],
            "classification": {
                "class_id": 0,
                "class_name": "eye_normal",
                "confidence": 0.96,
            },
        }],
        "leg_results": [],
    }


def test_format_findings_with_anomaly(sample_pipeline_result):
    from src.app.report_generator import _format_findings
    text = _format_findings(sample_pipeline_result)
    assert "catarata" in text.lower()
    assert "94" in text  # confianza


def test_format_findings_normal(normal_pipeline_result):
    from src.app.report_generator import _format_findings
    text = _format_findings(normal_pipeline_result)
    assert "normal" in text.lower()


def test_format_findings_empty():
    from src.app.report_generator import _format_findings
    text = _format_findings({"has_anomaly": False, "eye_results": [], "leg_results": []})
    assert "no se detectaron" in text.lower()


@pytest.mark.skipif(
    not os.environ.get("GROQ_API_KEY"),
    reason="Requiere GROQ_API_KEY en .env"
)
def test_generate_report_live(sample_pipeline_result):
    """Test de integración real con Groq API."""
    from src.app.report_generator import VetReportGenerator
    gen = VetReportGenerator(model="llama-3.1-8b-instant")  # modelo rápido para tests
    report = gen.generate(sample_pipeline_result, animal_id="TEST-001")

    assert report["report_text"]
    assert len(report["report_text"]) > 100
    assert report["has_anomaly"] is True
    assert report["animal_id"] == "TEST-001"
    assert report["tokens_used"] is not None


@pytest.mark.skipif(
    not os.environ.get("GROQ_API_KEY"),
    reason="Requiere GROQ_API_KEY en .env"
)
def test_generate_from_text():
    from src.app.report_generator import VetReportGenerator
    gen = VetReportGenerator(model="llama-3.1-8b-instant")
    result = gen.generate_from_text(
        "Ojo izquierdo con secreción amarillenta, extremidades normales.",
        animal_id="TEST-002"
    )
    assert isinstance(result, str)
    assert len(result) > 50
