"""AlpacaVision AI -- Análisis visual con Groq Vision."""

import base64
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from dotenv import load_dotenv

load_dotenv()

VISION_MODEL  = "meta-llama/llama-4-scout-17b-16e-instruct"
FALLBACK_MODEL = "llama-3.2-11b-vision-preview"

SYSTEM_PROMPT = """Eres un veterinario especializado en camélidos sudamericanos (alpacas, llamas, vicuñas).
Analizas imágenes para detectar anomalías morfológicas visibles.

Responde SIEMPRE en formato JSON válido exactamente así:
{
  "has_anomaly": true/false,
  "severity": "none"|"mild"|"moderate"|"severe",
  "findings": [
    {"region": "nombre_región", "description": "descripción breve", "anomaly_type": "tipo"}
  ],
  "summary": "resumen en una oración"
}

Regiones posibles: piel/lana, ojo, pata_delantera, pata_trasera, hocico, oreja, cuello, cuerpo_general.
Tipos de anomalía: lesion_cutanea, sarna, herida, inflamacion, descarga_ocular, catarata, deformidad, postura_anormal, ninguna.
Si no hay anomalías visibles, has_anomaly=false y findings=[].
"""

USER_PROMPT = "Analiza esta alpaca y detecta cualquier anomalía morfológica o de salud visible en la imagen."


def _img_to_b64(img_bgr: np.ndarray) -> str:
    """Convierte imagen BGR numpy a base64 JPEG."""
    _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode()


def analyze_with_vision(img_bgr: np.ndarray, api_key: Optional[str] = None) -> dict:
    """Analyse an alpaca image with the hosted vision-language model."""
    key = api_key or os.environ.get("GROQ_API_KEY")
    if not key:
        return _empty_result(error="GROQ_API_KEY no configurada")

    try:
        from groq import Groq
    except ImportError:
        return _empty_result(error="groq no instalado")

    b64 = _img_to_b64(img_bgr)
    client = Groq(api_key=key)

    for model in (VISION_MODEL, FALLBACK_MODEL):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                        {"type": "text", "text": USER_PROMPT},
                    ]},
                ],
                max_tokens=400,
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            import json
            data = json.loads(resp.choices[0].message.content)
            data["model"]  = model
            data["error"]  = None
            # Garantizar campos mínimos
            data.setdefault("has_anomaly", False)
            data.setdefault("severity",    "none")
            data.setdefault("findings",    [])
            data.setdefault("summary",     "Análisis completado")
            return data
        except Exception as e:
            last_err = str(e)
            continue

    return _empty_result(error=last_err)


def _empty_result(error: str = "") -> dict:
    return {
        "has_anomaly": False,
        "severity":    "none",
        "findings":    [],
        "summary":     "No se pudo analizar la imagen",
        "model":       None,
        "error":       error,
    }


def vision_to_pipeline_result(vision: dict, detections: list) -> dict:
    """Convert a vision_analyzer result into the AlpacaVisionPipeline format."""
    eye_results = []
    leg_results = []

    for f in vision.get("findings", []):
        region = f.get("region", "")
        atype  = f.get("anomaly_type", "ninguna")
        conf   = 0.85 if vision.get("severity") == "severe" \
                 else 0.70 if vision.get("severity") == "moderate" \
                 else 0.55

        clf = {
            "class_id":    0 if atype == "ninguna" else 1,
            "class_name":  atype,
            "confidence":  conf,
            "description": f.get("description", ""),
        }

        if "ojo" in region or "ocular" in region:
            eye_results.append({"class_id": 2, "class_name": "alpaca_eye",
                                 "confidence": conf, "bbox": [],
                                 "classification": clf})
        elif "pata" in region or "extremidad" in region:
            leg_results.append({"class_id": 3, "class_name": "alpaca_leg_front",
                                 "confidence": conf, "bbox": [],
                                 "classification": clf})

    return {
        "detections":    detections,
        "eye_results":   eye_results,
        "leg_results":   leg_results,
        "has_anomaly":   vision.get("has_anomaly", False),
        "summary":       vision.get("summary", ""),
        "vision_detail": vision.get("findings", []),
        "severity":      vision.get("severity", "none"),
    }
