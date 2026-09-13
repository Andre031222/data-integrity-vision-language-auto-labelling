"""Generate field veterinary reports; the report text itself is in Spanish by design."""

import os
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

ATYPE_ES = {
    "lesion_cutanea":   "Lesión cutánea",
    "sarna":            "Sarna",
    "herida":           "Herida",
    "inflamacion":      "Inflamación",
    "descarga_ocular":  "Descarga ocular",
    "catarata":         "Catarata",
    "deformidad":       "Deformidad",
    "postura_anormal":  "Postura anormal",
    "ninguna":          "Sin anomalía",
}

SEV_ES = {
    "mild":     "leve",
    "moderate": "moderada",
    "severe":   "grave",
    "none":     "sin severidad registrada",
}

SYSTEM_PROMPT = """Eres un médico veterinario especializado en camélidos sudamericanos (alpacas, llamas, vicuñas) del altiplano peruano.

Recibes hallazgos de un sistema de visión computacional (AlpacaVision AI: YOLO + Groq Vision) y redactas un reporte veterinario de campo en español.

REGLAS ESTRICTAS:
1. Empieza DIRECTAMENTE con el problema más importante detectado -- sin introducción al sistema, sin fecha, sin metadata.
2. Si hay anomalía: describe qué es, qué región afecta, qué significa clínicamente y qué acción tomar HOY.
3. Si no hay anomalía: confirma que el animal luce saludable y da una recomendación de seguimiento breve.
4. Secciones: **DIAGNÓSTICO PRELIMINAR**, **HALLAZGOS**, **ACCIÓN RECOMENDADA**.
5. Máximo 250 palabras. Español claro para productores y veterinarios de campo.
6. Al final, una línea pequeña: "[!] Reporte de apoyo diagnóstico -- requiere evaluación presencial."
7. No inventes hallazgos no presentes en los datos."""


def _format_findings(pipeline_result: dict) -> str:
    """Build the findings text, giving priority to the vision-language output."""
    lines = []

    # 1. Hallazgos de Groq Vision (siempre disponibles cuando hay GROQ_API_KEY)
    vision_detail = pipeline_result.get("vision_detail", [])
    severity      = pipeline_result.get("severity", "none")
    has_anomaly   = pipeline_result.get("has_anomaly", False)

    if vision_detail:
        sev_text = SEV_ES.get(severity, severity)
        lines.append(f"ANÁLISIS GROQ VISION -- Severidad: {sev_text}")
        for f in vision_detail:
            atype = ATYPE_ES.get(f.get("anomaly_type", ""), f.get("anomaly_type", ""))
            region = f.get("region", "región no especificada")
            desc   = f.get("description", "")
            lines.append(f"  * {atype} en {region}: {desc}")
    elif not has_anomaly:
        lines.append("Groq Vision: sin anomalías morfológicas visibles.")

    # 2. Detecciones YOLO
    detections = pipeline_result.get("detections", [])
    yolo_labels = {0:"alpaca",1:"cabeza",2:"ojo",3:"pata_delantera",4:"pata_trasera"}
    if detections:
        det_str = ", ".join(
            f"{yolo_labels.get(d['class_id'], d['class_name'])} ({d['confidence']:.0%})"
            for d in detections
        )
        lines.append(f"YOLO detectó: {det_str}")
    else:
        lines.append("YOLO: ningún objeto detectado (imagen puede ser de baja calidad o ángulo incorrecto).")

    # 3. Clasificadores EfficientNet (si estuvieran disponibles)
    eye_results = pipeline_result.get("eye_results", [])
    leg_results = pipeline_result.get("leg_results", [])
    for r in eye_results:
        clf = r.get("classification", {})
        lines.append(f"  Ojo -- {clf.get('class_name','?')} ({clf.get('confidence',0):.1%})")
    for r in leg_results:
        clf = r.get("classification", {})
        lines.append(f"  Extremidad -- {clf.get('class_name','?')} ({clf.get('confidence',0):.1%})")

    return "\n".join(lines) if lines else "Sin datos suficientes para análisis."


class VetReportGenerator:
    """Generate veterinary reports with a hosted large language model."""

    def __init__(self, model: str = "llama-3.3-70b-versatile", api_key: Optional[str] = None):
        try:
            from groq import Groq
        except ImportError:
            raise ImportError("Instalar: pip install groq")
        self.model  = model
        self.client = Groq(api_key=api_key or os.environ.get("GROQ_API_KEY"))

    def generate(self, pipeline_result: dict, animal_id: str = "SIN-ID",
                 additional_context: str = "") -> dict:
        findings_text = _format_findings(pipeline_result)
        has_anomaly   = pipeline_result.get("has_anomaly", False)
        timestamp     = datetime.now().strftime("%d/%m/%Y %H:%M")

        user_prompt = (
            f"Animal: {animal_id}  |  {timestamp}\n\n"
            f"HALLAZGOS:\n{findings_text}\n"
            + (f"\nOBSERVACIONES DEL OPERADOR: {additional_context}\n" if additional_context else "")
            + "\nRedacta el reporte veterinario."
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=500,
            temperature=0.25,
        )

        return {
            "report_text":  response.choices[0].message.content,
            "has_anomaly":  has_anomaly,
            "animal_id":    animal_id,
            "timestamp":    timestamp,
            "model":        self.model,
            "findings_raw": findings_text,
            "tokens_used":  response.usage.total_tokens if response.usage else None,
        }
