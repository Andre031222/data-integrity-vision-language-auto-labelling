import io
import logging
import os
import uuid
from pathlib import Path

import cv2
import numpy as np
from flask import jsonify, request, current_app
from flask_login import login_required, current_user
from PIL import Image

from src.webapp.extensions import db
from src.webapp.models.analysis import Analysis
from src.webapp.models.animal import Animal
import src.webapp.services.pipeline_service as _svc
from src.webapp.services.pipeline_service import get_pipeline, draw_boxes, img_to_b64
from . import bp

log = logging.getLogger(__name__)
ROOT = Path(__file__).parent.parent.parent.parent.parent


# -- helpers ----------------------------------------------------------------

def _normalise_findings(findings: list) -> list:
    """Ensure every finding has the keys the template expects: type, severity, description."""
    out = []
    for f in findings:
        out.append({
            "type":        f.get("type") or f.get("anomaly_type") or f.get("condition") or "Hallazgo",
            "severity":    f.get("severity", ""),
            "description": f.get("description") or f.get("region") or "",
            "region":      f.get("region", ""),
        })
    return out


# -- routes -----------------------------------------------------------------

@bp.route("/predict", methods=["POST"])
@login_required
def predict():
    try:
        return _do_predict()
    except Exception as exc:
        log.exception("Unhandled error in /predict")
        return jsonify({"error": f"Error interno del servidor: {exc}"}), 500


def _do_predict():
    if "image" not in request.files:
        return jsonify({"error": "No se recibio imagen"}), 400

    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "Archivo vacio"}), 400

    conf       = float(request.form.get("confidence", 0.4))
    animal_tag = request.form.get("animal_id", "").strip()
    gen_report = request.form.get("gen_report") == "true"

    # Resolve animal FK (optional)
    animal_id_fk = None
    if animal_tag:
        a = Animal.query.filter_by(arete=animal_tag, owner_id=current_user.id).first()
        if a:
            animal_id_fk = a.id

    # Decode image
    try:
        contents = file.read()
        img_pil  = Image.open(io.BytesIO(contents)).convert("RGB")
        img_bgr  = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except Exception as e:
        return jsonify({"error": f"No se pudo leer la imagen: {e}"}), 400

    # Pipeline
    pipeline = get_pipeline()
    if pipeline is None:
        err_msg = _svc._pipeline_error or "Modelo no disponible"
        return jsonify({"error": f"Pipeline no disponible: {err_msg}"}), 503

    pipeline.detector_conf = conf
    try:
        result = pipeline.predict(img_bgr)
    except Exception as e:
        log.exception("pipeline.predict failed")
        return jsonify({"error": f"Error durante la inferencia: {e}"}), 500

    # Groq Vision (optional, never crashes the request)
    vision_detail = []
    vision_model  = None
    severity      = "none"

    if os.environ.get("GROQ_API_KEY"):
        try:
            from src.app.vision_analyzer import analyze_with_vision
            vision = analyze_with_vision(img_bgr)
            if not vision.get("error"):
                raw_findings          = vision.get("findings", [])
                vision_detail         = _normalise_findings(raw_findings)
                severity              = vision.get("severity", "none")
                vision_model          = vision.get("model")
                result["has_anomaly"] = vision.get("has_anomaly", result["has_anomaly"])
                result["summary"]     = vision.get("summary", result.get("summary", ""))
        except Exception:
            log.exception("Groq Vision failed (non-fatal)")

    # Annotate image
    annotated = draw_boxes(img_bgr, result["detections"])
    img_b64   = img_to_b64(annotated)

    # Save annotated image to disk
    upload_dir = Path(current_app.config["UPLOAD_FOLDER"])
    upload_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{uuid.uuid4().hex}.jpg"
    fpath = upload_dir / fname
    try:
        cv2.imwrite(str(fpath), annotated, [cv2.IMWRITE_JPEG_QUALITY, 88])
        rel_path = str(fpath.relative_to(ROOT))
    except Exception:
        rel_path = str(fpath)

    # Vet report (optional, never crashes)
    vet_report = None
    if gen_report and os.environ.get("GROQ_API_KEY"):
        try:
            from src.app.report_generator import VetReportGenerator
            payload = {**result, "vision_detail": vision_detail, "severity": severity}
            rep        = VetReportGenerator().generate(payload, animal_id=animal_tag or "SIN-ID")
            vet_report = rep["report_text"]
        except Exception as e:
            log.exception("VetReportGenerator failed (non-fatal)")
            vet_report = f"Error al generar reporte: {e}"

    # Persist to DB
    try:
        analysis = Analysis(
            user_id         = current_user.id,
            animal_id       = animal_id_fk,
            annotated_path  = rel_path,
            n_detections    = len(result["detections"]),
            detections_json = result["detections"],
            eye_results     = result.get("eye_results"),
            leg_results     = result.get("leg_results"),
            has_anomaly     = result["has_anomaly"],
            severity        = severity,
            findings_json   = vision_detail,
            summary         = result.get("summary", ""),
            vet_report      = vet_report,
            conf_threshold  = conf,
        )
        db.session.add(analysis)
        db.session.commit()
        analysis_id = analysis.id
    except Exception:
        log.exception("DB persist failed (non-fatal)")
        db.session.rollback()
        analysis_id = None

    return jsonify({
        "image_b64":    img_b64,
        "has_anomaly":  result["has_anomaly"],
        "severity":     severity,
        "detections":   result["detections"],
        "eye_results":  result.get("eye_results", []),
        "leg_results":  result.get("leg_results", []),
        "summary":      result.get("summary", ""),
        "vision_detail": vision_detail,
        "vision_model": vision_model,
        "vet_report":   vet_report,
        "animal_id":    animal_tag,
        "n_detections": len(result["detections"]),
        "analysis_id":  analysis_id,
    })


@bp.route("/animals", methods=["GET"])
@login_required
def list_animals():
    animals = Animal.query.filter_by(owner_id=current_user.id).all()
    return jsonify([
        {"id": a.id, "arete": a.arete, "name": a.name or a.arete}
        for a in animals
    ])


@bp.route("/public/config", methods=["GET"])
def public_config():
    """Public endpoint - returns only whitelisted config keys."""
    from src.webapp.models.system_config import get_site_config, PUBLIC_KEYS
    full = get_site_config()
    return jsonify({k: v for k, v in full.items() if k in PUBLIC_KEYS})
