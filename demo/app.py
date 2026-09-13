"""paqocha -- AlpacaVision public demo."""
import base64
import io
import logging
import os
import threading
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT))

try:                                    # load GROQ_API_KEY from a local .env if present
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except Exception:                       # noqa: BLE001
    pass

DETECTOR_PATH = ROOT / "models" / "detector" / "best_v2_n.pt"  # scene-level protocol, seed 0

# Local vision model via Ollama -- ONLY a plain-language description, never a diagnosis.
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_VISION_MODEL = os.environ.get("OLLAMA_VISION_MODEL", "moondream")
# Short prompt: small vision models (moondream) describe faithfully and do not diagnose on
DESCRIBE_PROMPT = "Describe this photo of an alpaca."


def _two_sentences(text, limit=320):
    """Trim a description to at most two sentences / `limit` chars for the result card."""
    import re
    parts = re.split(r"(?<=[.!?])\s+", " ".join(text.split()))
    out = " ".join(parts[:2]).strip()
    return (out[:limit].rsplit(" ", 1)[0] + "...") if len(out) > limit else out


def ai_describe(img_bgr):
    """Short experimental description via a local Ollama vision model, or None. Never raises."""
    import json as _json, urllib.request
    try:
        # downscale for the vision model -- big images make CPU inference very slow
        h, w = img_bgr.shape[:2]
        scale = 512.0 / max(h, w)
        small = cv2.resize(img_bgr, (int(w * scale), int(h * scale))) if scale < 1 else img_bgr
        payload = _json.dumps({
            "model": OLLAMA_VISION_MODEL,
            "prompt": DESCRIBE_PROMPT,
            "images": [img_to_b64(small)],
            "stream": False,
            "options": {"temperature": 0.2, "num_predict": 90},
        }).encode()
        req = urllib.request.Request(
            OLLAMA_URL.rstrip("/") + "/api/generate",
            data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=90) as r:
            data = _json.loads(r.read().decode())
        txt = (data.get("response") or "").strip()
        return _two_sentences(txt) if txt else None
    except Exception:  # noqa: BLE001  (Ollama down, model missing, timeout -- all non-fatal)
        return None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("paqocha")

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB

# -- detector singleton (thread-safe, CUDA->CPU fallback) ---------------------
_pipeline = None
_pipeline_err = None
_lock = threading.Lock()


def get_pipeline():
    global _pipeline, _pipeline_err
    if _pipeline is not None:
        return _pipeline
    with _lock:
        if _pipeline is not None:
            return _pipeline
        if not DETECTOR_PATH.exists():
            _pipeline_err = f"Detector weights not found: {DETECTOR_PATH}"
            log.error(_pipeline_err)
            return None
        from src.models.pipeline import AlpacaVisionPipeline
        import torch
        devices = ("cuda", "cpu") if torch.cuda.is_available() else ("cpu",)
        for device in devices:
            try:
                _pipeline = AlpacaVisionPipeline(
                    detector_path=DETECTOR_PATH,
                    eye_classifier_path=None,   # detector-only demo -- classifier is not diagnostic
                    leg_classifier_path=None,
                    device=device,
                )
                log.info("Detector loaded on %s", device)
                _pipeline_err = None
                return _pipeline
            except Exception as exc:  # noqa: BLE001
                log.warning("Could not load on %s: %s", device, exc)
                _pipeline, _pipeline_err = None, str(exc)
        return None


CLASS_COLORS = {0: (50, 205, 50), 1: (0, 140, 255), 2: (255, 60, 60),
                3: (200, 0, 220), 4: (100, 0, 220)}


def draw_boxes(img_bgr, detections):
    out = img_bgr.copy()
    h, w = out.shape[:2]
    # scale line thickness and font to the image size so boxes stay visible
    thick = max(2, round(min(h, w) / 220))
    fscale = max(0.5, min(h, w) / 900)
    ftk = max(1, round(thick * 0.7))
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        color = CLASS_COLORS.get(det.get("class_id", 0), (46, 204, 113))
        label = f"{det.get('class_name', 'alpaca')} {det['confidence']:.0%}"
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fscale, ftk)
        ty = max(y1, th + 12)
        cv2.rectangle(out, (x1, ty - th - 12), (x1 + tw + 12, ty), color, -1)
        cv2.putText(out, label, (x1 + 6, ty - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, fscale, (255, 255, 255), ftk, cv2.LINE_AA)
    return out


def img_to_b64(img_bgr):
    _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf).decode()


# -- routes ------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "detector": DETECTOR_PATH.exists()})


@app.route("/api/predict", methods=["POST"])
def predict():
    if "image" not in request.files or not request.files["image"].filename:
        return jsonify({"error": "No image received"}), 400
    try:
        conf = float(request.form.get("confidence", 0.4))
    except ValueError:
        conf = 0.4
    conf = min(max(conf, 0.05), 0.95)

    try:
        contents = request.files["image"].read()
        img_pil = Image.open(io.BytesIO(contents)).convert("RGB")
        img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Could not read the image: {exc}"}), 400

    pipeline = get_pipeline()
    if pipeline is None:
        return jsonify({"error": f"Detector unavailable: {_pipeline_err}"}), 503

    pipeline.detector_conf = conf
    try:
        result = pipeline.predict(img_bgr)
    except Exception as exc:  # noqa: BLE001
        log.exception("inference failed")
        return jsonify({"error": f"Inference error: {exc}"}), 500

    detections = [d for d in result["detections"] if d.get("class_id", 0) == 0] or result["detections"]
    annotated = draw_boxes(img_bgr, detections)
    confs = [d["confidence"] for d in detections]
    return jsonify({
        "image_b64": img_to_b64(annotated),
        "n_detections": len(detections),
        "confidence_used": conf,
        "max_confidence": max(confs) if confs else 0.0,
        "detections": [
            {"bbox": d["bbox"], "confidence": d["confidence"],
             "class_name": d.get("class_name", "alpaca")} for d in detections
        ],
    })


def _read_image(file):
    contents = file.read()
    img_pil = Image.open(io.BytesIO(contents)).convert("RGB")
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


@app.route("/api/describe", methods=["POST"])
def describe():
    """Optional, slow-friendly: a local, private, experimental image description."""
    if "image" not in request.files or not request.files["image"].filename:
        return jsonify({"description": None})
    try:
        img_bgr = _read_image(request.files["image"])
    except Exception:  # noqa: BLE001
        return jsonify({"description": None})
    return jsonify({"description": ai_describe(img_bgr)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5070, debug=False)
