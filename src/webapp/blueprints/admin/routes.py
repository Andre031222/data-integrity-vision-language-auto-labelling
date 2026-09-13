"""Admin blueprint - super-admin only panel + config API."""
import os
import uuid
from functools import wraps
from pathlib import Path

from flask import (abort, current_app, jsonify, redirect,
                   render_template, request, url_for)
from flask_login import current_user, login_required

from src.webapp.extensions import db
from src.webapp.models.system_config import (DEFAULT_CONFIG, SystemConfig,
                                              get_site_config)
from . import bp

ALLOWED_IMAGES = {"image/jpeg", "image/png", "image/gif", "image/webp", "image/svg+xml"}


# -- decorator -----------------------------------------------------------------
def super_admin_required(f):
    @wraps(f)
    @login_required
    def decorated(*args, **kwargs):
        if current_user.role != "super_admin":
            abort(403)
        return f(*args, **kwargs)
    return decorated


# -- HTML panel ----------------------------------------------------------------
@bp.route("/")
@super_admin_required
def panel():
    config = get_site_config()
    return render_template("admin/panel.html", config=config)


# -- API: get all config (for panel JS) ----------------------------------------
@bp.route("/api/config/all")
@super_admin_required
def api_config_all():
    config = get_site_config()
    return jsonify(config)


# -- API: save single key -------------------------------------------------------
@bp.route("/api/config", methods=["POST"])
@super_admin_required
def api_config_save():
    data = request.get_json(silent=True) or {}
    key   = data.get("key", "").strip()
    value = data.get("value", "")
    ctype = data.get("type", "text")
    if not key:
        return jsonify({"error": "key requerido"}), 400
    SystemConfig.set(key, str(value), ctype)
    return jsonify({"ok": True, "key": key})


# -- API: bulk save -------------------------------------------------------------
@bp.route("/api/config/bulk", methods=["POST"])
@super_admin_required
def api_config_bulk():
    data = request.get_json(silent=True) or {}
    for key, value in data.items():
        if key in DEFAULT_CONFIG or True:   # allow any key
            SystemConfig.set(str(key), str(value) if value is not None else "")
    return jsonify({"ok": True, "saved": len(data)})


# -- API: image upload ---------------------------------------------------------
@bp.route("/api/config/upload/<config_key>", methods=["POST"])
@super_admin_required
def api_config_upload(config_key: str):
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400
    f = request.files["file"]
    if f.content_type not in ALLOWED_IMAGES:
        return jsonify({"error": "Tipo no permitido"}), 400

    ext        = Path(f.filename).suffix.lower() or ".png"
    fname      = f"{config_key}_{uuid.uuid4().hex[:8]}{ext}"
    upload_dir = Path(current_app.root_path) / "static" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    fpath = upload_dir / fname
    f.save(str(fpath))

    url = f"/static/uploads/{fname}"
    SystemConfig.set(config_key, url, "image")
    return jsonify({"ok": True, "url": url, "key": config_key})


# -- API: reset to defaults -----------------------------------------------------
@bp.route("/api/config/reset", methods=["POST"])
@super_admin_required
def api_config_reset():
    SystemConfig.query.delete()
    db.session.commit()
    return jsonify({"ok": True, "message": "Configuración restablecida a valores por defecto"})
