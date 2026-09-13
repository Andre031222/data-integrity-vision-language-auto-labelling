from flask import render_template
from . import bp
from src.webapp.services.pipeline_service import pipeline_ok


@bp.route("/")
def landing():
    return render_template("public/landing.html")


@bp.route("/health")
def health():
    return {"status": "ok", "pipeline": pipeline_ok()}
