"""Entry point for the AlpacaVision AI web application."""
import logging
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path so "src.*" imports resolve
ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

log = logging.getLogger(__name__)

# -- Pre-load the ML pipeline BEFORE Flask accepts any requests --------------
def _warmup_pipeline():
    from src.webapp.services.pipeline_service import get_pipeline, pipeline_ok
    if not pipeline_ok():
        log.warning("Detector model not found -- pipeline disabled.")
        return
    log.info("Pre-loading ML pipeline...")
    p = get_pipeline()
    if p:
        import numpy as np
        dummy = np.zeros((320, 320, 3), dtype=np.uint8)
        try:
            p.predict(dummy)
            log.info("Pipeline warmup OK.")
        except Exception as e:
            log.warning("Pipeline warmup failed: %s", e)
    else:
        from src.webapp.services import pipeline_service as ps
        log.warning("Pipeline not loaded: %s", ps._pipeline_error)

_warmup_pipeline()

# -- Create Flask app ---------------------------------------------------------
from src.webapp import create_app

app = create_app(os.environ.get("FLASK_ENV", "development"))

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5050)),
        debug=True,
        use_reloader=False,   # reloader forks the process -> reinitialises PyTorch -> crash
        threaded=True,
    )
