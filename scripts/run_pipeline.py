"""AlpacaVision AI -- Pipeline completo Stage 2 + retomar Stage 1."""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Forzar UTF-8 en stdout para evitar UnicodeEncodeError en Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR    = Path(__file__).parent.parent
PYTHON      = BASE_DIR / "venv" / "Scripts" / "python.exe"
YOLO_CLI    = BASE_DIR / "venv" / "Scripts" / "yolo.exe"
LAST_PT     = BASE_DIR / "runs" / "detect" / "outputs" / "training_runs" / \
              "stage1_alpaca_detector" / "weights" / "last.pt"
LOG_DIR     = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def run_step(label: str, cmd: list[str], log_file: Path) -> bool:
    log(f"=== INICIO: {label} ===")
    t0 = time.time()

    with open(log_file, "w", buffering=1, encoding="utf-8", errors="replace") as f:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        proc = subprocess.Popen(
            cmd,
            cwd=str(BASE_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        for line in proc.stdout:
            try:
                sys.stdout.write(line)
                sys.stdout.flush()
            except (UnicodeEncodeError, UnicodeDecodeError):
                sys.stdout.write(line.encode("ascii", errors="replace").decode("ascii") + "\n")
                sys.stdout.flush()
            f.write(line)
        proc.wait()

    elapsed = time.time() - t0
    mins, secs = divmod(int(elapsed), 60)
    if proc.returncode == 0:
        log(f"=== OK: {label} ({mins}m{secs:02d}s) -- log: {log_file.name} ===")
        return True
    else:
        log(f"=== FALLO: {label} (código {proc.returncode}) -- ver {log_file} ===")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-autolabel",  action="store_true")
    parser.add_argument("--skip-upload",     action="store_true")
    parser.add_argument("--skip-training",   action="store_true")
    args = parser.parse_args()

    log("AlpacaVision AI -- Pipeline completo")
    log(f"Base: {BASE_DIR}")
    log(f"Python: {PYTHON}")

    # -- 1. Auto-labeling --------------------------------------------------------
    if not args.skip_autolabel:
        ok = run_step(
            "Auto-labeling Stage 2 (Grounding DINO)",
            [str(PYTHON), "scripts/auto_label.py"],
            LOG_DIR / "autolabel.log",
        )
        if not ok:
            log("Pipeline abortado en auto-labeling.")
            sys.exit(1)
    else:
        log("SKIP: auto-labeling")

    # -- 2. Upload a Roboflow -----------------------------------------------------
    if not args.skip_upload:
        ok = run_step(
            "Upload pseudo-labels -> Roboflow Stage 2",
            [str(PYTHON), "scripts/upload_labels_roboflow.py"],
            LOG_DIR / "upload.log",
        )
        if not ok:
            log("Upload falló -- continuando con entrenamiento de todas formas.")
    else:
        log("SKIP: upload Roboflow")

    # -- 3. Resume training Stage 1 -----------------------------------------------
    if not args.skip_training:
        if not LAST_PT.exists():
            log(f"ERROR: no existe {LAST_PT}")
            sys.exit(1)

        ok = run_step(
            "Resume entrenamiento Stage 1 YOLOv11n",
            [
                str(YOLO_CLI),
                "train", "resume",
                f"model={LAST_PT}",
            ],
            LOG_DIR / "training.log",
        )
        if not ok:
            log("Entrenamiento falló -- revisar logs/training.log")
            sys.exit(1)
    else:
        log("SKIP: entrenamiento")

    log("Pipeline completado exitosamente.")


if __name__ == "__main__":
    main()
