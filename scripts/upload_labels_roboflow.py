"""AlpacaVision AI -- Sube pseudo-labels a Roboflow Stage 2."""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

IMAGES_DIR    = Path("data/annotation_batch")
LABELS_DIR    = Path("data/auto_labels")
MANIFEST_FILE = LABELS_DIR / "auto_label_manifest.json"

# stage-2 Roboflow target; adjust if the workspace or project changes
WORKSPACE  = "andre-nftgn"
PROJECT_ID = "alpacavision-stage2"


def load_manifest() -> dict:
    if not MANIFEST_FILE.exists():
        print(f"ERROR: no existe {MANIFEST_FILE}. Corre primero scripts/auto_label.py")
        sys.exit(1)
    with open(MANIFEST_FILE) as f:
        return json.load(f)


def _clean_label_file(label_path: Path, tmp_dir: Path) -> Path:
    """Return a copy of the label file in tmp_dir with inline comments stripped."""
    lines = []
    for raw in label_path.read_text().splitlines():
        stripped = raw.split("#")[0].strip()  # quitar comentarios
        if stripped:
            lines.append(stripped)
    clean_path = tmp_dir / label_path.name
    clean_path.write_text("\n".join(lines))
    return clean_path


def upload_image_with_annotation(rf_project, img_path: Path, label_path: Path,
                                 tmp_dir: Path) -> bool:
    """Sube imagen + anotación YOLO a Roboflow (limpia comentarios inline)."""
    try:
        clean = _clean_label_file(label_path, tmp_dir)
        rf_project.upload(
            image_path=str(img_path),
            annotation_path=str(clean),
            annotation_labelmap={
                "0": "alpaca_body",
                "1": "alpaca_head",
                "2": "alpaca_eye",
                "3": "alpaca_leg_front",
                "4": "alpaca_leg_rear",
            },
            split="train",
            is_prediction=True,   # flag as a pseudo-label pending review
            overwrite=False,
        )
        return True
    except Exception as e:
        print(f"  ERROR subiendo {img_path.name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-boxes", type=int, default=1,
                        help="Mínimo de cajas para incluir imagen (default: 1)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Simular sin subir")
    args = parser.parse_args()

    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        print("ERROR: ROBOFLOW_API_KEY no encontrada en .env")
        sys.exit(1)

    manifest = load_manifest()
    stats = manifest["stats"]
    images = manifest["images"]

    print(f"Manifest cargado: {stats['total']} imágenes totales")
    print(f"  Con detecciones: {stats['with_labels']}")
    print(f"  Sin detecciones: {stats['empty']}")
    print(f"  Errores:         {stats['errors']}")
    print(f"\nFiltro: >= {args.min_boxes} caja(s) por imagen")

    to_upload = [
        entry for entry in images
        if entry["boxes"] >= args.min_boxes
    ]
    print(f"Imágenes a subir: {len(to_upload)}")

    if args.dry_run:
        print("\n[DRY RUN] Imágenes que se subirían:")
        for e in to_upload[:10]:
            print(f"  {e['filename']}  ({e['boxes']} cajas)")
        if len(to_upload) > 10:
            print(f"  ... y {len(to_upload) - 10} más")
        return

    # Conectar a Roboflow
    from roboflow import Roboflow
    import tempfile, shutil
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(WORKSPACE).project(PROJECT_ID)
    print(f"\nConectado a: {WORKSPACE}/{PROJECT_ID}")

    ok = 0
    failed = 0
    skipped = 0

    tmp_dir = Path(tempfile.mkdtemp(prefix="alpaca_labels_"))
    try:
        for entry in to_upload:
            img_path   = IMAGES_DIR / entry["filename"]
            label_path = LABELS_DIR / entry["label_file"]

            if not img_path.exists():
                print(f"  SKIP (imagen no encontrada): {entry['filename']}")
                skipped += 1
                continue
            if not label_path.exists() or label_path.stat().st_size == 0:
                skipped += 1
                continue

            success = upload_image_with_annotation(project, img_path, label_path, tmp_dir)
            if success:
                ok += 1
            else:
                failed += 1

            # report progress every 25 images
            done = ok + failed + skipped
            if done % 25 == 0:
                print(f"  Progreso: {done}/{len(to_upload)}  (ok={ok}, err={failed})")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"\n{'='*50}")
    print(f"Subida completada:")
    print(f"  Subidas OK: {ok}")
    print(f"  Errores:    {failed}")
    print(f"  Saltadas:   {skipped}")
    print(f"\nRevisar en: https://app.roboflow.com/{WORKSPACE}/{PROJECT_ID}/annotate")


if __name__ == "__main__":
    main()
