"""AlpacaVision AI -- Augmentacion offline de crops anomalos."""

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent.parent
CROPS_DIR = ROOT / "data" / "crops"


def augment_image(img: np.ndarray, aug_id: int) -> np.ndarray:
    """Aplica la augmentacion numero aug_id (0-7) a la imagen."""
    h, w = img.shape[:2]

    if aug_id == 0:   # flip horizontal
        return cv2.flip(img, 1)
    elif aug_id == 1:  # flip vertical
        return cv2.flip(img, 0)
    elif aug_id == 2:  # rot 90
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif aug_id == 3:  # rot 180
        return cv2.rotate(img, cv2.ROTATE_180)
    elif aug_id == 4:  # rot 270
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif aug_id == 5:  # brillo +30%
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.3, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    elif aug_id == 6:  # contraste
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab[:, :, 0] = np.clip(lab[:, :, 0] * 1.3, 0, 255)
        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    elif aug_id == 7:  # blur leve (suavizar artefactos de resize)
        return cv2.GaussianBlur(img, (3, 3), 0)
    return img


def process_task(task: str, factor: int, clean: bool):
    anomaly_dir = CROPS_DIR / task / "anomaly"
    if not anomaly_dir.exists():
        print(f"[{task}] No existe {anomaly_dir}")
        return

    # Contar originales (sin prefijo aug_)
    originals = [f for f in sorted(anomaly_dir.glob("*.jpg")) if not f.stem.startswith("aug_")]
    augmented  = [f for f in sorted(anomaly_dir.glob("aug_*.jpg"))]

    if clean:
        removed = 0
        for f in augmented:
            f.unlink()
            removed += 1
        print(f"[{task}] Borradas {removed} imagenes augmentadas")
        return

    print(f"\n[{task}] Originales: {len(originals)} | Augmentadas previas: {len(augmented)}")

    created = 0
    skipped = 0
    for img_path in originals:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  ERROR leyendo {img_path.name}")
            continue

        for aug_id in range(factor):
            out_name = f"aug_{aug_id:02d}_{img_path.name}"
            out_path = anomaly_dir / out_name
            if out_path.exists():
                skipped += 1
                continue
            augmented_img = augment_image(img, aug_id)
            cv2.imwrite(str(out_path), augmented_img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            created += 1

    total_anomaly = len(list(anomaly_dir.glob("*.jpg")))
    normal_count  = len(list((CROPS_DIR / task / "normal").glob("*.jpg")))
    print(f"[{task}] Creadas: {created} | Saltadas: {skipped}")
    print(f"[{task}] Total anomaly: {total_anomaly} | Normal: {normal_count} | Ratio: {normal_count/total_anomaly:.1f}:1")


def main():
    parser = argparse.ArgumentParser(description="Augmentacion offline de crops anomalos")
    parser.add_argument("--task",   choices=["eyes", "legs", "both"], default="eyes")
    parser.add_argument("--factor", type=int, default=8,
                        help="Augmentaciones por imagen original (max 8)")
    parser.add_argument("--clean",  action="store_true",
                        help="Borrar augmentadas previas")
    args = parser.parse_args()

    tasks = ["eyes", "legs"] if args.task == "both" else [args.task]
    for task in tasks:
        process_task(task, min(args.factor, 8), args.clean)

    print("\nListo. Vuelve a entrenar:")
    print("  venv/Scripts/python.exe src/training/train_classifier.py \\")
    print("    --task eyes --data_dir data/crops/eyes --output models/classifier/eyes \\")
    print("    --epochs 100 --batch 16 --lr 0.0001 --patience 25 --device 0")


if __name__ == "__main__":
    main()
