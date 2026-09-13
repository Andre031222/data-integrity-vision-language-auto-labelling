"""Extract crops from additional imagery (llamas, vicunas, further alpacas)."""

import argparse
import sys
from pathlib import Path

import cv2
from tqdm import tqdm
from ultralytics import YOLO

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

MODEL_PATH  = ROOT / "models" / "detector" / "best.pt"
CROPS_EYES  = ROOT / "data" / "crops" / "eyes"  / "normal"
CROPS_LEGS  = ROOT / "data" / "crops" / "legs"  / "normal"
MIN_SIZE    = 32   # minimum side in pixels for a crop to be valid
CONF_THRESH = 0.35


def extract_crops(source_dir: Path, skip_existing: bool, max_images: int):
    if not MODEL_PATH.exists():
        print(f"ERROR: Modelo no encontrado en {MODEL_PATH}")
        sys.exit(1)

    CROPS_EYES.mkdir(parents=True, exist_ok=True)
    CROPS_LEGS.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(MODEL_PATH))

    images = sorted(source_dir.glob("*.jpg")) + sorted(source_dir.glob("*.png"))
    if max_images:
        images = images[:max_images]

    print(f"\nFuente: {source_dir} | {len(images)} imagenes")
    print(f"Destino ojos: {CROPS_EYES}")
    print(f"Destino patas: {CROPS_LEGS}")

    # load the class names from the model
    class_names = model.names  # {0: 'eye', 1: 'leg', ...}
    print(f"Clases del detector: {class_names}")

    stats = {"eyes": 0, "legs": 0, "skipped": 0, "no_detection": 0}

    for img_path in tqdm(images, desc="Extrayendo crops"):
        # skip if already processed
        prefix = img_path.stem
        existing_eye = list(CROPS_EYES.glob(f"{prefix}_*.jpg"))
        existing_leg = list(CROPS_LEGS.glob(f"{prefix}_*.jpg"))

        if skip_existing and (existing_eye or existing_leg):
            stats["skipped"] += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        results = model(img, conf=CONF_THRESH, verbose=False)
        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0:
            stats["no_detection"] += 1
            continue

        eye_count = leg_count = 0
        for box in boxes:
            cls_id   = int(box.cls[0].item())
            cls_name = class_names.get(cls_id, "").lower()
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            bw = x2 - x1
            bh = y2 - y1

            if bw < MIN_SIZE or bh < MIN_SIZE:
                continue

            # prefer the model's own eye and leg classes when it has them
            if "eye" in cls_name:
                pad_x = int(bw * 0.10); pad_y = int(bh * 0.10)
                crop = img[max(0,y1-pad_y):min(h,y2+pad_y), max(0,x1-pad_x):min(w,x2+pad_x)]
                if crop.shape[0] >= MIN_SIZE and crop.shape[1] >= MIN_SIZE:
                    cv2.imwrite(str(CROPS_EYES / f"{prefix}_eye_{eye_count}.jpg"), crop)
                    eye_count += 1; stats["eyes"] += 1
            elif "leg" in cls_name or "pata" in cls_name:
                pad_x = int(bw * 0.10); pad_y = int(bh * 0.10)
                crop = img[max(0,y1-pad_y):min(h,y2+pad_y), max(0,x1-pad_x):min(w,x2+pad_x)]
                if crop.shape[0] >= MIN_SIZE and crop.shape[1] >= MIN_SIZE:
                    cv2.imwrite(str(CROPS_LEGS / f"{prefix}_leg_{leg_count}.jpg"), crop)
                    leg_count += 1; stats["legs"] += 1
            else:
                # generic model detects only "alpaca", so fall back to an anatomical estimate
                eye_left = img[
                    max(0, y1 + int(bh * 0.05)) : min(h, y1 + int(bh * 0.30)),
                    max(0, x1 + int(bw * 0.05)) : min(w, x1 + int(bw * 0.35)),
                ]
                # Ojo derecho: esquina superior derecha
                eye_right = img[
                    max(0, y1 + int(bh * 0.05)) : min(h, y1 + int(bh * 0.30)),
                    max(0, x1 + int(bw * 0.65)) : min(w, x2 - int(bw * 0.05)),
                ]
                # Pata delantera izquierda
                leg_fl = img[
                    max(0, y1 + int(bh * 0.60)) : min(h, y2),
                    max(0, x1 + int(bw * 0.05)) : min(w, x1 + int(bw * 0.35)),
                ]
                # Pata delantera derecha
                leg_fr = img[
                    max(0, y1 + int(bh * 0.60)) : min(h, y2),
                    max(0, x1 + int(bw * 0.65)) : min(w, x2 - int(bw * 0.05)),
                ]

                for i, crop in enumerate([eye_left, eye_right]):
                    if crop.shape[0] >= MIN_SIZE and crop.shape[1] >= MIN_SIZE:
                        cv2.imwrite(str(CROPS_EYES / f"{prefix}_eye_{eye_count}.jpg"), crop)
                        eye_count += 1; stats["eyes"] += 1

                for i, crop in enumerate([leg_fl, leg_fr]):
                    if crop.shape[0] >= MIN_SIZE and crop.shape[1] >= MIN_SIZE:
                        cv2.imwrite(str(CROPS_LEGS / f"{prefix}_leg_{leg_count}.jpg"), crop)
                        leg_count += 1; stats["legs"] += 1

    print(f"\nResultado:")
    print(f"  Crops ojos extraidos:  {stats['eyes']}")
    print(f"  Crops patas extraidos: {stats['legs']}")
    print(f"  Sin deteccion:         {stats['no_detection']}")
    print(f"  Saltados (existentes): {stats['skipped']}")
    print(f"\nTotal ojos en dataset: {len(list(CROPS_EYES.glob('*.jpg')))}")
    print(f"Total patas en dataset: {len(list(CROPS_LEGS.glob('*.jpg')))}")


def main():
    parser = argparse.ArgumentParser(description="Extraer crops de nuevas imagenes")
    parser.add_argument("--source", type=str, required=True,
                        help="Directorio con imagenes a procesar")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Saltar imagenes que ya tienen crops extraidos")
    parser.add_argument("--max_images", type=int, default=0,
                        help="Limite de imagenes a procesar (0=todas)")
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        print(f"ERROR: {source} no existe")
        sys.exit(1)

    extract_crops(source, args.skip_existing, args.max_images)


if __name__ == "__main__":
    main()
