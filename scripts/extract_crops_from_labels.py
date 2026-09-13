"""AlpacaVision AI -- Extrae crops de ojos y patas desde auto_labels Stage 2."""

from pathlib import Path
import cv2
from tqdm import tqdm

IMAGES_DIR  = Path("data/annotation_batch")
LABELS_DIR  = Path("data/auto_labels")
OUTPUT_DIR  = Path("data/crops")
PADDING     = 0.12   # margen adicional proporcional al box

REGION_MAP = {2: "eyes", 3: "legs", 4: "legs"}

# Confianza mínima para incluir un recorte (leída del comentario inline)
MIN_CONF = 0.30


def parse_label_line(line: str):
    """Parsea línea YOLO con comentario opcional: 'cls cx cy w h  # conf=X phrase=Y'"""
    raw = line.split("#")[0].strip()
    if not raw:
        return None
    parts = raw.split()
    if len(parts) < 5:
        return None
    cls_id = int(parts[0])
    cx, cy, w, h = map(float, parts[1:5])
    # Intentar leer conf del comentario
    conf = 1.0
    if "#" in line:
        comment = line.split("#", 1)[1]
        for token in comment.split():
            if token.startswith("conf="):
                try:
                    conf = float(token.split("=")[1])
                except ValueError:
                    pass
    return cls_id, cx, cy, w, h, conf


def extract_crops():
    for region in ("eyes", "legs"):
        (OUTPUT_DIR / region / "normal").mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / region / "anomaly").mkdir(parents=True, exist_ok=True)

    label_files = sorted(LABELS_DIR.glob("*.txt"))
    label_files = [f for f in label_files if f.name != "auto_label_manifest.json"
                   and f.stat().st_size > 0]

    counts = {"eyes": 0, "legs": 0, "skipped": 0}

    for lbl_path in tqdm(label_files, desc="Extrayendo crops"):
        img_path = IMAGES_DIR / (lbl_path.stem + ".jpg")
        if not img_path.exists():
            img_path = IMAGES_DIR / (lbl_path.stem + ".png")
        if not img_path.exists():
            counts["skipped"] += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            counts["skipped"] += 1
            continue
        img_h, img_w = img.shape[:2]

        for line in lbl_path.read_text().splitlines():
            parsed = parse_label_line(line)
            if parsed is None:
                continue
            cls_id, cx, cy, bw, bh, conf = parsed

            if cls_id not in REGION_MAP:
                continue
            if conf < MIN_CONF:
                continue

            region = REGION_MAP[cls_id]

            # Convertir YOLO norm -> píxeles con padding
            pad_x = bw * PADDING
            pad_y = bh * PADDING
            x1 = max(0.0, cx - bw / 2 - pad_x)
            y1 = max(0.0, cy - bh / 2 - pad_y)
            x2 = min(1.0, cx + bw / 2 + pad_x)
            y2 = min(1.0, cy + bh / 2 + pad_y)

            px1 = int(x1 * img_w)
            py1 = int(y1 * img_h)
            px2 = int(x2 * img_w)
            py2 = int(y2 * img_h)

            crop = img[py1:py2, px1:px2]
            if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8:
                continue

            crop_name = f"{lbl_path.stem}_cls{cls_id}_{px1}_{py1}.jpg"
            out = OUTPUT_DIR / region / "normal" / crop_name
            cv2.imwrite(str(out), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            counts[region] += 1

    print(f"\nCrops extraídos:")
    print(f"  Ojos (eyes/normal):  {counts['eyes']}")
    print(f"  Patas (legs/normal): {counts['legs']}")
    print(f"  Saltados:            {counts['skipped']}")
    print(f"\nProximo paso:")
    print(f"  El semillero debe mover imágenes con anomalías a:")
    print(f"    data/crops/eyes/anomaly/")
    print(f"    data/crops/legs/anomaly/")
    print(f"  Con >= 50 ejemplos anomaly, entrenar clasificadores con:")
    print(f"    venv/Scripts/python.exe scripts/run_pipeline.py --skip-autolabel --skip-upload --skip-training")


if __name__ == "__main__":
    extract_crops()
