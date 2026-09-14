"""
AlpacaVision AI -- Extracción de recortes de ROI (Regions of Interest).

Usa el detector YOLOv11 entrenado para recortar automáticamente las regiones
anatómicas de las imágenes y guardarlas en data/crops/ para entrenar los
clasificadores de anomalías.

Uso:
    python src/data/crop_regions.py \
        --detector models/detector/best.pt \
        --images data/annotated_v3/images/train \
        --output data/crops \
        --conf 0.5
"""

import argparse
from pathlib import Path

import cv2
from tqdm import tqdm

# map a detector class to its destination folder
REGION_MAP = {
    2: "eyes",  # alpaca_eye
    3: "legs",  # alpaca_leg_front
    4: "legs",  # alpaca_leg_rear
}
PADDING = 0.10  # extra margin relative to the crop size


def crop_regions(
    detector_path: str,
    images_dir: str,
    output_dir: str,
    confidence: float = 0.5,
):
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("Instalar: pip install ultralytics")

    model = YOLO(detector_path)
    images_path = Path(images_dir)
    output_path = Path(output_dir)

    for region in ("eyes", "legs"):
        (output_path / region / "normal").mkdir(parents=True, exist_ok=True)
        (output_path / region / "anomaly").mkdir(parents=True, exist_ok=True)

    image_files = list(images_path.glob("*.jpg")) + list(images_path.glob("*.png"))
    print(f"Imágenes a procesar: {len(image_files)}")

    crop_count = {r: 0 for r in ("eyes", "legs")}

    for img_path in tqdm(image_files, desc="Extrayendo recortes"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]
        results = model(img_path, conf=confidence, verbose=False)[0]

        for box in results.boxes:
            cls = int(box.cls[0])
            if cls not in REGION_MAP:
                continue

            region = REGION_MAP[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # Añadir padding
            bw, bh = x2 - x1, y2 - y1
            pad_x, pad_y = int(bw * PADDING), int(bh * PADDING)
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # store under 'normal' by default; the classifier assigns the final class
            crop_name = f"{img_path.stem}_cls{cls}_{x1}_{y1}.jpg"
            out_path = output_path / region / "normal" / crop_name
            cv2.imwrite(str(out_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            crop_count[region] += 1

    print(f"\nRecortes extraídos:")
    for region, count in crop_count.items():
        print(f"  {region}: {count}")
    print(f"\nRevisa y mueve imágenes con anomalías a la subcarpeta 'anomaly/' manualmente")
    print(f"(o importa en Roboflow/CVAT para etiquetado de clasificador)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extraer recortes ROI con detector entrenado")
    parser.add_argument("--detector", type=str, required=True,
                        help="Ruta al modelo detector: models/detector/best.pt")
    parser.add_argument("--images", type=str, default="data/annotated_v3/images/train")
    parser.add_argument("--output", type=str, default="data/crops")
    parser.add_argument("--conf", type=float, default=0.5)
    args = parser.parse_args()

    crop_regions(
        detector_path=args.detector,
        images_dir=args.images,
        output_dir=args.output,
        confidence=args.conf,
    )
