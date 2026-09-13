"""
Dataset preprocessing.

Provides:
  - image integrity checks
  - exact duplicate removal by MD5 hash
  - letterbox resizing, without distortion
  - CLAHE enhancement for altiplano conditions (high UV, haze, side light)
  - per-source statistics

Usage:
    python src/data/preprocess.py --input data/raw --output data/processed
"""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def compute_md5(filepath: Path) -> str:
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def is_valid_image(filepath: Path, min_size: int = 100) -> bool:
    try:
        img = Image.open(filepath)
        img.verify()
        img = Image.open(filepath)
        w, h = img.size
        return w >= min_size and h >= min_size
    except Exception:
        return False


def enhance_altitude_image(image: np.ndarray) -> np.ndarray:
    """CLAHE en espacio LAB para mejorar contraste en condiciones de altiplano."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab_enhanced = cv2.merge([clahe.apply(l), a, b])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


def letterbox(image: np.ndarray, target: int = 640) -> np.ndarray:
    """Redimensiona manteniendo aspecto con padding gris (letterbox)."""
    h, w = image.shape[:2]
    scale = min(target / w, target / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    canvas = np.full((target, target, 3), 114, dtype=np.uint8)
    pad_x = (target - new_w) // 2
    pad_y = (target - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas


def preprocess_dataset(
    input_dirs: list,
    output_dir: str,
    target_size: int = 640,
    apply_enhancement: bool = True,
    remove_duplicates: bool = True,
    min_image_size: int = 100,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    extensions = ("*.jpg", "*.jpeg", "*.png", "*.webp")
    all_images = []
    for d in input_dirs:
        for ext in extensions:
            all_images.extend(Path(d).rglob(ext))

    print(f"Total imágenes encontradas: {len(all_images)}")

    seen_hashes: set = set()
    valid_count = duplicate_count = corrupt_count = 0
    source_counts: dict = {}

    for img_path in tqdm(all_images, desc="Preprocesando"):
        if not is_valid_image(img_path, min_size=min_image_size):
            corrupt_count += 1
            continue

        if remove_duplicates:
            md5 = compute_md5(img_path)
            if md5 in seen_hashes:
                duplicate_count += 1
                continue
            seen_hashes.add(md5)

        img = cv2.imread(str(img_path))
        if img is None:
            corrupt_count += 1
            continue

        if apply_enhancement:
            img = enhance_altitude_image(img)

        img = letterbox(img, target=target_size)

        source = img_path.parent.parent.name
        out_name = f"{source}_{img_path.stem}.jpg"
        cv2.imwrite(str(output_path / out_name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])

        source_counts[source] = source_counts.get(source, 0) + 1
        valid_count += 1

    stats = {
        "total_valid": valid_count,
        "duplicates_removed": duplicate_count,
        "corrupt_removed": corrupt_count,
        "sources": source_counts,
    }
    (output_path / "preprocessing_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\nPreprocesamiento completado:")
    print(f"  Válidas:    {valid_count}")
    print(f"  Duplicadas: {duplicate_count}")
    print(f"  Corruptas:  {corrupt_count}")
    print(f"  Por fuente: {source_counts}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocesar dataset de alpacas")
    parser.add_argument("--input", nargs="+", default=["data/raw"])
    parser.add_argument("--output", type=str, default="data/processed")
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--no-enhancement", action="store_true")
    parser.add_argument("--keep-duplicates", action="store_true")
    args = parser.parse_args()

    preprocess_dataset(
        input_dirs=args.input,
        output_dir=args.output,
        target_size=args.size,
        apply_enhancement=not args.no_enhancement,
        remove_duplicates=not args.keep_duplicates,
    )
