"""
AlpacaVision AI -- División estratificada del dataset (70/15/15).

Uso:
    python src/data/split_dataset.py \
        --images data/processed \
        --labels data/annotated/labels_raw \
        --output data/annotated
"""

import argparse
import json
import random
import shutil
from pathlib import Path


def split_dataset(
    images_dir: str,
    labels_dir: str,
    output_dir: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Los ratios deben sumar 1.0"

    random.seed(seed)
    images_path = Path(images_dir)
    labels_path = Path(labels_dir)
    output_path = Path(output_dir)

    for split in ("train", "val", "test"):
        (output_path / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_path / "labels" / split).mkdir(parents=True, exist_ok=True)

    image_files = list(images_path.glob("*.jpg")) + list(images_path.glob("*.png"))
    paired = [
        (img, labels_path / (img.stem + ".txt"))
        for img in image_files
        if (labels_path / (img.stem + ".txt")).exists()
    ]

    print(f"Pares imagen-etiqueta: {len(paired)}")
    if not paired:
        print("No se encontraron pares. Verifica rutas --images y --labels.")
        return

    random.shuffle(paired)
    n = len(paired)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits = {
        "train": paired[:n_train],
        "val":   paired[n_train:n_train + n_val],
        "test":  paired[n_train + n_val:],
    }

    stats = {"seed": seed, "total": n}
    for name, pairs in splits.items():
        for img_path, lbl_path in pairs:
            shutil.copy2(img_path, output_path / "images" / name / img_path.name)
            shutil.copy2(lbl_path, output_path / "labels" / name / lbl_path.name)
        stats[name] = len(pairs)
        print(f"  {name:5s}: {len(pairs):4d} ({len(pairs)/n*100:.1f}%)")

    (output_path / "split_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"\nDataset dividido en: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=str, default="data/processed")
    parser.add_argument("--labels", type=str, default="data/annotated/labels_raw")
    parser.add_argument("--output", type=str, default="data/annotated")
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--val",   type=float, default=0.15)
    parser.add_argument("--test",  type=float, default=0.15)
    parser.add_argument("--seed",  type=int,   default=42)
    args = parser.parse_args()

    split_dataset(
        images_dir=args.images,
        labels_dir=args.labels,
        output_dir=args.output,
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
        seed=args.seed,
    )
