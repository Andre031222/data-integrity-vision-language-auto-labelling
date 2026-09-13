"""Consolida todos los datasets de Roboflow en uno solo."""

import random
import shutil
from pathlib import Path

ROBOFLOW_DIR = Path("data/raw/roboflow")
OUT_DIR      = Path("data/annotated")
SEED         = 42
MIN_BOX_AREA = 0.01   # descartar cajas < 1% del área imagen (ruido)
SPLITS       = {"train": 0.70, "val": 0.15, "test": 0.15}

# Datasets a incluir (excluimos alpaca-zehtv por calidad muy baja)
INCLUDE = [
    "alpaca-5jmfl", "alpaca-8baig", "alpaca-epqna", "alpaca-gkbmi",
    "alpaca-ibscl", "alpaca-lls3s", "alpaca-nrzos", "alpaca-ofxv9",
    "alpaca-xqfiw",
]

def filter_label(txt_path: Path) -> list[str]:
    """Devuelve líneas válidas: clase->0, descarta cajas tiny."""
    good = []
    for line in txt_path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        w, h = float(parts[3]), float(parts[4])
        if w * h < MIN_BOX_AREA:
            continue
        good.append(f"0 {parts[1]} {parts[2]} {parts[3]} {parts[4]}")
    return good

def collect_pairs() -> list[tuple[Path, Path]]:
    pairs = []
    for ds_name in INCLUDE:
        ds_dir = ROBOFLOW_DIR / ds_name
        if not ds_dir.exists():
            print(f"  SKIP (no existe): {ds_name}")
            continue
        imgs = list(ds_dir.rglob("*.jpg")) + list(ds_dir.rglob("*.png"))
        for img in imgs:
            lbl = img.parent.parent / "labels" / (img.stem + ".txt")
            if not lbl.exists():
                lbl = img.with_suffix(".txt")
            if lbl.exists():
                pairs.append((img, lbl))
    return pairs

def main():
    for split in ("train", "val", "test"):
        (OUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    pairs = collect_pairs()
    print(f"Pares imagen+label encontrados: {len(pairs)}")

    random.seed(SEED)
    random.shuffle(pairs)

    n = len(pairs)
    n_train = int(n * SPLITS["train"])
    n_val   = int(n * SPLITS["val"])

    split_pairs = {
        "train": pairs[:n_train],
        "val":   pairs[n_train:n_train + n_val],
        "test":  pairs[n_train + n_val:],
    }

    stats = {}
    skipped = 0
    for split, sp in split_pairs.items():
        count = 0
        for img_path, lbl_path in sp:
            lines = filter_label(lbl_path)
            if not lines:
                skipped += 1
                continue
            # Nombre único: dataset + stem original
            ds = img_path.parent.parent.parent.name
            stem = f"{ds}_{img_path.stem}"
            suffix = img_path.suffix

            shutil.copy2(img_path, OUT_DIR / "images" / split / (stem + suffix))
            (OUT_DIR / "labels" / split / (stem + ".txt")).write_text("\n".join(lines))
            count += 1
        stats[split] = count
        print(f"  {split:5s}: {count} imágenes")

    print(f"  Filtradas (sin cajas válidas): {skipped}")
    print(f"\nDataset listo en: {OUT_DIR}")
    print(f"Total entrenamiento: {stats['train']} imgs")

if __name__ == "__main__":
    main()
