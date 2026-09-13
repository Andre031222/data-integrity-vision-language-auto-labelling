"""Clean up the stage-1 detector dataset."""

import argparse
import hashlib
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
SRC = ROOT / "data" / "annotated"
DST = ROOT / "data" / "annotated_clean"
SPLITS = ("train", "val", "test")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
SEED = 42


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def collect_unique():
    """Return a dict md5 -> (img_path, label_path or None), one entry per unique image."""
    unique = {}
    dup = 0
    for split in SPLITS:
        img_dir = SRC / "images" / split
        lbl_dir = SRC / "labels" / split
        if not img_dir.exists():
            continue
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in IMG_EXTS:
                continue
            h = md5(img)
            if h in unique:
                dup += 1
                continue
            label = lbl_dir / (img.stem + ".txt")
            unique[h] = (img, label if label.exists() else None)
    return unique, dup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--val-frac", type=float, default=0.15)
    args = ap.parse_args()

    unique, dup = collect_unique()
    keys = sorted(unique.keys())
    rng = np.random.default_rng(SEED)
    rng.shuffle(keys)
    n = len(keys)
    n_test = int(round(n * args.test_frac))
    n_val = int(round(n * args.val_frac))
    assign = {
        "test": keys[:n_test],
        "val": keys[n_test:n_test + n_val],
        "train": keys[n_test + n_val:],
    }

    print(f"Imagenes totales en origen: {sum(1 for s in SPLITS for p in (SRC/'images'/s).iterdir() if p.suffix.lower() in IMG_EXTS)}")
    print(f"Imagenes UNICAS (MD5): {n} | duplicados descartados: {dup}")
    n_labels = sum(1 for _, (_, lbl) in unique.items() if lbl)
    print(f"Con label YOLO: {n_labels} | sin label (background): {n - n_labels}")
    for s in SPLITS:
        print(f"  {s}: {len(assign[s])} imagenes")

    if args.dry_run:
        print("\n(DRY-RUN) No se escribio nada.")
        return

    if DST.exists():
        shutil.rmtree(DST)
    for s in SPLITS:
        (DST / "images" / s).mkdir(parents=True, exist_ok=True)
        (DST / "labels" / s).mkdir(parents=True, exist_ok=True)

    for s in SPLITS:
        for h in assign[s]:
            img, lbl = unique[h]
            shutil.copy2(img, DST / "images" / s / img.name)
            if lbl:
                shutil.copy2(lbl, DST / "labels" / s / (img.stem + ".txt"))

    # Verificacion anti-leakage
    def md5set(s):
        return {md5(p) for p in (DST / "images" / s).iterdir() if p.suffix.lower() in IMG_EXTS}
    tr, va, te = md5set("train"), md5set("val"), md5set("test")
    leaks = (tr & te) | (tr & va) | (va & te)
    assert not leaks, f"Leakage residual: {len(leaks)} imagenes"
    print(f"\nVerificacion anti-leakage OK (0 imagenes compartidas entre splits)")

    yaml_path = ROOT / "config" / "dataset_stage1_clean.yaml"
    yaml_path.write_text(
        "# AlpacaVision AI -- Etapa 1 LIMPIA (sin duplicados ni leakage)\n"
        "path: ./data/annotated_clean\n"
        "train: images/train\nval:   images/val\ntest:  images/test\n\n"
        "nc: 1\nnames:\n  0: alpaca\n", encoding="utf-8")
    print(f"Dataset limpio: {DST}")
    print(f"Config: {yaml_path}")


if __name__ == "__main__":
    main()
