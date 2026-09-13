"""
Group-aware train/val/test split for the ocular crops.

The problem it solves: the anomaly crops were multiplied offline by
`augment_anomaly_crops.py`, producing `aug_NN_<original>.jpg` files. Splitting
those per file puts augmented versions of the SAME source image in both train
and test, which leaks data and inflates every metric.

This module groups files by source image, stripping the `aug_NN_` prefix, and
assigns whole GROUPS to a partition, so every variant of an image stays on one
side. By default validation and test hold only unaugmented originals, so
evaluation always runs on real data.

Usage:
    from src.data.group_split import build_group_split
    split = build_group_split("data/crops/eyes", seed=42,
                              test_frac=0.15, val_frac=0.15)
    split["train"]  # list of (path, label_idx)
    split["val"]    # same, originals only
    split["test"]   # same, originals only
    split["class_to_idx"]
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
_AUG_RE = re.compile(r"^aug_\d+_")


def original_stem(filename: str) -> str:
    """Return the source image name with the `aug_NN_` prefix stripped."""
    return _AUG_RE.sub("", filename)


def is_augmented(filename: str) -> bool:
    return bool(_AUG_RE.match(filename))


def build_group_split(
    data_dir: str | Path,
    seed: int = 42,
    test_frac: float = 0.15,
    val_frac: float = 0.15,
    originals_only_eval: bool = True,
) -> dict:
    """
    Construye un split honesto a nivel de grupo (imagen original).

    Args:
        data_dir: carpeta con subcarpetas por clase (ImageFolder-style).
        seed: semilla reproducible.
        test_frac / val_frac: fraccion de GRUPOS por clase para test / val.
        originals_only_eval: si True, val y test solo contienen originales;
            las augmentadas de esos grupos se descartan (no se filtran a train,
            para no contaminar). Train conserva originales + augmentadas.

    Returns:
        dict con: class_to_idx, classes, y listas train/val/test de tuplas
        (ruta_absoluta:str, label_idx:int), mas conteos y metadatos.
    """
    data_dir = Path(data_dir)
    classes = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    class_to_idx = {c: i for i, c in enumerate(classes)}
    rng = np.random.default_rng(seed)

    train, val, test = [], [], []
    per_class_counts = {}

    for cls in classes:
        label = class_to_idx[cls]
        cls_dir = data_dir / cls
        files = [p for p in cls_dir.iterdir() if p.suffix.lower() in IMG_EXTS]

        # group by source image
        groups: dict[str, list[Path]] = defaultdict(list)
        for p in files:
            groups[original_stem(p.name)].append(p)

        group_keys = sorted(groups.keys())
        rng.shuffle(group_keys)
        n = len(group_keys)
        n_test = max(1, int(round(n * test_frac)))
        n_val = max(1, int(round(n * val_frac)))
        test_keys = set(group_keys[:n_test])
        val_keys = set(group_keys[n_test:n_test + n_val])
        train_keys = set(group_keys[n_test + n_val:])

        def _add(keys, bucket, eval_split):
            n_orig = n_aug = 0
            for k in keys:
                for p in groups[k]:
                    if eval_split and originals_only_eval and is_augmented(p.name):
                        continue  # val/test: solo originales reales
                    bucket.append((str(p), label))
                    if is_augmented(p.name):
                        n_aug += 1
                    else:
                        n_orig += 1
            return n_orig, n_aug

        tr = _add(train_keys, train, eval_split=False)
        va = _add(val_keys, val, eval_split=True)
        te = _add(test_keys, test, eval_split=True)
        per_class_counts[cls] = {
            "groups": n,
            "train": {"orig": tr[0], "aug": tr[1]},
            "val": {"orig": va[0], "aug": va[1]},
            "test": {"orig": te[0], "aug": te[1]},
        }

    rng.shuffle(train)  # mezclar clases en train

    return {
        "data_dir": str(data_dir),
        "seed": seed,
        "test_frac": test_frac,
        "val_frac": val_frac,
        "originals_only_eval": originals_only_eval,
        "classes": classes,
        "class_to_idx": class_to_idx,
        "train": train,
        "val": val,
        "test": test,
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "per_class": per_class_counts,
    }


def assert_no_leakage(split: dict) -> None:
    """Raise if any source group appears in more than one partition."""
    def stems(bucket):
        return {original_stem(Path(p).name) for p, _ in bucket}
    tr, va, te = stems(split["train"]), stems(split["val"]), stems(split["test"])
    leaks = {
        "train_val": tr & va,
        "train_test": tr & te,
        "val_test": va & te,
    }
    bad = {k: len(v) for k, v in leaks.items() if v}
    if bad:
        raise AssertionError(f"Data leakage detectado entre splits: {bad}")
