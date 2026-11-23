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
_CROP_SUFFIX = re.compile(r"_(?:cls\d+_\d+_\d+|eye_\d+_c[\d.]+)\.(?:jpg|jpeg|png)$")
_RF_PHOTO = re.compile(r"alpaca-[a-z0-9]+_+([0-9a-f]{16,32})_(?:jpg|jpeg|png)\.rf\.")
_INAT_PHOTO = re.compile(r"inat_(\d+)_(\d+)")


def original_stem(filename: str) -> str:
    """Return the source image name with the `aug_NN_` prefix stripped."""
    return _AUG_RE.sub("", filename)


def is_augmented(filename: str) -> bool:
    return bool(_AUG_RE.match(filename))


def source_file(filename: str) -> str:
    """Return the name of the photograph a crop was cut from."""
    return _CROP_SUFFIX.sub("", original_stem(filename).replace("pseudo_", ""))


def source_photo(filename: str) -> str:
    """Return a photograph identity shared by every export and crop of the same photograph."""
    src = source_file(filename)
    m = _RF_PHOTO.match(src)
    if m:
        return "rf:" + m.group(1)
    m = _INAT_PHOTO.match(src)
    if m:
        return f"inat:{m.group(1)}_{m.group(2)}"
    return "file:" + src


def build_photo_split(
    data_dir: str | Path,
    seed: int = 42,
    test_frac: float = 0.15,
    val_frac: float = 0.15,
    originals_only_eval: bool = True,
) -> dict:
    """Assign whole photographs, with every crop and augmented copy, to one partition.

    Photographs are stratified by whether any of their crops is labelled anomaly, so each
    partition keeps a similar share of anomaly crops.
    """
    data_dir = Path(data_dir)
    classes = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    class_to_idx = {c: i for i, c in enumerate(classes)}
    rng = np.random.default_rng(seed)

    groups: dict[str, list[tuple[Path, int]]] = defaultdict(list)
    for cls in classes:
        for p in (data_dir / cls).iterdir():
            if p.suffix.lower() in IMG_EXTS:
                groups[source_photo(p.name)].append((p, class_to_idx[cls]))

    anomaly = class_to_idx.get("anomaly", 1)
    strata = defaultdict(list)
    for k in sorted(groups):
        strata[any(l == anomaly for _, l in groups[k])].append(k)

    keys = {"train": set(), "val": set(), "test": set()}
    for _, members in sorted(strata.items()):
        rng.shuffle(members)
        n = len(members)
        n_test = max(1, int(round(n * test_frac)))
        n_val = max(1, int(round(n * val_frac)))
        keys["test"].update(members[:n_test])
        keys["val"].update(members[n_test:n_test + n_val])
        keys["train"].update(members[n_test + n_val:])

    out = {"train": [], "val": [], "test": []}
    for part, ks in keys.items():
        for k in sorted(ks):
            for p, label in groups[k]:
                if part != "train" and originals_only_eval and is_augmented(p.name):
                    continue
                out[part].append((str(p), label))
    rng.shuffle(out["train"])

    return {
        "data_dir": str(data_dir), "seed": seed, "test_frac": test_frac, "val_frac": val_frac,
        "originals_only_eval": originals_only_eval, "grouping": "source_photo",
        "classes": classes, "class_to_idx": class_to_idx,
        **out,
        "n_train": len(out["train"]), "n_val": len(out["val"]), "n_test": len(out["test"]),
        "photographs": {k: len(v) for k, v in keys.items()},
    }


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


def assert_no_leakage(split: dict, key=None) -> None:
    """Raise if any source group appears in more than one partition."""
    key = key or original_stem

    def stems(bucket):
        return {key(Path(p).name) for p, _ in bucket}
    tr, va, te = stems(split["train"]), stems(split["val"]), stems(split["test"])
    leaks = {
        "train_val": tr & va,
        "train_test": tr & te,
        "val_test": va & te,
    }
    bad = {k: len(v) for k, v in leaks.items() if v}
    if bad:
        raise AssertionError(f"Data leakage detectado entre splits: {bad}")
