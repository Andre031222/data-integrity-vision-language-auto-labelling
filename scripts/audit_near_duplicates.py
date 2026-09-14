#!/usr/bin/env python3
"""Measure how much of a split is contaminated under three notions of duplicate."""
import argparse, hashlib, json
from pathlib import Path

import numpy as np
from PIL import Image
import imagehash

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TRANSFORMS = (
    lambda im: im,
    lambda im: im.transpose(Image.ROTATE_90),
    lambda im: im.transpose(Image.ROTATE_180),
    lambda im: im.transpose(Image.ROTATE_270),
    lambda im: im.transpose(Image.FLIP_LEFT_RIGHT),
    lambda im: im.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_90),
    lambda im: im.transpose(Image.FLIP_TOP_BOTTOM),
    lambda im: im.transpose(Image.FLIP_TOP_BOTTOM).transpose(Image.ROTATE_90),
)


def bits(v):
    return np.array([(v >> k) & 1 for k in range(64)], np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/annotated_v3_srcfilter")
    ap.add_argument("--taus", type=int, nargs="+", default=[0, 2, 6, 10])
    ap.add_argument("--out", default="outputs/figures/neardup_audit_v3.json")
    a = ap.parse_args()

    root = Path(a.root) / "images"
    paths, split = [], []
    for sp in ("train", "val", "test"):
        for f in sorted((root / sp).iterdir()):
            if f.suffix.lower() in EXT:
                paths.append(f); split.append(sp)
    split = np.array(split)
    n = len(paths)
    counts = {sp: int((split == sp).sum()) for sp in ("train", "val", "test")}
    print(f"root  : {a.root}")
    print(f"images: {n}  {counts}")

    md5 = {}
    V = np.zeros((n, len(TRANSFORMS), 64), np.uint8)
    for i, f in enumerate(paths):
        md5.setdefault(hashlib.md5(f.read_bytes()).hexdigest(), []).append(i)
        im = Image.open(f).convert("RGB")
        for t, fn in enumerate(TRANSFORMS):
            V[i, t] = bits(int(str(imagehash.phash(fn(im))), 16))
    dup_groups = sum(1 for v in md5.values() if len(v) > 1)
    print(f"exact duplicate groups remaining: {dup_groups}")

    base = V[:, 0, :].astype(np.int16)
    ob = base.sum(1)

    def dmat(dihedral):
        if not dihedral:
            return ob[:, None] + ob[None, :] - 2 * (base @ base.T)
        d = np.full((n, n), 127, np.int16)
        for t in range(V.shape[1]):
            y = V[:, t, :].astype(np.int16)
            d = np.minimum(d, y.sum(1)[:, None] + ob[None, :] - 2 * (y @ base.T))
        return np.minimum(d, d.T)

    tr, te = split == "train", split == "test"
    n_test = int(te.sum())
    out = {"root": a.root, "images": n, "per_split": counts,
           "exact_duplicate_groups": dup_groups, "n_test": n_test, "results": {}}

    print(f"\n{'hash':<16}{'tau':>4}{'pairs':>8}{'train-test':>12}{'contaminated test':>20}")
    for name, dih in (("perceptual", False), ("dihedral", True)):
        d = dmat(dih)
        out["results"][name] = {}
        for tau in a.taus:
            m = d <= tau
            pairs = int(np.triu(m, 1).sum())
            tt = int(np.triu(m & ((tr[:, None] & te[None, :]) |
                                  (te[:, None] & tr[None, :])), 1).sum())
            contam = int(((m & tr[None, :])[te].sum(1) > 0).sum())
            pct = round(100 * contam / n_test, 1)
            print(f"{name:<16}{tau:>4}{pairs:>8}{tt:>12}{f'{contam} ({pct}%)':>20}")
            out["results"][name][tau] = {"pairs": pairs, "train_test_pairs": tt,
                                         "contaminated_test": contam, "pct": pct}

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
