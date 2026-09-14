#!/usr/bin/env python3
"""Consolidate the Roboflow sources into a single corpus.

Labels are parsed as boxes or as polygons. Two sources export segmentation
polygons, and reading only their first four values as (cx, cy, w, h) produces a
meaningless box with a near-zero area, which the area filter then discards. That
is how one source was previously judged to be low quality and dropped by hand.
"""
import argparse
import random
from collections import Counter
from pathlib import Path

SEED = 42
EXT = {".jpg", ".jpeg", ".png"}

# Every source that provides YOLO annotations for the single class `alpaca`.
ALL_SOURCES = [
    "alpaca-5jmfl", "alpaca-8baig", "alpaca-epqna", "alpaca-gkbmi",
    "alpaca-ibscl", "alpaca-lls3s", "alpaca-nrzos", "alpaca-ofxv9",
    "alpaca-xqfiw", "alpaca-zehtv",
]


def parse_label(path):
    """Return [(cx, cy, w, h)] from a box or polygon label file."""
    out = []
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        parts = line.split()
        try:
            vals = [float(v) for v in parts[1:]]
        except ValueError:
            continue
        if len(vals) == 4:
            cx, cy, w, h = vals
        elif len(vals) >= 6 and len(vals) % 2 == 0:
            xs, ys = vals[0::2], vals[1::2]
            cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
            w, h = max(xs) - min(xs), max(ys) - min(ys)
        else:
            continue
        x0, x1 = max(0.0, cx - w / 2), min(1.0, cx + w / 2)
        y0, y1 = max(0.0, cy - h / 2), min(1.0, cy + h / 2)
        if x1 - x0 > 1e-4 and y1 - y0 > 1e-4:
            out.append(((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/raw/roboflow")
    ap.add_argument("--out", default="data/annotated_v3")
    ap.add_argument("--sources", nargs="+", default=ALL_SOURCES)
    ap.add_argument("--min-box-area", type=float, default=0.001,
                    help="drop boxes below this fraction of the image area")
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()

    src, out = Path(a.src), Path(a.out)
    pairs, per_src = [], Counter()
    for name in a.sources:
        d = src / name
        if not d.exists():
            print(f"  missing source: {name}")
            continue
        for img in sorted(d.rglob("*")):
            if img.suffix.lower() not in EXT or img.parent.name != "images":
                continue
            lbl = img.parent.parent / "labels" / (img.stem + ".txt")
            if lbl.exists():
                pairs.append((name, img, lbl))
                per_src[name] += 1
    print(f"image and label pairs found: {len(pairs)}")

    rng = random.Random(a.seed)
    rng.shuffle(pairs)
    n = len(pairs)
    n_tr, n_va = int(n * 0.70), int(n * 0.15)
    assign = {"train": pairs[:n_tr], "val": pairs[n_tr:n_tr + n_va],
              "test": pairs[n_tr + n_va:]}

    dropped = Counter()
    for split, items in assign.items():
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        kept = 0
        for name, img, lbl in items:
            boxes = [b for b in parse_label(lbl) if b[2] * b[3] >= a.min_box_area]
            if not boxes:
                dropped[name] += 1
                continue
            stem = f"{name}_{img.stem}"
            dst = out / "images" / split / (stem + img.suffix)
            if not dst.exists():
                dst.symlink_to(img.resolve())
            (out / "labels" / split / (stem + ".txt")).write_text(
                "\n".join("0 %.6f %.6f %.6f %.6f" % b for b in boxes) + "\n")
            kept += 1
        print(f"  {split:<6}{kept}")

    print("\nper source, images contributed:")
    for k, v in per_src.most_common():
        d = dropped[k]
        print(f"  {k:<18}{v:>6}" + (f"   ({d} dropped for having no valid box)" if d else ""))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
