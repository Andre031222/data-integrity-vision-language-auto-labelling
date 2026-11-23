#!/usr/bin/env python3
"""Characterise the byte-identical copies whose annotation files disagree."""
import argparse, hashlib, json, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dataset_v2 import EXCLUDED_SOURCES, collect
from analyse_contamination import iou

COLOURS = ["#e41a1c", "#377eb8", "#4daf4a", "#ff7f00", "#984ea3"]


def boxes(label):
    out = []
    if label.exists():
        for line in label.read_text().split("\n"):
            v = line.split()
            if len(v) == 5:
                cx, cy, w, h = map(float, v[1:])
                out.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
            elif len(v) > 5 and len(v) % 2 == 1:
                xy = np.array(v[1:], float).reshape(-1, 2)
                out.append([*xy.min(0), *xy.max(0)])
    return np.clip(np.array(out, float).reshape(-1, 4), 0, 1)


def compare(a, b):
    if len(a) == 0 or len(b) == 0:
        return {"matched": 0, "unmatched": int(len(a) + len(b)), "min_iou": 0.0, "mean_iou": 0.0}
    m = iou(a, b)
    r, c = linear_sum_assignment(-m)
    ious = m[r, c]
    good = ious >= 0.5
    return {"matched": int(good.sum()), "unmatched": int(len(a) + len(b) - 2 * good.sum()),
            "min_iou": round(float(ious.min()), 3), "mean_iou": round(float(ious.mean()), 3)}


def category(copies):
    counts = {len(b) for b in copies}
    worst = min((compare(copies[0], b) for b in copies[1:]), key=lambda x: (x["min_iou"], -x["unmatched"]))
    if len(counts) > 1:
        return "different number of boxes", worst
    if worst["min_iou"] >= 0.9:
        return "same boxes, coordinate jitter (IoU >= 0.9)", worst
    if worst["min_iou"] >= 0.5:
        return "same objects, looser or tighter boxes (0.5 <= IoU < 0.9)", worst
    return "boxes on different objects or regions (IoU < 0.5)", worst


def draw(group, copies, path, size=360):
    im = Image.open(group[0]["path"]).convert("RGB")
    im.thumbnail((size, size))
    w, h = im.size
    canvas = Image.new("RGB", (w * len(group), h + 18), "white")
    for k, (r, b) in enumerate(zip(group, copies)):
        panel = im.copy()
        d = ImageDraw.Draw(panel)
        for x0, y0, x1, y1 in b:
            d.rectangle([x0 * w, y0 * h, x1 * w, y1 * h], outline=COLOURS[k % len(COLOURS)], width=3)
        canvas.paste(panel, (k * w, 18))
        ImageDraw.Draw(canvas).text((k * w + 4, 2), f"{r['source']} ({len(b)})", fill="black")
    canvas.save(path, quality=90)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/annotated_v3")
    ap.add_argument("--released", default="data/v3.1/s6photoid")
    ap.add_argument("--out", default="outputs/figures/v31/box_conflicts.json")
    ap.add_argument("--sheets", default="outputs/figures/v31/box_conflicts")
    a = ap.parse_args()

    recs = collect(Path(a.src))
    order = {r["name"]: i for i, r in enumerate(recs)}
    copies = defaultdict(list)
    for r in recs:
        copies[hashlib.sha256(r["path"].read_bytes()).hexdigest()].append(r)
    released = {p.name for p in Path(a.released).glob("images/*/*")}

    sheets = Path(a.sheets)
    sheets.mkdir(parents=True, exist_ok=True)
    rows = []
    for digest, group in copies.items():
        texts = {r["label"].read_text().strip() if r["label"].exists() else "" for r in group}
        if len(texts) < 2:
            continue
        group.sort(key=lambda r: (r["source"] in EXCLUDED_SOURCES, order[r["name"]]))
        bx = [boxes(r["label"]) for r in group]
        cat, worst = category(bx)
        row = {"kept": group[0]["name"], "kept_source": group[0]["source"],
               "copies": len(group), "sources": sorted({r["source"] for r in group}),
               "boxes_per_copy": [len(b) for b in bx], "category": cat, **worst,
               "kept_copy_in_release": group[0]["name"] in released}
        rows.append(row)
        if cat != "same boxes, coordinate jitter (IoU >= 0.9)":
            draw(group, bx, sheets / f"{len(rows):02d}_{digest[:10]}.jpg")

    cats = Counter(r["category"] for r in rows)
    summary = {"groups": len(rows), "by_category": dict(cats),
               "kept_copy_in_release": sum(r["kept_copy_in_release"] for r in rows),
               "kept_copy_in_release_by_category": dict(Counter(r["category"] for r in rows
                                                                 if r["kept_copy_in_release"])),
               "sheets": str(sheets), "groups_detail": rows}
    Path(a.out).write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "groups_detail"}, indent=2))


if __name__ == "__main__":
    main()
