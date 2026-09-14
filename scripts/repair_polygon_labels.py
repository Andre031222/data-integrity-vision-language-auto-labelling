#!/usr/bin/env python3
"""Convert polygon-format YOLO labels to axis-aligned boxes.

Two Roboflow sources export segmentation polygons rather than boxes. Reading only
the first four values of such a line as (cx, cy, w, h) yields a meaningless box,
which is what the original consolidation did. This recovers the enclosing box.
"""
import argparse, json
from collections import Counter
from pathlib import Path


def parse_line(line):
    """Return (class_id, cx, cy, w, h) or None if the line cannot be read."""
    line = line.split("#")[0].strip()
    if not line:
        return None
    parts = line.split()
    try:
        cid = int(float(parts[0]))
        vals = [float(v) for v in parts[1:]]
    except (ValueError, IndexError):
        return None
    if len(vals) == 4:
        return (cid, *vals)
    if len(vals) >= 6 and len(vals) % 2 == 0:
        xs, ys = vals[0::2], vals[1::2]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        return (cid, (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0)
    return None


def clamp(cid, cx, cy, w, h):
    """Clip the box to the image and drop it if nothing meaningful is left."""
    x0, x1 = max(0.0, cx - w / 2), min(1.0, cx + w / 2)
    y0, y1 = max(0.0, cy - h / 2), min(1.0, cy + h / 2)
    w, h = x1 - x0, y1 - y0
    if w <= 1e-4 or h <= 1e-4:
        return None
    return (cid, (x0 + x1) / 2, (y0 + y1) / 2, w, h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/raw/roboflow")
    ap.add_argument("--out", default="data/raw/roboflow_repaired")
    ap.add_argument("--report", default="outputs/figures/polygon_repair.json")
    a = ap.parse_args()

    src, out = Path(a.src), Path(a.out)
    stats = {}
    for proj in sorted(p for p in src.iterdir() if p.is_dir()):
        c = Counter()
        for sp in ("train", "valid", "test"):
            ld, idir = proj / sp / "labels", proj / sp / "images"
            if not ld.exists():
                continue
            od = out / proj.name / sp / "labels"
            oi = out / proj.name / sp / "images"
            od.mkdir(parents=True, exist_ok=True)
            oi.mkdir(parents=True, exist_ok=True)
            for f in sorted(ld.iterdir()):
                if f.suffix != ".txt":
                    continue
                lines = []
                for line in f.read_text().splitlines():
                    p = parse_line(line)
                    if p is None:
                        if line.strip():
                            c["unparsable"] += 1
                        continue
                    n_vals = len(line.split()) - 1
                    c["polygon" if n_vals > 4 else "box"] += 1
                    q = clamp(*p)
                    if q is None:
                        c["dropped after clipping"] += 1
                        continue
                    if n_vals > 4:
                        c["recovered"] += 1
                    lines.append("%d %.6f %.6f %.6f %.6f" % q)
                (od / f.name).write_text("\n".join(lines) + ("\n" if lines else ""))
                for ext in (".jpg", ".jpeg", ".png"):
                    img = idir / (f.stem + ext)
                    if img.exists():
                        dst = oi / img.name
                        if not dst.exists():
                            dst.symlink_to(img.resolve())
                        break
        stats[proj.name] = dict(c)
        tag = "  POLYGON SOURCE" if c.get("polygon") else ""
        print(f"{proj.name:<16} box={c.get('box',0):>5} polygon={c.get('polygon',0):>5} "
              f"recovered={c.get('recovered',0):>5} dropped={c.get('dropped after clipping',0):>3}{tag}")

    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    Path(a.report).write_text(json.dumps(stats, indent=2))
    print(f"\nwrote {out} and {a.report}")


if __name__ == "__main__":
    main()
