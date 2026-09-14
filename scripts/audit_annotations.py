#!/usr/bin/env python3
"""Audit bounding-box geometry and render a contact sheet for human review."""
import argparse, json, random
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_labels(path):
    rows = []
    if not path.exists():
        return rows
    for ln, line in enumerate(path.read_text().splitlines(), 1):
        line = line.split("#")[0].strip()
        if not line:
            continue
        parts = line.split()
        try:
            rows.append((ln, int(float(parts[0])), *map(float, parts[1:5])))
        except (ValueError, IndexError):
            rows.append((ln, None, None, None, None, None))
    return rows


def audit(root, splits, nc):
    issues = Counter()
    examples = {}
    stats = {"boxes": 0, "images": 0, "empty": 0, "areas": [], "aspects": [], "per_image": []}

    def flag(key, ref):
        issues[key] += 1
        examples.setdefault(key, []).append(ref)

    for sp in splits:
        img_dir, lbl_dir = root / "images" / sp, root / "labels" / sp
        if not img_dir.exists():
            continue
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in EXT:
                continue
            stats["images"] += 1
            rows = read_labels(lbl_dir / (img.stem + ".txt"))
            if not (lbl_dir / (img.stem + ".txt")).exists():
                flag("label file missing", f"{sp}/{img.name}")
                continue
            if not rows:
                stats["empty"] += 1
                flag("no boxes (background image)", f"{sp}/{img.name}")
                continue
            stats["per_image"].append(len(rows))
            for ln, cid, cx, cy, w, h in rows:
                stats["boxes"] += 1
                ref = f"{sp}/{img.name}:{ln}"
                if cid is None:
                    flag("malformed line", ref); continue
                if not 0 <= cid < nc:
                    flag(f"class id out of range (nc={nc})", ref)
                if not all(0.0 <= v <= 1.0 for v in (cx, cy, w, h)):
                    flag("coordinate outside [0,1]", ref)
                if w <= 0 or h <= 0:
                    flag("degenerate box (zero side)", ref); continue
                if cx - w / 2 < -1e-6 or cx + w / 2 > 1 + 1e-6 \
                   or cy - h / 2 < -1e-6 or cy + h / 2 > 1 + 1e-6:
                    flag("box extends past the image edge", ref)
                area = w * h
                stats["areas"].append(area)
                stats["aspects"].append(w / h)
                if area >= 0.95:
                    flag("box covers >=95% of the image", ref)
                if area <= 1e-3:
                    flag("box covers <=0.1% of the image", ref)
                if w / h >= 10 or h / w >= 10:
                    flag("aspect ratio beyond 10:1", ref)
    return issues, examples, stats


def contact_sheet(root, split, out, n, cols, tile, seed):
    img_dir, lbl_dir = root / "images" / split, root / "labels" / split
    names = sorted(p.name for p in img_dir.iterdir() if p.suffix.lower() in EXT)
    random.Random(seed).shuffle(names)
    names = names[:n]
    rows = (len(names) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tile, rows * tile), "white")
    draw_sheet = ImageDraw.Draw(sheet)
    for i, name in enumerate(names):
        im = Image.open(img_dir / name).convert("RGB")
        W, H = im.size
        d = ImageDraw.Draw(im)
        for _, cid, cx, cy, w, h in read_labels(lbl_dir / (Path(name).stem + ".txt")):
            if cid is None or w is None:
                continue
            d.rectangle([(cx - w / 2) * W, (cy - h / 2) * H,
                         (cx + w / 2) * W, (cy + h / 2) * H],
                        outline=(217, 83, 25), width=max(2, int(min(W, H) * 0.008)))
        im.thumbnail((tile - 6, tile - 6))
        x, y = (i % cols) * tile, (i // cols) * tile
        sheet.paste(im, (x + (tile - im.width) // 2, y + (tile - im.height) // 2))
        draw_sheet.text((x + 4, y + 4), str(i + 1), fill=(0, 0, 0))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=92)
    return names, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/annotated_v3_final")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--nc", type=int, default=1)
    ap.add_argument("--sample", type=int, default=60)
    ap.add_argument("--sample-split", default="test")
    ap.add_argument("--cols", type=int, default=10)
    ap.add_argument("--tile", type=int, default=240)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="outputs/figures/annotation_audit.json")
    ap.add_argument("--sheet", default="outputs/figures/annotation_contact_sheet.jpg")
    a = ap.parse_args()

    root = Path(a.root)
    issues, examples, st = audit(root, a.splits, a.nc)

    n = len(st["areas"])
    med = lambda v: sorted(v)[len(v) // 2] if v else 0.0
    print(f"root            : {a.root}")
    print(f"images          : {st['images']}")
    print(f"boxes           : {st['boxes']}")
    print(f"images w/o boxes: {st['empty']}")
    if n:
        print(f"boxes per image : median {med(st['per_image'])}, max {max(st['per_image'])}")
        print(f"box area        : median {med(st['areas']):.4f}, "
              f"min {min(st['areas']):.5f}, max {max(st['areas']):.4f}")
        print(f"aspect ratio    : median {med(st['aspects']):.2f}, "
              f"min {min(st['aspects']):.2f}, max {max(st['aspects']):.2f}")
    print("\ngeometry and consistency issues")
    if not issues:
        print("  none")
    for k, v in issues.most_common():
        print(f"  {v:>5}  {k}")
        for ref in examples[k][:3]:
            print(f"           e.g. {ref}")

    names, sheet = contact_sheet(root, a.sample_split, Path(a.sheet),
                                 a.sample, a.cols, a.tile, a.seed)
    print(f"\ncontact sheet   : {sheet}  ({len(names)} images from {a.sample_split})")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({
        "root": a.root, "splits": a.splits,
        "images": st["images"], "boxes": st["boxes"], "images_without_boxes": st["empty"],
        "box_area_median": round(med(st["areas"]), 5) if n else None,
        "box_area_min": round(min(st["areas"]), 5) if n else None,
        "box_area_max": round(max(st["areas"]), 5) if n else None,
        "boxes_per_image_median": med(st["per_image"]) if n else None,
        "issues": dict(issues),
        "issue_examples": {k: v[:5] for k, v in examples.items()},
        "contact_sheet": str(sheet),
        "contact_sheet_images": names,
    }, indent=2))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
