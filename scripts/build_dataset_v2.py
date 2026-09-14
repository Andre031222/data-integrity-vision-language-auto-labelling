#!/usr/bin/env python3
"""Build the licence-clean, dihedral-invariant deduplicated detector dataset (v2)."""
import argparse, hashlib, json, random, re, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
import imagehash

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# Two sources are excluded, for unrelated reasons.
#   alpaca-xqfiw declares "License: undefined" on Roboflow Universe, so it cannot be
#     redistributed under CC BY 4.0.
#   alpaca-lls3s annotates heads rather than whole animals: its median box area is
#     0.071 against 0.19-0.22 for every other source, so it labels a different target
#     under the same single class.
EXCLUDED_SOURCES = {"alpaca-xqfiw", "alpaca-lls3s"}

# Human review of every distinct scene of alpaca-zehtv found one depicting sheep
# rather than alpacas. It is excluded by source-image id, which covers all three
# of its geometrically augmented copies.
EXCLUDED_SCENES = {"alpaca-zehtv_01ad3ff1d94eb557"}


def source_of(name):
    m = re.match(r"(alpaca-[a-z0-9]+)", name)
    return m.group(1) if m else "unknown"


def collect(src_root):
    recs = []
    for split in ("train", "val", "test"):
        for f in sorted((src_root / "images" / split).iterdir()):
            if f.suffix.lower() not in EXT:
                continue
            lbl = src_root / "labels" / split / (f.stem + ".txt")
            recs.append({"path": f, "label": lbl, "name": f.name,
                         "source": source_of(f.name)})
    return recs


# Three source projects ship 3 augmented copies (flips, 90-degree rotations) of
# every image, which a plain perceptual hash does not match. Hashing all eight
# dihedral transforms makes the comparison invariant to them.
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


def _bits(v):
    return np.array([(v >> k) & 1 for k in range(64)], np.uint8)


def phash_variants(recs):
    """Return (n, 8, 64) bits: one perceptual hash per dihedral transform."""
    out = np.zeros((len(recs), len(TRANSFORMS), 64), np.uint8)
    for i, r in enumerate(recs):
        im = Image.open(r["path"]).convert("RGB")
        for t, fn in enumerate(TRANSFORMS):
            out[i, t] = _bits(int(str(imagehash.phash(fn(im))), 16))
    return out


def dihedral_distance(variants):
    """Pairwise Hamming distance minimised over the eight dihedral transforms."""
    n = variants.shape[0]
    base = variants[:, 0, :].astype(np.int16)
    ob = base.sum(1)
    d = np.full((n, n), 127, np.int16)
    for t in range(variants.shape[1]):
        y = variants[:, t, :].astype(np.int16)
        dt = y.sum(1)[:, None] + ob[None, :] - 2 * (y @ base.T)
        d = np.minimum(d, dt)
    return np.minimum(d, d.T)


def group_by_distance(d, tau):
    """Union-find over pairs within Hamming distance tau."""
    n = d.shape[0]
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    ii, jj = np.where(np.triu(d <= tau, 1))
    for a, b_ in zip(ii, jj):
        ra, rb = find(int(a)), find(int(b_))
        if ra != rb:
            parent[ra] = rb
    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return list(groups.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/annotated_clean")
    ap.add_argument("--out", default="data/annotated_v2")
    ap.add_argument("--tau", type=int, default=6)
    # Ablation: "none" groups nothing, "phash" ignores rotations and flips.
    ap.add_argument("--hash", choices=("none", "phash", "dihedral"), default="dihedral")
    ap.add_argument("--keep-all-sources", action="store_true",
                    help="skip the source filters (ablation only, not for release)")
    ap.add_argument("--no-md5", dest="md5", action="store_false",
                    help="skip exact deduplication (ablation only)")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    src_root, out_root = Path(a.src), Path(a.out)
    recs = collect(src_root)
    print(f"images in source split      : {len(recs)}")

    if a.md5:
        seen, unique = {}, []
        for r in recs:
            h = hashlib.md5(r["path"].read_bytes()).hexdigest()
            if h not in seen:
                seen[h] = r["name"]
                unique.append(r)
        print(f"after cryptographic dedup   : {len(unique)}  "
              f"(removed {len(recs) - len(unique)} byte-identical files)")
        recs = unique
    else:
        print("cryptographic dedup         : skipped (ablation)")

    if a.keep_all_sources:
        kept = list(recs)
        print("licence filter              : skipped (ablation)")
    else:
        kept = [r for r in recs if r["source"] not in EXCLUDED_SOURCES]
        print(f"after licence filter        : {len(kept)}  "
              f"(removed {len(recs) - len(kept)} from {sorted(EXCLUDED_SOURCES)})")
    before = len(kept)
    kept = [r for r in kept
            if not any(r["name"].startswith(x) for x in EXCLUDED_SCENES)]
    if before != len(kept):
        print(f"after mislabelled-scene drop: {len(kept)}  "
              f"(removed {before - len(kept)}: depicts sheep, not alpacas)")

    if a.hash == "none":
        dist = np.full((len(kept), len(kept)), 127, np.int16)
        np.fill_diagonal(dist, 0)
        groups = [[i] for i in range(len(kept))]
        print(f"grouping                    : none, {len(groups)} singleton groups")
    else:
        variants = phash_variants(kept)
        if a.hash == "phash":
            variants = variants[:, :1, :]
        dist = dihedral_distance(variants)
        groups = group_by_distance(dist, a.tau)
        print(f"unique scene groups         : {len(groups)}  "
              f"({a.hash} pHash, tau={a.tau})")

    rng = random.Random(a.seed)
    order = sorted(groups, key=lambda g: kept[min(g)]["name"])
    rng.shuffle(order)
    n = len(order)
    n_tr, n_va = int(0.70 * n), int(0.15 * n)
    assign = {"train": order[:n_tr], "val": order[n_tr:n_tr + n_va], "test": order[n_tr + n_va:]}

    for split, gs in assign.items():
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)
        for g in gs:
            for i in g:
                r = kept[i]
                dst_i = out_root / "images" / split / r["name"]
                dst_l = out_root / "labels" / split / (r["path"].stem + ".txt")
                for dst, s in ((dst_i, r["path"]), (dst_l, r["label"])):
                    if dst.is_symlink() or dst.exists():
                        dst.unlink()
                    if s.exists():
                        dst.symlink_to(s.resolve())

    member_split = {}
    for split, gs in assign.items():
        for g in gs:
            for i in g:
                member_split[i] = split

    # No near-duplicate pair may cross a split boundary.
    ii, jj = np.where(np.triu(dist <= a.tau, 1))
    crossing = sum(1 for x, y in zip(ii, jj)
                   if member_split[int(x)] != member_split[int(y)])
    print(f"near-duplicate pairs crossing splits: {crossing}  (must be 0)")
    assert crossing == 0, "grouping failed: near-duplicates leaked across splits"

    counts = {s: sum(len(g) for g in gs) for s, gs in assign.items()}
    print("\nsplit      groups  images")
    for s in ("train", "val", "test"):
        print(f"{s:<10} {len(assign[s]):>6}  {counts[s]:>6}")
    print(f"{'total':<10} {len(order):>6}  {sum(counts.values()):>6}")

    print("\nprovenance:")
    for k, v in Counter(r["source"] for r in kept).most_common():
        print(f"  {k:<18} {v}")

    (out_root / "data.yaml").write_text(
        f"path: {out_root.resolve()}\ntrain: images/train\nval: images/val\n"
        f"test: images/test\n\nnc: 1\nnames:\n  0: alpaca\n")
    json.dump({"tau": a.tau, "hash": a.hash, "md5": a.md5, "seed": a.seed,
               "excluded_sources": [] if a.keep_all_sources else sorted(EXCLUDED_SOURCES),
               "images": sum(counts.values()), "groups": len(order),
               "per_split_groups": {s: len(g) for s, g in assign.items()},
               "per_split_images": counts,
               "provenance": dict(Counter(r["source"] for r in kept))},
              open(out_root / "build_manifest.json", "w"), indent=2)
    print(f"\nwrote {out_root}/data.yaml and build_manifest.json")


if __name__ == "__main__":
    sys.exit(main())
