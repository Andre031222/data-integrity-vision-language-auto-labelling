#!/usr/bin/env python3
"""Build the licence-clean, dihedral-invariant deduplicated detector dataset (v2)."""
import argparse, hashlib, json, random, re, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
import imagehash

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EXCLUDED_SOURCES = {
    "alpaca-xqfiw": "licence undefined",
    "alpaca-lls3s": "annotates heads, not whole animals",
}
EXCLUDED_CONTENT = {
    "01ad3ff1d94eb557": "sheep",
    "98f778a1e4596438": "sheep",
    "e0cf0c3894b469d3": "sheep",
    "0a46037beed719ca": "camel",
    "77e1b88e936be7c3": "camel",
    "f8b69bc9cb1d75ef": "camel",
    "0a5e8d559aa5d680": "camel",
    "5322b9e8e350fbfa": "camel",
    "e7d680d2edc500a2": "camel",
    "62bead59ad0d0dff": "vehicle decal",
    "c3e8c57b4cc8d3f5": "illustration",
    "e530c1b79b651ac9": "drawing",
    "f8064d7f797ee894": "sculpture",
    "alpaca-nrzos_images-11-_jpeg": "drawing",
}
PHOTO_ID = re.compile(r"alpaca-[a-z0-9]+_([0-9a-f]{16,32})_(?:jpg|jpeg|png)\.rf\.")


def source_of(name):
    m = re.match(r"(alpaca-[a-z0-9]+)", name)
    return m.group(1) if m else "unknown"


def photo_id(name):
    m = PHOTO_ID.match(name)
    return m.group(1) if m else None


def merge_by_photo_id(groups, recs):
    owner = {}
    parent = list(range(len(groups)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for g, members in enumerate(groups):
        for i in members:
            pid = recs[i]["photo"]
            if pid is None:
                continue
            if pid in owner:
                ra, rb = find(owner[pid]), find(g)
                if ra != rb:
                    parent[ra] = rb
            else:
                owner[pid] = g
    merged = defaultdict(list)
    for g, members in enumerate(groups):
        merged[find(g)].extend(members)
    return list(merged.values())


def content_key(rec):
    for key in EXCLUDED_CONTENT:
        if rec["photo"] == key or rec["name"].startswith(key):
            return key
    return None


def content_exclusions(recs, tau):
    groups = merge_by_photo_id(group_by_distance(dihedral_distance(phash_variants(recs)), tau),
                               recs)
    excluded = {}
    for members in groups:
        keys = {content_key(recs[i]) for i in members} - {None}
        if keys:
            for i in members:
                excluded[i] = sorted(keys)[0]
    return excluded


def collect(src_root):
    recs = []
    for split in ("train", "val", "test"):
        for f in sorted((src_root / "images" / split).iterdir()):
            if f.suffix.lower() not in EXT:
                continue
            lbl = src_root / "labels" / split / (f.stem + ".txt")
            recs.append({"path": f, "label": lbl, "name": f.name,
                         "source": source_of(f.name), "photo": photo_id(f.name)})
    return recs


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
    out = np.zeros((len(recs), len(TRANSFORMS), 64), np.uint8)
    for i, r in enumerate(recs):
        im = Image.open(r["path"]).convert("RGB")
        for t, fn in enumerate(TRANSFORMS):
            out[i, t] = _bits(int(str(imagehash.phash(fn(im))), 16))
    return out


def dihedral_distance(variants):
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
    ap.add_argument("--src", default="data/annotated_v3")
    ap.add_argument("--out", default="data/annotated_v3_final")
    ap.add_argument("--tau", type=int, default=6)
    ap.add_argument("--hash", choices=("none", "phash", "dihedral"), default="dihedral")
    ap.add_argument("--keep-all-sources", action="store_true")
    ap.add_argument("--no-md5", dest="md5", action="store_false")
    ap.add_argument("--source-id", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    src_root, out_root = Path(a.src), Path(a.out)
    recs = collect(src_root)
    print(f"images in source split      : {len(recs)}")

    if a.md5:
        order = {r["name"]: i for i, r in enumerate(recs)}
        copies = defaultdict(list)
        for r in recs:
            copies[hashlib.sha256(r["path"].read_bytes()).hexdigest()].append(r)
        unique, label_conflicts = [], 0
        for group in copies.values():
            group.sort(key=lambda r: (r["source"] in EXCLUDED_SOURCES, order[r["name"]]))
            unique.append(group[0])
            texts = {r["label"].read_text().strip() if r["label"].exists() else ""
                     for r in group}
            label_conflicts += len(texts) > 1
        unique.sort(key=lambda r: order[r["name"]])
        print(f"after cryptographic dedup   : {len(unique)}  "
              f"(removed {len(recs) - len(unique)} byte-identical files; "
              f"{label_conflicts} groups whose copies carry different boxes)")
        recs = unique
    else:
        print("cryptographic dedup         : skipped (ablation)")

    if a.keep_all_sources:
        kept = list(recs)
        print("source filters              : skipped (ablation)")
    else:
        kept = [r for r in recs if r["source"] not in EXCLUDED_SOURCES]
        print(f"after licence filter        : {len(kept)}  "
              f"(removed {len(recs) - len(kept)} from {sorted(EXCLUDED_SOURCES)})")
    if not a.keep_all_sources:
        excluded = content_exclusions(kept, a.tau)
        reasons = Counter(EXCLUDED_CONTENT[k] for k in excluded.values())
        kept = [r for i, r in enumerate(kept) if i not in excluded]
        print(f"after content review        : {len(kept)}  "
              f"(removed {len(excluded)} files: {dict(reasons)})")

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
    if a.source_id:
        groups = merge_by_photo_id(groups, kept)
        print(f"after photo-id grouping     : {len(groups)}")

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

    ii, jj = np.where(np.triu(dist <= a.tau, 1))
    crossing = sum(1 for x, y in zip(ii, jj)
                   if member_split[int(x)] != member_split[int(y)])
    print(f"near-duplicate pairs crossing splits: {crossing}  (must be 0)")
    assert crossing == 0, "grouping failed: near-duplicates leaked across splits"
    photo_splits = defaultdict(set)
    for i, sp in member_split.items():
        if kept[i]["photo"]:
            photo_splits[kept[i]["photo"]].add(sp)
    shared = sum(1 for v in photo_splits.values() if len(v) > 1)
    print(f"photo ids shared across splits      : {shared}")
    if a.source_id:
        assert shared == 0, "grouping failed: a source photo id leaked across splits"

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
    json.dump({"tau": a.tau, "hash": a.hash, "exact_dedup": "sha256" if a.md5 else None,
               "keep_rule": "redistributable source first, then traversal order (split, file name)",
               "source_id_grouping": a.source_id, "photo_ids_shared_across_splits": shared,
               "seed": a.seed,
               "excluded_sources": {} if a.keep_all_sources else EXCLUDED_SOURCES,
               "excluded_content": {} if a.keep_all_sources else EXCLUDED_CONTENT,
               "images": sum(counts.values()), "groups": len(order),
               "per_split_groups": {s: len(g) for s, g in assign.items()},
               "per_split_images": counts,
               "provenance": dict(Counter(r["source"] for r in kept))},
              open(out_root / "build_manifest.json", "w"), indent=2)
    print(f"\nwrote {out_root}/data.yaml and build_manifest.json")


if __name__ == "__main__":
    sys.exit(main())
