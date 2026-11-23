#!/usr/bin/env python3
"""Audit photograph-level independence of the ocular train, validation and test partitions."""
import argparse, json, re, sys
from collections import Counter, defaultdict
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.data.group_split import (build_group_split, build_photo_split, original_stem,
                                  source_file, source_photo)
from build_dataset_v2 import TRANSFORMS, _bits
from analyse_contamination import cross_distance

CROPS = Path("data/crops/eyes")
EXT = {".jpg", ".jpeg", ".png", ".bmp"}
INAT_PHOTO = re.compile(r"inat_(\d+)_(\d+)")


def source_observation(name):
    m = INAT_PHOTO.match(source_file(name))
    return f"inatobs:{m.group(1)}" if m else source_photo(name)


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def link(keys, uf):
    first = {}
    for i, k in enumerate(keys):
        if k in first:
            uf.union(i, first[k])
        else:
            first[k] = i


def crop_bits(paths):
    out = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        out.append(np.stack([_bits(int(str(imagehash.phash(fn(im))), 16)) for fn in TRANSFORMS]))
    return np.stack(out)


def grouping(files, paths, tau):
    levels = {"crop": [original_stem(f) for f in files],
              "source_file": [source_file(f) for f in files],
              "source_photo": [source_photo(f) for f in files]}
    labels = {}
    uf = UnionFind(len(files))
    for name in ("crop", "source_file", "source_photo"):
        link(levels[name], uf)
        labels[name] = np.array([uf.find(i) for i in range(len(files))])
    originals = [i for i, f in enumerate(files) if not f.startswith("aug_")]
    d = cross_distance(*(2 * [crop_bits([paths[i] for i in originals])]))
    for a, b in zip(*np.nonzero(np.triu(d <= tau, 1))):
        uf.union(originals[a], originals[b])
    labels["photo_plus_phash"] = np.array([uf.find(i) for i in range(len(files))])
    link([source_observation(f) for f in files], uf)
    labels["inat_observation"] = np.array([uf.find(i) for i in range(len(files))])
    return labels


def audit_split(split, index, labels, ylab):
    part = {}
    for k in ("train", "val", "test"):
        part[k] = np.array([index[Path(p).name] for p, _ in split[k]])
    res = {"files": {k: int(len(v)) for k, v in part.items()}}
    for level, lab in labels.items():
        g = {k: set(lab[v]) for k, v in part.items()}
        test_overlap = [i for i in part["test"] if lab[i] in g["train"] or lab[i] in g["val"]]
        conflicts = sum(1 for i in part["test"]
                        if any(lab[j] == lab[i] and ylab[j] != ylab[i] for j in part["train"]))
        res[level] = {"groups": {k: len(v) for k, v in g.items()},
                      "shared_groups": {"train_val": len(g["train"] & g["val"]),
                                        "train_test": len(g["train"] & g["test"]),
                                        "val_test": len(g["val"] & g["test"])},
                      "test_files_with_group_in_train_or_val": len(test_overlap),
                      "test_anomaly_files_affected": int(sum(ylab[i] for i in test_overlap)),
                      "test_files_with_opposite_label_in_train": conflicts}
    return res, [Path(p).name for p, _ in split["test"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=int, default=6)
    ap.add_argument("--runs", nargs="+", default=["outputs/figures/v31/classifier_fixed_split.json",
                                                  "outputs/figures/v31/classifier_split_sweep.json",
                                                  "outputs/figures/v31/classifier_fixed_split_photo.json",
                                                  "outputs/figures/v31/classifier_split_sweep_photo.json"])
    ap.add_argument("--out", default="outputs/figures/v31/ocular/grouping_audit.json")
    a = ap.parse_args()

    paths = sorted(p for c in ("anomaly", "normal") for p in (CROPS / c).iterdir()
                   if p.suffix.lower() in EXT)
    files = [p.name for p in paths]
    ylab = np.array([p.parent.name == "anomaly" for p in paths], int)
    index = {}
    for i, p in enumerate(paths):
        index.setdefault(p.name, i)
    dup_names = [n for n, c in Counter(files).items() if c > 1]
    labels = grouping(files, paths, a.tau)
    orig = np.array([not f.startswith("aug_") for f in files])

    summary = {"files": len(files), "original_crops": int(orig.sum()),
               "same_name_in_both_classes": len(dup_names),
               "distinct_groups_over_originals": {k: int(len(set(v[orig]))) for k, v in labels.items()},
               "groups_with_both_labels": {k: int(sum(len(set(ylab[orig][v[orig] == g])) > 1
                                                      for g in set(v[orig])))
                                           for k, v in labels.items()},
               "runs": {}}
    for run_file in a.runs:
        if not Path(run_file).exists():
            continue
        runs = json.load(open(run_file))["per_seed"]
        design = Path(run_file).stem.replace("classifier_", "")
        out = []
        for r in runs:
            builder = build_photo_split if r.get("grouping") == "photo" else build_group_split
            split = builder(CROPS, seed=r["split_seed"], test_frac=0.15, val_frac=0.15,
                            originals_only_eval=True)
            res, test_files = audit_split(split, index, labels, ylab)
            res["seed"], res["split_seed"] = r["seed"], r["split_seed"]
            res["test_files_match_recorded_run"] = sorted(test_files) == sorted(r["test_files"])
            out.append(res)
        summary["runs"][design] = out
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2))
    brief = {k: v for k, v in summary.items() if k != "runs"}
    print(json.dumps(brief, indent=2))
    for design, out in summary["runs"].items():
        for res in out:
            print(design, res["split_seed"], res["test_files_match_recorded_run"],
                  {lvl: (res[lvl]["groups"]["test"], res[lvl]["test_files_with_group_in_train_or_val"],
                         res[lvl]["test_files_with_opposite_label_in_train"])
                   for lvl in labels})


if __name__ == "__main__":
    main()
