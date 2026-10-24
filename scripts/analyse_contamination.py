#!/usr/bin/env python3
"""Split each stage's test score by contamination and bootstrap it over scene groups."""
import argparse, json, shutil, sys
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dataset_v2 import TRANSFORMS, _bits, group_by_distance, photo_id

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CACHE = Path("outputs/cache")


def list_split(root, split):
    imgs = sorted(p for p in (Path(root) / "images" / split).iterdir() if p.suffix.lower() in EXT)
    out = []
    for p in imgs:
        lbl = Path(root) / "labels" / split / (p.stem + ".txt")
        boxes = []
        if lbl.exists():
            for line in lbl.read_text().split("\n"):
                v = line.split()
                if len(v) == 5:
                    cx, cy, w, h = map(float, v[1:])
                    boxes.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
        out.append({"name": p.name, "path": p, "gt": np.clip(np.array(boxes, float).reshape(-1, 4), 0, 1)})
    return out


def dihedral_bits(records):
    CACHE.mkdir(parents=True, exist_ok=True)
    store = CACHE / "dihedral_bits.npz"
    known = dict(np.load(store)) if store.exists() else {}
    fresh = False
    for r in records:
        if r["name"] not in known:
            im = Image.open(r["path"]).convert("RGB")
            known[r["name"]] = np.stack([_bits(int(str(imagehash.phash(fn(im))), 16))
                                         for fn in TRANSFORMS])
            fresh = True
    if fresh:
        np.savez_compressed(store, **known)
    return np.stack([known[r["name"]] for r in records])


def cross_distance(a, b):
    base = b[:, 0, :].astype(np.int16)
    ob = base.sum(1)
    d = np.full((len(a), len(b)), 127, np.int16)
    for t in range(a.shape[1]):
        y = a[:, t, :].astype(np.int16)
        d = np.minimum(d, y.sum(1)[:, None] + ob[None, :] - 2 * (y @ base.T))
    base_a = a[:, 0, :].astype(np.int16)
    oa = base_a.sum(1)
    for t in range(b.shape[1]):
        y = b[:, t, :].astype(np.int16)
        d = np.minimum(d, oa[:, None] + y.sum(1)[None, :] - 2 * (base_a @ y.T))
    return d


def twins(a_bits, b_bits, a_recs, b_recs, tau, use_photo):
    t = cross_distance(a_bits, b_bits) <= tau
    if use_photo:
        pa = np.array([photo_id(r["name"]) or f"-{i}" for i, r in enumerate(a_recs)])
        pb = np.array([photo_id(r["name"]) or f"+{i}" for i, r in enumerate(b_recs)])
        t |= pa[:, None] == pb[None, :]
    return t


def scene_groups(bits, tau, recs=None):
    d = cross_distance(bits, bits)
    if recs is not None:
        d = np.where(twins(bits, bits, recs, recs, tau, True), 0, d)
    label = np.empty(len(bits), int)
    for k, g in enumerate(group_by_distance(d, tau)):
        label[g] = k
    return label


def predict(weights, root, split, records, imgsz, device):
    from ultralytics import YOLO
    CACHE.mkdir(parents=True, exist_ok=True)
    run = Path(weights).parts[-3]
    key = CACHE / f"pred_{run}_{Path(root).name}_{split}.npz"
    if not key.exists():
        tmp = CACHE / "val_tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        YOLO(str(weights)).val(data=str(Path(root) / "data.yaml"), split=split, imgsz=imgsz,
                               batch=8, conf=0.001, iou=0.7, save_txt=True, save_conf=True,
                               device=device, project=str(tmp.resolve()), name="run",
                               exist_ok=True, plots=False, verbose=False)
        preds = {}
        for f in (tmp / "run" / "labels").glob("*.txt"):
            v = np.loadtxt(f, ndmin=2)
            cx, cy, w, h, c = v[:, 1], v[:, 2], v[:, 3], v[:, 4], v[:, 5]
            preds[f.stem] = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, c], 1)
        shutil.rmtree(tmp, ignore_errors=True)
        np.savez_compressed(key, names=np.array(list(preds)),
                            preds=np.array(list(preds.values()), dtype=object))
    z = np.load(key, allow_pickle=True)
    cached = dict(zip(z["names"], z["preds"]))
    return [cached.get(Path(r["name"]).stem, np.zeros((0, 5))) for r in records]


def iou(a, b):
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def match(pred, gt, thr=0.5):
    tp = np.zeros(len(pred), bool)
    if len(pred) and len(gt):
        m = iou(gt, pred[:, :4])
        gi, pi = np.nonzero(m >= thr)
        if len(gi):
            order = np.argsort(-m[gi, pi], kind="stable")
            gi, pi = gi[order], pi[order]
            _, keep = np.unique(pi, return_index=True)
            gi, pi = gi[keep], pi[keep]
            _, keep = np.unique(gi, return_index=True)
            tp[pi[keep]] = True
    return tp


def average_precision(conf, tp, n_gt):
    if n_gt == 0:
        return float("nan")
    order = np.argsort(-conf, kind="stable")
    tpc = np.cumsum(tp[order])
    fpc = np.cumsum(~tp[order])
    recall = tpc / n_gt
    precision = tpc / np.maximum(tpc + fpc, 1e-9)
    mrec = np.concatenate(([0.0], recall, [recall[-1] if len(recall) else 1.0], [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0], [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    return float(np.trapezoid(np.interp(x, mrec, mpre), x))


class Scored:
    def __init__(self, records, preds):
        self.conf, self.tp, self.img, self.ngt = [], [], [], np.zeros(len(records), int)
        for i, (r, p) in enumerate(zip(records, preds)):
            self.conf.append(p[:, 4])
            self.tp.append(match(p, r["gt"]))
            self.img.append(np.full(len(p), i))
            self.ngt[i] = len(r["gt"])
        self.conf, self.tp, self.img = map(np.concatenate, (self.conf, self.tp, self.img))

    def ap(self, weights_per_image):
        w = weights_per_image[self.img]
        sel = w > 0
        conf = np.repeat(self.conf[sel], w[sel])
        tp = np.repeat(self.tp[sel], w[sel])
        return average_precision(conf, tp, int((self.ngt * weights_per_image).sum()))


def bootstrap(scored, groups, mask, reps, rng):
    idx = np.nonzero(mask)[0]
    uniq = np.unique(groups[idx])
    members = {g: idx[groups[idx] == g] for g in uniq}
    point = float(np.mean([s.ap(mask.astype(int)) for s in scored]))
    draws = []
    for _ in range(reps):
        w = np.zeros(len(mask), int)
        for g in rng.choice(uniq, len(uniq), replace=True):
            w[members[g]] += 1
        draws.append(np.mean([s.ap(w) for s in scored]))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"map50": round(point, 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "images": int(mask.sum()), "scene_groups": int(len(uniq))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="+", required=True)
    ap.add_argument("--runs", default="outputs/training_runs_v31")
    ap.add_argument("--reference", default="s5dihedral")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--tau", type=int, default=6)
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-photo-id", dest="photo_id", action="store_false")
    ap.add_argument("--out", default="outputs/figures/v31/contamination_analysis.json")
    a = ap.parse_args()

    stages = dict(s.split("=", 1) for s in a.stages)
    rng = np.random.default_rng(0)
    ref_test = list_split(stages[a.reference], "test")
    ref_bits = dihedral_bits(ref_test)
    ref_groups = scene_groups(ref_bits, a.tau, ref_test if a.photo_id else None)
    result = {"tau": a.tau, "photo_id": a.photo_id, "reps": a.reps, "reference": a.reference, "stages": {}}

    for tag, root in stages.items():
        train, test = list_split(root, "train"), list_split(root, "test")
        tr_bits, te_bits = dihedral_bits(train), dihedral_bits(test)
        contaminated = twins(te_bits, tr_bits, test, train, a.tau, a.photo_id).any(1)
        groups = scene_groups(te_bits, a.tau, test if a.photo_id else None)
        weights = [Path(a.runs) / f"{tag}_seed{s}" / "weights" / "best.pt" for s in a.seeds]
        scored = [Scored(test, predict(w, root, "test", test, a.imgsz, a.device)) for w in weights]
        ones = np.ones(len(test), bool)
        entry = {"own_test": bootstrap(scored, groups, ones, a.reps, rng),
                 "contaminated": bootstrap(scored, groups, contaminated, a.reps, rng)
                 if contaminated.any() else None,
                 "clean": bootstrap(scored, groups, ~contaminated, a.reps, rng)
                 if (~contaminated).any() else None,
                 "per_seed_own_test": [round(s.ap(np.ones(len(test), int)), 4) for s in scored]}

        unseen = ~twins(ref_bits, tr_bits, ref_test, train, a.tau, a.photo_id).any(1)
        ref_scored = [Scored(ref_test, predict(w, stages[a.reference], "test", ref_test, a.imgsz,
                                              a.device)) for w in weights]
        entry["reference_test_unseen"] = bootstrap(ref_scored, ref_groups, unseen, a.reps, rng)
        result["stages"][tag] = entry
        print(tag, json.dumps({k: v for k, v in entry.items() if k != "per_seed_own_test"}))

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=2))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
