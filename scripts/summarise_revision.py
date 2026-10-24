#!/usr/bin/env python3
"""Collect every number reported in the revised manuscript into one summary file."""
import argparse, json, statistics
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

STAGES = ["s1raw", "s2sha", "s3src", "s4phash", "s5dihedral", "s6photoid"]
SPLIT_STAGES = ["s3src", "s4phash", "s5dihedral", "s6photoid"]


def load(d, name):
    return json.loads((d / f"{name}.json").read_text())


def train_size(stage_dir):
    return {sp: sum(1 for _ in (stage_dir / "images" / sp).iterdir())
            for sp in ("train", "val", "test")}


def per_run_ci(y, p, reps, rng):
    y, p = np.array(y), np.array(p)
    draws = []
    for _ in range(reps):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            draws.append(roc_auc_score(y[i], p[i]))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


def classifier_summary(c, reps, rng):
    widths, f1s = [], []
    for run in c["per_seed"]:
        lo, hi = per_run_ci(run["y_true"], run["p_anomaly"], reps, rng)
        widths.append(hi - lo)
        f1s.append(run["f1_anomaly"])
    aucs = [r["auc_roc"] for r in c["per_seed"]]
    return {"runs": len(aucs), "auc_mean": round(statistics.mean(aucs), 3),
            "auc_sd": round(statistics.pstdev(aucs), 3),
            "auc_range": [round(min(aucs), 3), round(max(aucs), 3)],
            "mean_ci_width": round(statistics.mean(widths), 3),
            "f1_range": [round(min(f1s), 3), round(max(f1s), 3)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="outputs/figures/v31")
    ap.add_argument("--data", default="data/v3.1")
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--out", default="outputs/figures/v31/summary.json")
    a = ap.parse_args()
    d, data = Path(a.results), Path(a.data)
    rng = np.random.default_rng(0)
    out = {"ablation": {}, "split_variance": {}, "size_control": {}, "capacity": {}}

    for st in STAGES:
        r = load(d, st)["summary"]
        sizes = train_size(data / st)
        out["ablation"][st] = {"train_images": sizes["train"],
                               "test_images": sizes.get("test"),
                               "map50": [r["mAP50"]["mean"], r["mAP50"]["std"]],
                               "map50_95": [r["mAP50_95"]["mean"], r["mAP50_95"]["std"]],
                               "precision": r["precision"]["mean"],
                               "recall": r["recall"]["mean"]}
    for st in SPLIT_STAGES:
        vals = [load(d, st)["per_seed"]["0"]["mAP50"]]
        vals += [load(d, f"{st}_split{k}")["summary"]["mAP50"]["mean"] for k in (1, 2, 3, 4)]
        out["split_variance"][st] = {"values": vals, "mean": round(statistics.mean(vals), 4),
                                     "sd": round(statistics.stdev(vals), 4),
                                     "range": [min(vals), max(vals)]}
    for st in ("s1raw", "s2sha"):
        tag = next(p.stem for p in d.glob(f"{st}_n*.json") if "1035" in p.stem)
        r = load(d, tag)["summary"]
        out["size_control"][st] = {"tag": tag, "map50": [r["mAP50"]["mean"], r["mAP50"]["std"]],
                                   "map50_95": [r["mAP50_95"]["mean"], r["mAP50_95"]["std"]]}
    for tag, name in (("s6photoid", "yolo11n"), ("s6photoid_yolo11s", "yolo11s")):
        r = load(d, tag)["summary"]
        out["capacity"][name] = {k: [r[k]["mean"], r[k]["std"]]
                                 for k in ("mAP50", "mAP50_95", "precision", "recall")}
        out["capacity"][name]["per_seed"] = load(d, tag)["per_seed"]

    out["contamination"] = load(d, "contamination_analysis")["stages"]
    out["neardup_audit"] = load(d, "neardup_audit")["results"]
    out["assessability"] = load(d, "assessability_agreement")
    out["classifier"] = {"fixed_split": classifier_summary(load(d, "classifier_fixed_split"), a.reps, rng),
                         "split_sweep": classifier_summary(load(d, "classifier_split_sweep"), a.reps, rng)}
    out["fundus"] = json.loads(Path("outputs/figures/stage1_fundus_metrics.json").read_text())
    out["dedup_cost"] = load(d, "dedup_method_comparison")
    out["latency"] = load(d, "latency")

    Path(a.out).write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("ablation", "split_variance", "size_control", "classifier")},
                     indent=1))


if __name__ == "__main__":
    main()
