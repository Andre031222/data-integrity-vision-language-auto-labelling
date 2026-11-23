#!/usr/bin/env python3
"""Compare the ocular classifier with crop size on identical held-out crops."""
import argparse, csv, json, sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.group_split import source_photo

RATINGS = Path("outputs/figures/v31/ocular/ocular_crop_ratings.csv")
KEY = Path("data/vet_labeling/batch_01/_clave_interna.csv")


def crop_sizes():
    side = {r["crop"]: float(r["smaller_side_px"]) for r in csv.DictReader(open(RATINGS))}
    return {r["archivo_original"]: side[r["archivo"]] for r in csv.DictReader(open(KEY))}


def stratified_auc(y, score, strata):
    num = den = 0.0
    for s in np.unique(strata):
        m = strata == s
        pos, neg = score[m & (y == 1)], score[m & (y == 0)]
        if len(pos) and len(neg):
            diff = pos[:, None] - neg[None, :]
            num += (diff > 0).sum() + 0.5 * (diff == 0).sum()
            den += len(pos) * len(neg)
    return num / den if den else float("nan")


def run_stats(r, sizes, edges, reps, rng):
    y = np.array(r["y_true"])
    p = np.array(r["p_anomaly"])
    s = np.array([sizes[f] for f in r["test_files"]])
    strata = np.digitize(s, edges)
    groups = np.array([source_photo(f) for f in r["test_files"]])
    uniq = np.unique(groups)
    members = {g: np.nonzero(groups == g)[0] for g in uniq}
    point = {"classifier": roc_auc_score(y, p), "crop_size": roc_auc_score(y, s),
             "classifier_within_size_tertiles": stratified_auc(y, p, strata)}
    diffs = []
    for _ in range(reps):
        idx = np.concatenate([members[g] for g in rng.choice(uniq, len(uniq), replace=True)])
        if 0 < y[idx].sum() < len(idx):
            diffs.append(roc_auc_score(y[idx], p[idx]) - roc_auc_score(y[idx], s[idx]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {**{k: round(float(v), 4) for k, v in point.items()},
            "classifier_minus_size": round(float(point["classifier"] - point["crop_size"]), 4),
            "classifier_minus_size_ci95": [round(float(lo), 4), round(float(hi), 4)],
            "test_photographs": int(len(uniq))}


def summarise(per_run):
    keys = ("classifier", "crop_size", "classifier_within_size_tertiles", "classifier_minus_size")
    out = {k: {"mean": round(float(np.mean([r[k] for r in per_run])), 4),
               "sd": round(float(np.std([r[k] for r in per_run], ddof=1)), 4)} for k in keys}
    out["runs_size_beats_classifier"] = int(sum(r["classifier_minus_size"] < 0 for r in per_run))
    out["runs_ci_excludes_zero"] = int(sum(r["classifier_minus_size_ci95"][1] < 0 or
                                           r["classifier_minus_size_ci95"][0] > 0 for r in per_run))
    out["runs"] = len(per_run)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=[
        "outputs/figures/v31/classifier_fixed_split.json",
        "outputs/figures/v31/classifier_split_sweep.json",
        "outputs/figures/v31/classifier_fixed_split_photo.json",
        "outputs/figures/v31/classifier_split_sweep_photo.json"])
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--out", default="outputs/figures/v31/ocular/shortcut_same_test.json")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    sizes = crop_sizes()
    edges = np.quantile(list(sizes.values()), [1 / 3, 2 / 3])
    all_y = []
    res = {"size_tertile_edges_px": [round(float(e), 1) for e in edges], "designs": {}}
    for f in a.runs:
        if not Path(f).exists():
            continue
        per_run = [run_stats(r, sizes, edges, a.reps, rng) for r in json.load(open(f))["per_seed"]]
        res["designs"][Path(f).stem.replace("classifier_", "")] = {"summary": summarise(per_run),
                                                                  "per_run": per_run}
    Path(a.out).write_text(json.dumps(res, indent=2))
    for k, v in res["designs"].items():
        print(k, json.dumps({kk: vv for kk, vv in v["summary"].items()}))


if __name__ == "__main__":
    main()
