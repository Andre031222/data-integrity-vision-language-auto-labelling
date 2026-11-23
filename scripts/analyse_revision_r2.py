#!/usr/bin/env python3
"""Analyses requested in the second review round: SD ratios, the lost-copy contrast and a photograph-weighted benchmark score."""
import argparse, json, statistics, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyse_contamination as A
from analyse_revision_extra import mean_ap, scored

RESULTS = Path("outputs/figures/v31")
RELEASED = Path("data/v3.1/s6photoid")


def ci(draws):
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


def sd_ratios():
    out = {}
    for stage in ("s3src", "s4phash", "s5dihedral", "s6photoid"):
        main = json.loads((RESULTS / f"{stage}.json").read_text())
        seeds = main["summary"]["mAP50"]["values"]
        splits = [main["per_seed"]["0"]["mAP50"]] + [
            json.loads((RESULTS / f"{stage}_split{k}.json").read_text())["per_seed"]["0"]["mAP50"]
            for k in range(1, 5)]
        s_seed, s_split = statistics.stdev(seeds), statistics.stdev(splits)
        out[stage] = {"seed_sd": round(s_seed, 4), "split_sd": round(s_split, 4),
                      "ratio": round(s_split / s_seed, 1), "split_mean": round(statistics.mean(splits), 4),
                      "estimator": "sample SD (n-1) for both"}
    return out


def lost_copy_contrast(runs, reps, rng):
    full, sub = "data/v3.1/s1raw", "data/v3.1/size_control/s1raw_n1035"
    test = A.list_split(full, "test")
    bits = A.dihedral_bits(test)
    tr_full, tr_sub = A.list_split(full, "train"), A.list_split(sub, "train")
    c_full = A.twins(bits, A.dihedral_bits(tr_full), test, tr_full, 6, True).any(1)
    c_sub = A.twins(bits, A.dihedral_bits(tr_sub), test, tr_sub, 6, True).any(1)
    lost, kept = c_full & ~c_sub, c_full & c_sub
    groups = A.scene_groups(bits, 6, test)
    m_full, m_sub = scored("s1raw", full, test, runs), scored("s1raw_n1035", full, test, runs)

    def contrast(w):
        d_lost = mean_ap(m_full, w * lost) - mean_ap(m_sub, w * lost)
        d_kept = mean_ap(m_full, w * kept) - mean_ap(m_sub, w * kept)
        return d_lost, d_kept, d_lost - d_kept

    point = contrast(np.ones(len(test), int))
    idx = np.nonzero(lost | kept)[0]
    uniq = np.unique(groups[idx])
    members = {g: idx[groups[idx] == g] for g in uniq}
    draws = []
    for _ in range(reps):
        w = np.zeros(len(test), int)
        for g in rng.choice(uniq, len(uniq), replace=True):
            w[members[g]] += 1
        if (w * lost).sum() and (w * kept).sum():
            draws.append(contrast(w))
    draws = np.array(draws)
    return {"images_losing_copy": int(lost.sum()), "images_keeping_copy": int(kept.sum()),
            "groups_losing_copy": int(len(np.unique(groups[lost]))),
            "groups_keeping_copy": int(len(np.unique(groups[kept]))),
            "drop_copy_lost": {"diff": round(point[0], 4), "ci95": ci(draws[:, 0])},
            "drop_copy_kept": {"diff": round(point[1], 4), "ci95": ci(draws[:, 1])},
            "difference_of_differences": {"diff": round(point[2], 4), "ci95": ci(draws[:, 2]),
                                          "share_of_draws_above_zero": round(float((draws[:, 2] > 0).mean()), 3)},
            "bootstrap_reps_used": int(len(draws))}


def photo_weighted_benchmark(runs, reps, draws_per_estimate, rng, list_out):
    test = A.list_split(RELEASED, "test")
    bits = A.dihedral_bits(test)
    groups = A.scene_groups(bits, 6, test)
    uniq = np.unique(groups)
    members = {g: np.nonzero(groups == g)[0] for g in uniq}
    models = scored("s6photoid", str(RELEASED), test, runs)

    canonical = np.zeros(len(test), int)
    chosen = []
    for g in uniq:
        i = min(members[g], key=lambda k: test[k]["name"])
        canonical[i] = 1
        chosen.append(test[i]["name"])
    Path(list_out).write_text("\n".join(sorted(chosen)) + "\n")

    def one_per_group(group_draw):
        w = np.zeros(len(test), int)
        for g in group_draw:
            w[rng.choice(members[g])] += 1
        return w

    expected = [np.mean([m.ap(one_per_group(uniq)) for _ in range(draws_per_estimate)]) for m in models]
    boot = []
    for _ in range(reps):
        w = one_per_group(rng.choice(uniq, len(uniq), replace=True))
        boot.append(np.mean([m.ap(w) for m in models]))
    sizes = np.array([len(members[g]) for g in uniq])
    return {"test_images": len(test), "photographs": int(len(uniq)),
            "images_per_photograph": {"min": int(sizes.min()), "median": float(np.median(sizes)),
                                      "max": int(sizes.max())},
            "per_image_map50": round(mean_ap(models, np.ones(len(test), int)), 4),
            "per_image_map50_seed0": round(models[0].ap(np.ones(len(test), int)), 4),
            "canonical_list_map50": round(mean_ap(models, canonical), 4),
            "canonical_list_map50_seed0": round(models[0].ap(canonical), 4),
            "photo_weighted_map50": round(float(np.mean(expected)), 4),
            "photo_weighted_map50_seed0": round(float(expected[0]), 4),
            "photo_weighted_ci95": ci(boot),
            "canonical_list": str(list_out),
            "canonical_rule": "lexicographically first file name within each photograph"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="outputs/training_runs_v31")
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--list-out", default=str(RELEASED / "test_one_per_photograph.txt"))
    ap.add_argument("--out", default=str(RESULTS / "r2_detector_analyses.json"))
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    res = {"sd_ratios": sd_ratios()}
    print(json.dumps(res, indent=2), flush=True)
    res["lost_copy_contrast"] = lost_copy_contrast(a.runs, a.reps, rng)
    print(json.dumps(res["lost_copy_contrast"], indent=2), flush=True)
    res["photo_weighted_benchmark"] = photo_weighted_benchmark(a.runs, a.reps, a.draws, rng, a.list_out)
    print(json.dumps(res["photo_weighted_benchmark"], indent=2), flush=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
