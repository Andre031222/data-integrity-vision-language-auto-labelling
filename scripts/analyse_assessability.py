#!/usr/bin/env python3
"""Agreement of the authors' blind assessability ratings and their relation to the VLM auto-labels."""
import argparse, csv, itertools, json
from pathlib import Path

import numpy as np
from PIL import Image


def read_ratings(path):
    rows = list(csv.DictReader(open(path)))
    return {r["archivo"]: r for r in rows}


def cohen(a, b):
    po = np.mean(a == b)
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def fleiss(x):
    n, k = x.shape
    p1 = x.mean()
    counts = np.stack([x.sum(1), k - x.sum(1)], 1)
    pi = (np.sum(counts ** 2, 1) - k) / (k * (k - 1))
    pe = p1 ** 2 + (1 - p1) ** 2
    return float((pi.mean() - pe) / (1 - pe)) if pe < 1 else float("nan")


def ci(stat, x, reps, rng):
    n = len(x)
    draws = [stat(x[rng.integers(0, n, n)]) for _ in range(reps)]
    draws = [d for d in draws if np.isfinite(d)]
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return [round(float(lo), 3), round(float(hi), 3)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", default="data/vet_labeling/batch_01")
    ap.add_argument("--raters", nargs="+", default=["VIQ", "DMYY", "RAVS"])
    ap.add_argument("--reps", type=int, default=5000)
    ap.add_argument("--out", default="outputs/figures/v31/assessability_agreement.json")
    a = ap.parse_args()

    batch = Path(a.batch)
    ratings = {r: read_ratings(batch / "resultados_evaluabilidad" / f"evaluabilidad_{r}.csv")
               for r in a.raters}
    files = sorted(next(iter(ratings.values())))
    key = {r["archivo"]: r for r in csv.DictReader(open(batch / "_clave_interna.csv"))}
    prescreen = {r["id"] + ".jpg": r for r in csv.DictReader(open(batch / "ai_prescreen.csv"))}

    assess = np.array([[ratings[r][f]["evaluable"] == "si" for r in a.raters] for f in files], int)
    sign = np.array([[ratings[r][f]["signo"] == "visible" for r in a.raters] for f in files], int)
    side = np.array([min(Image.open(batch / "00_POR_REVISAR" / f).size) for f in files])
    vlm = np.array([key[f]["etiqueta_llm_previa"] == "anomaly" for f in files])
    rng = np.random.default_rng(0)

    result = {"crops": len(files), "raters": a.raters, "per_rater": {}, "pairwise_kappa": {}}
    for j, r in enumerate(a.raters):
        result["per_rater"][r] = {"assessable": int(assess[:, j].sum()),
                                  "assessable_pct": round(100 * assess[:, j].mean(), 1),
                                  "visible_sign": int(sign[:, j].sum())}
    for (i, ri), (j, rj) in itertools.combinations(enumerate(a.raters), 2):
        pair = assess[:, [i, j]]
        result["pairwise_kappa"][f"{ri}-{rj}"] = {
            "kappa": round(cohen(pair[:, 0], pair[:, 1]), 3),
            "ci95": ci(lambda x: cohen(x[:, 0], x[:, 1]), pair, a.reps, rng),
            "raw_agreement": round(float(np.mean(pair[:, 0] == pair[:, 1])), 3)}
    result["fleiss_kappa"] = {"kappa": round(fleiss(assess), 3),
                              "ci95": ci(fleiss, assess, a.reps, rng)}

    votes = assess.sum(1)
    majority = votes * 2 > len(a.raters)
    result["consensus"] = {
        "assessable_by_all": int((votes == len(a.raters)).sum()),
        "assessable_by_majority": int(majority.sum()),
        "assessable_by_none": int((votes == 0).sum()),
        "not_assessable_by_majority_pct": round(100 * (1 - majority.mean()), 1),
        "visible_sign_by_any": int((sign.sum(1) > 0).sum()),
        "visible_sign_by_majority": int((sign.sum(1) * 2 > len(a.raters)).sum())}

    result["vlm_anomaly_labels"] = {
        "total": int(vlm.sum()),
        "assessable_by_majority": int((vlm & majority).sum()),
        "visible_sign_by_any_rater": int((vlm & (sign.sum(1) > 0)).sum()),
        "visible_sign_by_majority": int((vlm & (sign.sum(1) * 2 > len(a.raters))).sum())}

    bins = [(0, 32), (32, 64), (64, 100), (100, 10 ** 6)]
    result["by_crop_side_px"] = {
        f"{lo}-{hi if hi < 10 ** 6 else 'max'}": {
            "crops": int(((side >= lo) & (side < hi)).sum()),
            "assessable_by_majority": int((majority & (side >= lo) & (side < hi)).sum())}
        for lo, hi in bins}
    result["median_side_px"] = float(np.median(side))
    result["median_side_px_assessable_by_majority"] = float(np.median(side[majority])) \
        if majority.any() else None

    ai = np.array([prescreen[f]["ai_label"] != "no_evaluable" for f in files], int)
    result["vs_ai_prescreen"] = {
        "ai_assessable": int(ai.sum()),
        "kappa_ai_vs_majority": round(cohen(ai, majority.astype(int)), 3),
        "ci95": ci(lambda x: cohen(x[:, 0], x[:, 1]), np.stack([ai, majority.astype(int)], 1),
                   a.reps, rng)}

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
