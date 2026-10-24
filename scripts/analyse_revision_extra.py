#!/usr/bin/env python3
"""Paired contrasts for the detector stages and shortcut checks for the ocular classifier."""
import argparse, csv, json, re, sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyse_contamination as A

STAGES = {"s1raw": "data/v3.1/s1raw", "s2sha": "data/v3.1/s2sha", "s3src": "data/v3.1/s3src",
          "s4phash": "data/v3.1/s4phash", "s5dihedral": "data/v3.1/s5dihedral",
          "s1raw_n1035": "data/v3.1/size_control/s1raw_n1035",
          "s2sha_n1035": "data/v3.1/size_control/s2sha_n1035"}
REF = ("s6photoid", "data/v3.1/s6photoid")


def scored(tag, root, records, runs):
    return [A.Scored(records, A.predict(Path(runs) / f"{tag}_seed{s}" / "weights" / "best.pt",
                                        root, "test", records, 640, "cpu")) for s in (0, 1, 2)]


def mean_ap(models, w):
    return float(np.mean([m.ap(w) for m in models]))


def paired(models_a, mask_a, models_b, mask_b, groups, reps, rng):
    union = mask_a | mask_b
    idx = np.nonzero(union)[0]
    uniq = np.unique(groups[idx])
    members = {g: idx[groups[idx] == g] for g in uniq}
    point = mean_ap(models_a, mask_a.astype(int)) - mean_ap(models_b, mask_b.astype(int))
    draws = []
    for _ in range(reps):
        w = np.zeros(len(union), int)
        for g in rng.choice(uniq, len(uniq), replace=True):
            w[members[g]] += 1
        da, db = w * mask_a, w * mask_b
        if da.sum() and db.sum():
            draws.append(mean_ap(models_a, da) - mean_ap(models_b, db))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"diff": round(point, 4), "ci95": [round(float(lo), 4), round(float(hi), 4)]}


def detector(a, rng):
    out = {"common_unseen": {}, "contaminated_minus_clean": {}}
    ref_test = A.list_split(REF[1], "test")
    ref_bits = A.dihedral_bits(ref_test)
    ref_groups = A.scene_groups(ref_bits, 6, ref_test)
    ref_models = scored(REF[0], REF[1], ref_test, a.runs)
    for tag, root in STAGES.items():
        train, test = A.list_split(root, "train"), A.list_split(root, "test")
        tr_bits, te_bits = A.dihedral_bits(train), A.dihedral_bits(test)
        unseen = ~A.twins(ref_bits, tr_bits, ref_test, train, 6, True).any(1)
        if unseen.sum() >= 10:
            stage_models = scored(tag, REF[1], ref_test, a.runs)
            res = paired(stage_models, unseen, ref_models, unseen, ref_groups, a.reps, rng)
            res.update({"images": int(unseen.sum()),
                        "groups": int(len(np.unique(ref_groups[unseen]))),
                        "stage_map50": round(mean_ap(stage_models, unseen.astype(int)), 4),
                        "released_map50": round(mean_ap(ref_models, unseen.astype(int)), 4)})
            out["common_unseen"][tag] = res
        cont = A.twins(te_bits, tr_bits, test, train, 6, True).any(1)
        groups = A.scene_groups(te_bits, 6, test)
        models = scored(tag, root, test, a.runs)
        out["contaminated_minus_clean"][tag] = paired(models, cont, models, ~cont, groups,
                                                      a.reps, rng)

    full, sub = STAGES["s1raw"], STAGES["s1raw_n1035"]
    test = A.list_split(full, "test")
    bits = A.dihedral_bits(test)
    c_full = A.twins(bits, A.dihedral_bits(A.list_split(full, "train")), test,
                     A.list_split(full, "train"), 6, True).any(1)
    c_sub = A.twins(bits, A.dihedral_bits(A.list_split(sub, "train")), test,
                    A.list_split(sub, "train"), 6, True).any(1)
    lost = c_full & ~c_sub
    groups = A.scene_groups(bits, 6, test)
    m_full, m_sub = scored("s1raw", full, test, a.runs), scored("s1raw_n1035", full, test, a.runs)
    kept = c_full & c_sub
    out["lost_copy_within_image"] = {
        "images_losing_copy": int(lost.sum()),
        "copy_lost": paired(m_full, lost, m_sub, lost, groups, a.reps, rng),
        "copy_kept_both": paired(m_full, kept, m_sub, kept, groups, a.reps, rng),
        "full_map_on_lost": round(mean_ap(m_full, lost.astype(int)), 4),
        "sub_map_on_lost": round(mean_ap(m_sub, lost.astype(int)), 4)}
    return out


def laplacian_var(path):
    g = np.asarray(Image.open(path).convert("L"), float)
    lap = g[1:-1, 1:-1] * -4 + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    return float(lap.var())


def ocular(a, rng):
    batch = Path("data/vet_labeling/batch_01")
    rows = list(csv.DictReader(open("outputs/figures/v31/ocular/ocular_crop_ratings.csv")))
    y = np.array([r["vlm_label_llama4scout"] == "anomaly" for r in rows], int)
    side = np.array([float(r["smaller_side_px"]) for r in rows])
    inat = np.array([r["source"] == "inaturalist" for r in rows], int)
    blur = np.array([laplacian_var(batch / "00_POR_REVISAR" / r["crop"]) for r in rows])
    assess = np.array([sum(r[f"rater_{k}_assessable"] == "yes" for k in "ABC") >= 2 for r in rows])
    sign = np.array([sum(r[f"rater_{k}_sign"] == "visible" for k in "ABC") >= 2 for r in rows])
    out = {"trivial_auc_vs_vlm_label": {
        "smaller_side": round(roc_auc_score(y, side), 3),
        "inaturalist_source": round(roc_auc_score(y, inat), 3),
        "sharpness_laplacian_var": round(roc_auc_score(y, blur), 3)},
        "vlm_by_majority_assessable": {
            "anomaly_assessable": int((y & assess).sum()), "anomaly_not": int((y & ~assess).sum()),
            "normal_assessable": int((~y.astype(bool) & assess).sum()),
            "normal_not": int((~y.astype(bool) & ~assess).sum())},
        "majority_visible_sign_by_vlm": {"anomaly": int((y & sign).sum()),
                                         "normal": int((~y.astype(bool) & sign).sum())}}
    key = {r["archivo_original"]: r["archivo"] for r in csv.DictReader(open(batch / "_clave_interna.csv"))}
    photo = lambda name: re.sub(r"_(cls\d+|eye)_.*$", "", name.replace("pseudo_", "").replace("aug_", ""))
    for design in ("fixed_split", "split_sweep"):
        runs = json.load(open(f"outputs/figures/v31/classifier_{design}.json"))["per_seed"]
        aucs = [r["auc_roc"] for r in runs]
        widths = []
        for r in runs:
            yt, p = np.array(r["y_true"]), np.array(r["p_anomaly"])
            g = np.array([photo(f) for f in r["test_files"]])
            uniq = np.unique(g)
            members = {u: np.nonzero(g == u)[0] for u in uniq}
            draws = []
            for _ in range(a.reps):
                idx = np.concatenate([members[u] for u in rng.choice(uniq, len(uniq), replace=True)])
                if 0 < yt[idx].sum() < len(idx):
                    draws.append(roc_auc_score(yt[idx], p[idx]))
            lo, hi = np.percentile(draws, [2.5, 97.5])
            widths.append(hi - lo)
        out[design] = {"runs_above_chance": int(sum(x > 0.5 for x in aucs)),
                       "sign_test_p": float(binomtest(sum(x > 0.5 for x in aucs), len(aucs), 0.5).pvalue),
                       "mean_ci_width_by_source_photo": round(float(np.mean(widths)), 3),
                       "test_source_photos": int(np.mean([len(set(photo(f) for f in r["test_files"])) for r in runs]))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="outputs/training_runs_v31")
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--out", default="outputs/figures/v31/extra_analyses.json")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    res = {"ocular": ocular(a, rng), "detector": detector(a, rng)}
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
