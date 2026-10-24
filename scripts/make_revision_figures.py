#!/usr/bin/env python3
"""Draw the data figures of the revised manuscript from outputs/figures/v31/summary.json."""
import argparse, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, GREEN, RED = "#0072B2", "#E69F00", "#009E73", "#D55E00"
GREY, INK, MUTED = "#9a9a9a", "#1d1d1b", "#5f5f5a"
MM = 1 / 25.4


def style():
    plt.rcParams.update({
        "font.family": "Liberation Sans", "font.size": 7, "axes.titlesize": 7.5,
        "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.edgecolor": MUTED,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK,
        "axes.grid": False, "legend.frameon": False, "legend.fontsize": 6.5,
    })


def panel(ax, letter, title):
    ax.set_title(f"({letter}) {title}", loc="left", fontweight="bold", color=INK)


def save(fig, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=600)
    fig.savefig(out.with_suffix(".tiff"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print(f"wrote {out}")


def hbar_labels(ax, bars, values, fmt="{:,}"):
    for b, v in zip(bars, values):
        ax.text(b.get_width() + ax.get_xlim()[1] * 0.01, b.get_y() + b.get_height() / 2,
                fmt.format(v), va="center", fontsize=6.5, color=INK)


def fig_leakage(s, out):
    stages = ["s1raw", "s2sha", "s3src", "s4phash", "s5dihedral", "s6photoid"]
    names = ["1 Raw", "2 Exact", "3 Source", "4 pHash", "5 Dihedral", "6 Photo ID"]
    fig, axes = plt.subplots(1, 3, figsize=(181 * MM, 66 * MM),
                             gridspec_kw={"width_ratios": [1.35, 1.25, 0.9]})

    ax = axes[0]
    x = np.arange(len(stages))
    for i, st in enumerate(stages):
        per_seed = s["contamination"][st]["per_seed_own_test"]
        ax.scatter(np.full(3, i) + np.array([-0.08, 0, 0.08]), per_seed, s=9, color=GREY, zorder=2,
                   lw=0)
        if st in s["split_variance"]:
            sv = s["split_variance"][st]
            ax.errorbar(i + 0.22, sv["mean"], yerr=sv["sd"], fmt="D", ms=4, color=BLUE, mfc=BLUE,
                        ecolor=BLUE, elinewidth=1, capsize=2, zorder=3)
        if st in s["size_control"]:
            sc = s["size_control"][st]["map50"]
            ax.errorbar(i + 0.22, sc[0], yerr=sc[1], fmt="o", ms=4.5, mfc="white", color=RED,
                        mew=1.2, elinewidth=1, capsize=2, zorder=3)
    ax.set_xticks(x, names, rotation=35, ha="right", rotation_mode="anchor")
    ax.set_ylabel("mAP@0.5 on the stage's own test set")
    ax.set_ylim(0.74, 0.96)
    ax.set_xlim(-0.5, len(stages) - 0.4)
    handles = [plt.Line2D([], [], ls="", marker="o", ms=3.5, color=GREY, label="3 seeds, split 42"),
               plt.Line2D([], [], ls="", marker="D", ms=4, color=BLUE, label="5 splits, mean $\\pm$ SD"),
               plt.Line2D([], [], ls="", marker="o", ms=4.5, mfc="white", color=RED, mew=1.2,
                          label="Train cut to 1035 images")]
    ax.legend(handles=handles, loc="lower left", handletextpad=0.3, borderaxespad=0.2)
    panel(ax, "A", "Six constructions of the dataset")

    ax = axes[1]
    rows = [("1 Raw", "s1raw"), ("1 Raw, 1035", "s1raw_n1035"), ("2 Dedup.", "s2sha"),
            ("2 Dedup., 1035", "s2sha_n1035"), ("3 Source filters", "s3src"),
            ("4 Percept. hash", "s4phash"), ("5 Dihedral hash", "s5dihedral"),
            ("6 Released", "s6photoid")]
    for j, (label, st) in enumerate(rows):
        y = len(rows) - 1 - j
        c = s["contamination"][st]
        clean, cont = c["clean"], c["contaminated"]
        if cont:
            ax.plot([clean["map50"], cont["map50"]], [y, y], color="#cfcfca", lw=1.6, zorder=1,
                    solid_capstyle="round")
            ax.errorbar(cont["map50"], y + 0.12, xerr=[[cont["map50"] - cont["ci95"][0]],
                                                       [cont["ci95"][1] - cont["map50"]]],
                        fmt="o", ms=4, color=RED, elinewidth=0.8, capsize=1.5, zorder=3)
        ax.errorbar(clean["map50"], y - 0.12, xerr=[[clean["map50"] - clean["ci95"][0]],
                                                    [clean["ci95"][1] - clean["map50"]]],
                    fmt="s", ms=3.8, color=BLUE, elinewidth=0.8, capsize=1.5, zorder=3)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1])
    ax.set_xlabel("mAP@0.5 (95% CI over scene groups)")
    ax.set_xlim(0.6, 1.0)
    ax.legend(handles=[plt.Line2D([], [], ls="", marker="o", ms=4, color=RED, label="With a training copy"),
                       plt.Line2D([], [], ls="", marker="s", ms=3.8, color=BLUE, label="Without")],
              loc="lower left", handletextpad=0.3, borderaxespad=0.2)
    panel(ax, "B", "Test images with and without a copy")

    ax = axes[2]
    au = s["neardup_audit"]
    labels = ["Bytes", "pHash", "Dihedral", "+ Photo ID"]
    vals = [0.0, au["perceptual"]["6"]["pct"], au["dihedral"]["6"]["pct"],
            au["photo_id_or_dihedral"]["6"]["pct"]]
    bars = ax.bar(range(4), vals, width=0.62, color=BLUE, edgecolor="white", lw=1)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}%", ha="center", fontsize=6.5,
                color=INK)
    ax.set_xticks(range(4), labels, rotation=35, ha="right", rotation_mode="anchor")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Test images with a training copy (%)")
    panel(ax, "C", "Notion of identity")

    fig.tight_layout(w_pad=1.2)
    save(fig, out)


def fig_dataset(s, out, manifest):
    fig, axes = plt.subplots(1, 3, figsize=(181 * MM, 62 * MM),
                             gridspec_kw={"width_ratios": [1.45, 0.9, 1.0]})
    ax = axes[0]
    steps = [("Consolidated files", 3419, GREY), ("Byte-identical copies", 1041, RED),
             ("Licence undefined", 556, RED), ("Annotates heads", 322, RED),
             ("No live camelid", 39, RED), ("Released images", 1461, BLUE),
             ("Dihedral scene groups", 613, GREEN), ("Distinct photographs", 517, GREEN)]
    y = np.arange(len(steps))[::-1]
    bars = ax.barh(y, [v for _, v, _ in steps], color=[c for *_, c in steps], height=0.62,
                   edgecolor="white", lw=1)
    ax.set_yticks(y, [n for n, *_ in steps])
    ax.set_xlim(0, 4200)
    ax.set_xlabel("Count")
    for b, (_, v, c) in zip(bars, steps):
        sign = "−" if c == RED else ""
        ax.text(b.get_width() + 60, b.get_y() + b.get_height() / 2, f"{sign}{v:,}", va="center",
                fontsize=6.5, color=INK)
    panel(ax, "A", "From files to photographs")

    ax = axes[1]
    splits = ["train", "val", "test"]
    photos = [manifest["per_split_groups"][k] for k in splits]
    images = [manifest["per_split_images"][k] for k in splits]
    xx = np.arange(3)
    b1 = ax.bar(xx - 0.19, photos, 0.36, color=GREEN, edgecolor="white", lw=1, label="Photographs")
    b2 = ax.bar(xx + 0.19, images, 0.36, color=BLUE, edgecolor="white", lw=1, label="Images")
    for bars, vals in ((b1, photos), (b2, images)):
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 15, str(v), ha="center", fontsize=6, color=INK)
    ax.set_xticks(xx, ["Train", "Validation", "Test"])
    ax.set_ylim(0, 1180)
    ax.set_ylabel("Count")
    ax.legend(loc="upper right")
    panel(ax, "B", "Released split")

    ax = axes[2]
    bins = s["assessability"]["by_crop_side_px"]
    labels = ["<32", "32–63", "64–99", "≥100"]
    total = [v["crops"] for v in bins.values()]
    ok = [v["assessable_by_majority"] for v in bins.values()]
    xx = np.arange(4)
    ax.bar(xx, [t - o for t, o in zip(total, ok)], 0.62, color="#c9c9c4", edgecolor="white", lw=1,
           label="Not assessable")
    ax.bar(xx, ok, 0.62, bottom=[t - o for t, o in zip(total, ok)], color=BLUE, edgecolor="white",
           lw=1, label="Assessable (majority)")
    for i, (t, o) in enumerate(zip(total, ok)):
        ax.text(i, t + 4, f"{o}/{t}", ha="center", fontsize=6.5, color=INK)
    ax.set_xticks(xx, labels)
    ax.set_xlabel("Smaller side of the crop (px)")
    ax.set_ylabel("Crops")
    ax.set_ylim(0, 330)
    ax.legend(loc="upper right", borderaxespad=0.1)
    panel(ax, "C", "Ocular crops by size")
    fig.tight_layout(w_pad=1.2)
    save(fig, out)


def fig_classifier(s, out, runs):
    fig, axes = plt.subplots(1, 3, figsize=(181 * MM, 60 * MM),
                             gridspec_kw={"width_ratios": [1.1, 1.0, 1.0]})
    ax = axes[0]
    rng = np.random.default_rng(1)
    for i, (name, key) in enumerate((("Fixed split,\n10 seeds", "fixed_split"),
                                     ("10 different\nsplits", "split_sweep"))):
        aucs = [r["auc_roc"] for r in runs[key]["per_seed"]]
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(aucs)), aucs, s=12, color=BLUE, lw=0, zorder=3)
        m = np.mean(aucs)
        ax.plot([i - 0.25, i + 0.25], [m, m], color=INK, lw=1.2)
        ax.text(i + 0.28, m, f"{m:.2f}", va="center", fontsize=6.5, color=INK)
    fundus = s["fundus"]["tta_flip"]["auc_roc"]
    ax.scatter([2], [fundus], s=18, marker="D", color=GREEN, zorder=3)
    ax.text(2, fundus - 0.05, f"{fundus:.3f}", ha="center", fontsize=6.5, color=INK)
    ax.axhline(0.5, color=GREY, lw=0.8, ls="--")
    ax.text(2.45, 0.505, "chance", fontsize=6, color=MUTED, ha="right", va="bottom")
    ax.set_xticks([0, 1, 2], ["Fixed split,\n10 seeds", "10 different\nsplits", "Human fundus\n(control)"])
    ax.set_ylim(0.4, 1.02)
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylabel("AUC-ROC against held-out labels")
    panel(ax, "A", "Classifier score")

    ax = axes[1]
    a = s["assessability"]
    names = ["Rater A", "Rater B", "Rater C", "Majority"]
    vals = [a["per_rater"][r]["assessable"] for r in a["raters"]] + \
           [a["consensus"]["assessable_by_majority"]]
    bars = ax.bar(range(4), vals, 0.62, color=[BLUE] * 3 + [INK], edgecolor="white", lw=1)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v}\n({100 * v / 462:.0f}%)", ha="center",
                fontsize=6, color=INK)
    ax.set_xticks(range(4), names)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Crops judged assessable (of 462)")
    ax.text(0.98, 0.97, f"Fleiss' $\\kappa$ = {a['fleiss_kappa']['kappa']:.2f}\n"
            f"[{a['fleiss_kappa']['ci95'][0]:.2f}, {a['fleiss_kappa']['ci95'][1]:.2f}]",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.5, color=INK)
    panel(ax, "B", "Blind human rating")

    ax = axes[2]
    v = a["vlm_anomaly_labels"]
    steps = [("VLM 'anomaly'", v["total"], ORANGE), ("Assessable\n(majority)", v["assessable_by_majority"], BLUE),
             ("Visible sign\n(majority)", v["visible_sign_by_majority"], INK)]
    bars = ax.bar(range(3), [x for _, x, _ in steps], 0.62, color=[c for *_, c in steps],
                  edgecolor="white", lw=1)
    for b, (_, x, _) in zip(bars, steps):
        ax.text(b.get_x() + b.get_width() / 2, x + 1.5, str(x), ha="center", fontsize=6.5, color=INK)
    ax.set_xticks(range(3), [n for n, *_ in steps])
    ax.set_ylim(0, 105)
    ax.set_ylabel("Crops")
    panel(ax, "C", "What the anomaly labels hold")
    fig.tight_layout(w_pad=1.2)
    save(fig, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="outputs/figures/v31/summary.json")
    ap.add_argument("--manifest", default="data/v3.1/s6photoid/build_manifest.json")
    ap.add_argument("--out", default="paper/figures_R")
    a = ap.parse_args()
    style()
    s = json.loads(Path(a.summary).read_text())
    runs = {k: json.loads(Path(f"outputs/figures/v31/classifier_{k}.json").read_text())
            for k in ("fixed_split", "split_sweep")}
    out = Path(a.out)
    fig_leakage(s, out / "fig_leakage")
    fig_dataset(s, out / "fig_dataset", json.loads(Path(a.manifest).read_text()))
    fig_classifier(s, out / "fig_classifier", runs)


if __name__ == "__main__":
    main()
