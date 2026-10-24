#!/usr/bin/env python3
"""Schematic of the two-stage ocular classifier and how its labels were checked."""
import argparse, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK, MUTED = "#1d1d1b", "#5f5f5a"
BLUE, GREEN, ORANGE, RED = "#0072B2", "#009E73", "#E69F00", "#D55E00"


def box(ax, x, y, w, h, edge, title, lines):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.025",
                                fc="white", ec=edge, lw=1.3))
    ax.add_patch(FancyBboxPatch((x, y + h - 0.22), w, 0.22,
                                boxstyle="round,pad=0.012,rounding_size=0.025", fc=edge, ec=edge,
                                lw=1.3))
    ax.text(x + w / 2, y + h - 0.11, title, ha="center", va="center", color="white",
            fontsize=7, fontweight="bold")
    ax.text(x + 0.03, y + h - 0.29, "\n".join(lines), ha="left", va="top", color=INK,
            fontsize=6.3, linespacing=1.35)


def arrow(ax, x0, x1, y):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=9,
                                 color=MUTED, lw=1.1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="outputs/figures/v31/summary.json")
    ap.add_argument("--out", default="paper/figures/fig_transfer")
    a = ap.parse_args()
    s = json.loads(Path(a.summary).read_text())
    fx, sp = s["classifier"]["fixed_split"], s["classifier"]["split_sweep"]
    fundus = s["fundus"]["tta_flip"]
    rate = s["assessability"]["consensus"]["not_assessable_by_majority_pct"]

    plt.rcParams.update({"font.family": "Liberation Sans"})
    fig = plt.figure(figsize=(181 / 25.4, 34 / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 4.0)
    ax.set_ylim(0, 1.0)
    ax.set_axis_off()
    w, h, y = 0.86, 0.90, 0.05
    xs = [0.04, 1.04, 2.04, 3.04]
    box(ax, xs[0], y, w, h, MUTED, "Initialisation",
        ["EfficientNet-B2", "7.7M parameters", "ImageNet weights"])
    box(ax, xs[1], y, w, h, GREEN, "Stage 1: human fundus",
        ["1896 public photographs", "cataract 937 / normal 959", "full fine-tuning, 80/20 split",
         f"held-out AUC {fundus['auc_roc']:.3f}", f"[{fundus['auc_ci95'][0]:.3f}, "
         f"{fundus['auc_ci95'][1]:.3f}], n = 379"])
    box(ax, xs[2], y, w, h, ORANGE, "Stage 2: alpaca eye crops",
        ["462 crops, labels by a VLM", "(371 normal / 91 anomaly)", "no expert review",
         "focal loss, group-aware split", f"AUC vs labels {fx['auc_mean']:.2f}–{sp['auc_mean']:.2f}"])
    box(ax, xs[3], y, w, h, BLUE, "Check of the labels",
        ["3 authors rated all crops", "blind to the labels", f"{rate:.0f}% not assessable",
         "(majority)", "2 of 91 'anomalies' show", "a visible sign"])
    for i in range(3):
        arrow(ax, xs[i] + w + 0.015, xs[i + 1] - 0.015, y + h / 2)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=600)
    fig.savefig(out.with_suffix(".tiff"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
