"""AlpacaVision AI -- Regenerate HONEST summary figures (fig3 detector, fig4 classifier, fig11."""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT_FIG = ROOT / "paper" / "figures"
OUT_FIG.mkdir(parents=True, exist_ok=True)
OUTPUTS = ROOT / "outputs" / "figures"

DPI = 300

BLUE = "#1F77B4"
GREEN = "#27AE60"
AMBER = "#E67E22"
RED = "#D62728"
TEAL = "#14b8a6"
GRAY = "#6b7280"
DARK = "#2C3E50"
GRAD = "#8E44AD"

# ---- HONEST NUMBERS (single source of truth) -------------------------------
DET = {"mAP@0.5": 0.860, "mAP@0.5:0.95": 0.693, "Precision": 0.913, "Recall": 0.731}

CLS = json.loads((OUTPUTS / "classifier_eyes_honest_metrics.json").read_text())

# dataset
DATASET_UNIQUE = 2051
DATASET_FILES = 3088
DATASET_DUP = 1037
SPLIT = {"Train": 1435, "Val": 308, "Test": 308}
CROPS = {"anomaly": 91, "normal": 371}


def _save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT_FIG / ("%s.%s" % (name, ext)), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  %s  OK" % name)


def fig3_detector():
    """YOLOv11n detector honest test metrics (bar panel)."""
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    names = list(DET.keys())
    vals = [DET[k] for k in names]
    colors = [BLUE, GREEN, AMBER, RED]
    bars = ax.barh(names, vals, color=colors, height=0.6, edgecolor="white", linewidth=1.2)
    for bar, v in zip(bars, vals):
        ax.text(v + 0.008, bar.get_y() + bar.get_height() / 2,
                "%.3f" % v, va="center", fontsize=11, fontweight="bold")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Score")
    ax.invert_yaxis()
    ax.set_title("YOLOv11n Stage-1 Detector -- Scene-Level Test Set\n(n=224 images, 95 scene groups)",
                 fontsize=10.5, fontweight="bold")
    ax.grid(True, alpha=0.25, axis="x")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(pad=0.8)
    _save(fig, "fig3_detector")


def fig4_classifier():
    """Classifier honest test metrics -- shows chance-level performance."""
    auc = CLS["auc_roc"]
    ci = CLS["auc_roc_ci_95"]
    ap = CLS["average_precision"]
    f1 = CLS["f1_anomaly"]
    f1_ci = CLS["f1_anomaly_ci_95"]
    acc = CLS["accuracy"]
    n_test = CLS["n_test"]

    names = ["AUC-ROC", "Avg. Precision", "F1 (anomaly)", "Accuracy"]
    vals = [auc, ap, f1, acc]
    colors = [BLUE, GREEN, AMBER, TEAL]
    ypos = np.arange(len(names))[::-1]  # AUC on top

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    bars = ax.barh(ypos, vals, color=colors, height=0.6, edgecolor="white", linewidth=1.2)

    # error bars (95% CI) for AUC and F1
    xerr = np.zeros((2, len(vals)))
    xerr[0, 0] = auc - ci[0]; xerr[1, 0] = ci[1] - auc          # AUC (index 0)
    xerr[0, 2] = f1 - f1_ci[0]; xerr[1, 2] = f1_ci[1] - f1      # F1 (index 2)
    ax.errorbar(vals, ypos, xerr=xerr, fmt="none", ecolor="#374151", lw=1.8, capsize=4)

    for y, v in zip(ypos, vals):
        ax.text(v + 0.02, y, "%.3f" % v, va="center", fontsize=10.5, fontweight="bold")

    # chance reference line for AUC
    ax.axvline(0.5, color=RED, ls="--", lw=1.3, alpha=0.8)
    ax.text(0.5, 0.15, "chance\n(AUC=0.5)", color=RED, fontsize=7.5,
            ha="center", va="bottom")

    ax.set_yticks(ypos)
    ax.set_yticklabels(names)
    ax.set_xlim(0, 1.18)
    ax.set_xlabel("Score")
    ax.set_title("EfficientNet-B2 Ocular Classifier -- Honest Test Set\n"
                 "(n=%d, group-aware split; chance-level performance)" % n_test,
                 fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.25, axis="x")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(pad=0.8)
    _save(fig, "fig4_classifier")


def fig11_dataset_stats():
    """Honest dataset composition: unique images, splits, ocular crops."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), gridspec_kw={"wspace": 0.34})

    # Panel A: deduplication
    ax = axes[0]
    cats = ["Raw files", "Duplicates\n(MD5)", "Unique\nimages"]
    vals = [DATASET_FILES, DATASET_DUP, DATASET_UNIQUE]
    cols = [GRAY, RED, GREEN]
    bars = ax.bar(cats, vals, color=cols, width=0.62, edgecolor="white", linewidth=1.2)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 40,
                "%s" % format(v, ","), ha="center", va="bottom",
                fontsize=9.5, fontweight="bold")
    ax.set_ylabel("Images")
    ax.set_ylim(0, DATASET_FILES * 1.15)
    ax.set_title("A -- Detector Dataset Deduplication\n(MD5; 1,037 duplicates removed)",
                 fontweight="bold", fontsize=9.5)
    ax.spines[["top", "right"]].set_visible(False)

    # Panel B: honest detector splits
    ax = axes[1]
    sp_names = list(SPLIT.keys())
    sp_vals = [SPLIT[k] for k in sp_names]
    cols2 = [BLUE, GREEN, RED]
    bars = ax.bar(sp_names, sp_vals, color=cols2, width=0.58, edgecolor="white", linewidth=1.2)
    for bar, v in zip(bars, sp_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 18,
                str(v), ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Images")
    ax.set_ylim(0, max(sp_vals) * 1.18)
    ax.set_title("B -- Detector Split (honest)\n2,051 unique = 1435 / 308 / 308",
                 fontweight="bold", fontsize=9.5)
    ax.spines[["top", "right"]].set_visible(False)

    # Panel C: real ocular crops (group-aware, originals only)
    ax = axes[2]
    cc_names = ["Normal", "Anomaly"]
    cc_vals = [CROPS["normal"], CROPS["anomaly"]]
    cc_cols = [BLUE, RED]
    bars = ax.bar(cc_names, cc_vals, color=cc_cols, width=0.55, edgecolor="white", linewidth=1.2)
    for bar, v in zip(bars, cc_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 6,
                str(v), ha="center", va="bottom", fontsize=10.5, fontweight="bold")
    ax.set_ylabel("Real crops (image groups)")
    ax.set_ylim(0, CROPS["normal"] * 1.2)
    ax.set_title("C -- Ocular Crops (real, group-aware)\n91 anomaly / 371 normal",
                 fontweight="bold", fontsize=9.5)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout(pad=1.0)
    _save(fig, "fig11_dataset_stats")


if __name__ == "__main__":
    print("Generating honest summary figures -> %s" % OUT_FIG)
    fig3_detector()
    fig4_classifier()
    fig11_dataset_stats()
    print("DONE.")
