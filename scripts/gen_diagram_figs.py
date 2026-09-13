"""Redesigned publication diagrams (clean, consistent style) for the paper: - fig1_pipeline.pdf."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper" / "figures"
DPI = 300
plt.rcParams.update({
    "font.family": "DejaVu Sans", "savefig.dpi": DPI,
    "svg.fonttype": "none",
})

# validated colorblind-safe palette (light mode)
BLUE, AQUA, YELLOW, GREEN = "#2a78d6", "#1baf7a", "#eda100", "#008300"
VIOLET, RED, ORANGE, GRAY = "#4a3aa7", "#e34948", "#eb6834", "#5b5a56"
INK, MUTED = "#1a1a19", "#6f6e6a"


def _tint(hex_color, a=0.12):
    r = int(hex_color[1:3], 16) / 255; g = int(hex_color[3:5], 16) / 255; b = int(hex_color[5:7], 16) / 255
    return (r + (1 - r) * (1 - a), g + (1 - g) * (1 - a), b + (1 - b) * (1 - a))


def node(ax, x, y, w, h, title, color, subtitle="", detail="", dashed=False):
    """A rounded node: tinted fill, colored border, dark title, optional detail below."""
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.05",
                         linewidth=1.8, edgecolor=color, facecolor=_tint(color),
                         linestyle=("--" if dashed else "-"),
                         path_effects=[pe.withSimplePatchShadow(offset=(1.2, -1.6), alpha=0.10)])
    ax.add_patch(box)
    cx, cy = x + w / 2, y + h / 2
    title_y = (y + h * 0.60) if subtitle else cy
    ax.text(cx, title_y, title, ha="center", va="center",
            fontsize=10.2, fontweight="bold", color=INK, zorder=5)
    if subtitle:
        ax.text(cx, y + h * 0.22, subtitle, ha="center", va="center",
                fontsize=8.2, color=color, fontweight="bold", zorder=5)
    if detail:
        ax.text(cx, y - 0.11, detail, ha="center", va="top", fontsize=8, color=MUTED, zorder=5)


def arrow(ax, x1, y1, x2, y2, color=GRAY):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=15,
                                 lw=1.8, color=color, shrinkA=2, shrinkB=2, zorder=1))


# Figure 5 -- system pipeline
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(13, 3.8))
    ax.set_xlim(0, 13); ax.set_ylim(0, 3.8); ax.axis("off")
    w, h, y = 1.85, 1.3, 1.5
    gap = 0.28
    nodes = [
        ("Input\nimage", GRAY, "", "field photograph", False),
        ("YOLOv11n\ndetector", BLUE, "Stage 1", "mAP@0.5 = 0.860", False),
        ("Anatomical\ncrops", AQUA, "eyes / legs", "region extraction", False),
        ("EfficientNet-B2\nclassifier", YELLOW, "feasibility", "AUC = 0.506 (chance)", True),
        ("Groq VLM +\nLLM report", VIOLET, "deployed", "Spanish report", False),
        ("Flask\nweb app", GREEN, "ONNX", "role-based, multi-user", False),
    ]
    x = 0.35
    centers = []
    for title, color, sub, det, dash in nodes:
        node(ax, x, y, w, h, title, color, sub, det, dashed=dash)
        centers.append((x, x + w))
        x += w + gap
    for i in range(len(nodes) - 1):
        arrow(ax, centers[i][1], y + h / 2, centers[i + 1][0], y + h / 2)
    ax.text(6.5, 3.25, "AlpacaVision AI -- deployment pipeline", ha="center",
            fontsize=12.5, fontweight="bold", color=INK)
    fig.tight_layout(pad=0.4)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig1_pipeline.{ext}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("fig1_pipeline (Figure 5) regenerated")


# Figure 2 -- two-stage transfer learning
def fig_transfer():
    fig, ax = plt.subplots(figsize=(12, 4.0))
    ax.set_xlim(0, 12); ax.set_ylim(0, 4.0); ax.axis("off")
    w, h, y = 2.5, 1.5, 1.35
    gap = 0.55
    x = 0.4
    # ImageNet init
    node(ax, x, y, 2.0, h, "ImageNet\npre-training", GRAY, "1.3M images", "backbone init")
    a1 = x + 2.0; x = a1 + gap
    # Stage 1 (success)
    node(ax, x, y, w, h, "Stage 1\nHuman fundus", BLUE, "val F1 = 0.957",
         "937 anomaly / 959 normal")
    ax.text(x + w - 0.18, y + h - 0.16, "[OK]", ha="center", va="center",
            fontsize=16, fontweight="bold", color=GREEN, zorder=6)
    b1, b2 = x, x + w; x = b2 + gap
    # Stage 2 (chance)
    node(ax, x, y, w, h, "Stage 2\nAlpaca eye crops", YELLOW, "AUC-ROC = 0.506",
         "fine-tune (LR $10^{-4}$, focal $\\gamma$=1)", dashed=True)
    ax.text(x + w - 0.18, y + h - 0.16, "[X]", ha="center", va="center",
            fontsize=16, fontweight="bold", color=RED, zorder=6)
    c1 = x
    arrow(ax, a1, y + h / 2, b1, y + h / 2)
    arrow(ax, b2, y + h / 2, c1, y + h / 2)
    ax.text(6.0, 3.6, "Two-stage transfer learning (EfficientNet-B2)", ha="center",
            fontsize=12.5, fontweight="bold", color=INK)
    ax.text(6.0, 0.62, "The pipeline succeeds on real, expert-labelled human data ([OK]) but "
            "collapses to chance on alpaca data ([X]):\nthe failure is in the auto-generated "
            "labels and low crop resolution, not the model.",
            ha="center", va="top", fontsize=8.6, color=MUTED, style="italic")
    fig.tight_layout(pad=0.4)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig6_transfer_learning.{ext}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("fig6_transfer_learning (Figure 2) regenerated")


if __name__ == "__main__":
    fig_pipeline()
    fig_transfer()
