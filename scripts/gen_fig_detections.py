#!/usr/bin/env python3
"""Qualitative detections of the released YOLOv11n on the released test split."""
import argparse, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyse_contamination as A

GT, PRED, MISS = "#009E73", "#E69F00", "#D55E00"
PANELS = [
    (37, "Three animals behind a fence"),
    (5, "Single animal"),
    (4, "Distant animal missed"),
    (17, "Resting group, overlapping"),
    (14, "Calf and adult in one box"),
    (70, "Sheep boxed as alpaca by the source"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="data/v3.1/s6photoid")
    ap.add_argument("--weights", default="outputs/training_runs_v31/s6photoid_seed0/weights/best.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", default="paper/figures/fig_detections")
    a = ap.parse_args()

    plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 7})
    test = A.list_split(a.split, "test")
    preds = A.predict(a.weights, a.split, "test", test, 640, "cpu")

    fig, axes = plt.subplots(2, 3, figsize=(181 / 25.4, 128 / 25.4))
    for k, (ax, (i, title)) in enumerate(zip(axes.ravel(), PANELS)):
        img = np.asarray(Image.open(test[i]["path"]).convert("RGB"))
        h, w = img.shape[:2]
        ax.imshow(img)
        p = preds[i][preds[i][:, 4] >= a.conf]
        tp = A.match(p, test[i]["gt"])
        hit = np.zeros(len(test[i]["gt"]), bool)
        if len(p) and len(test[i]["gt"]):
            iou = A.iou(test[i]["gt"], p[:, :4])
            hit = (iou >= 0.5).any(1)
        for b, found in zip(test[i]["gt"], hit):
            ax.add_patch(Rectangle((b[0] * w, b[1] * h), (b[2] - b[0]) * w, (b[3] - b[1]) * h,
                                   fill=False, ec=GT if found else MISS, lw=1.3, ls="--"))
        for b, ok in zip(p, tp):
            ax.add_patch(Rectangle((b[0] * w, b[1] * h), (b[2] - b[0]) * w, (b[3] - b[1]) * h,
                                   fill=False, ec=PRED, lw=1.3))
            ax.text(b[0] * w + 3, b[1] * h + 3, f"{b[4]:.2f}", color="black", fontsize=7,
                    fontweight="bold", va="top", ha="left",
                    bbox=dict(boxstyle="square,pad=0.15", fc=PRED, ec="none", alpha=0.9))
        ax.set_title(f"({chr(97 + k)}) {title}", fontsize=7, loc="left", pad=2)
        ax.set_axis_off()
    handles = [plt.Line2D([], [], color=GT, ls="--", lw=1.3, label="Ground truth, detected"),
               plt.Line2D([], [], color=MISS, ls="--", lw=1.3, label="Ground truth, missed"),
               plt.Line2D([], [], color=PRED, lw=1.3, label=f"Prediction (confidence $\\geq$ {a.conf})")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=7,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.04, 1, 1), w_pad=0.6, h_pad=1.4)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), dpi=600)
    fig.savefig(out.with_suffix(".png"), dpi=600)
    fig.savefig(out.with_suffix(".tiff"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    print(f"wrote {out}.pdf/.png/.tiff")


if __name__ == "__main__":
    main()
