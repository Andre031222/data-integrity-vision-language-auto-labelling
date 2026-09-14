"""Generate fig8 (YOLOv11n detections: ground truth vs predictions) for the paper."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "outputs" / "figures" / "stage1_test_eval"
OUT = ROOT / "paper" / "figures"
DPI = 300
DARK = "#2C3E50"
FRAME = "#B8C0C8"

# Pick the first two available batches for a side-by-side GT-vs-prediction comparison.
batches = []
for i in range(3):
    lp, pp = SRC / f"val_batch{i}_labels.jpg", SRC / f"val_batch{i}_pred.jpg"
    if lp.exists() and pp.exists():
        batches.append((lp, pp))
    if len(batches) == 2:
        break
if not batches:
    raise SystemExit("No val_batch images in %s (run scripts/eval_detector.py first)" % SRC)

nrows = len(batches)
col_titles = ["Ground Truth", "YOLOv11n Predictions"]

fig, axes = plt.subplots(nrows, 2, figsize=(10, 5.15 * nrows),
                         gridspec_kw={"hspace": 0.06, "wspace": 0.04})
axes = np.atleast_2d(axes)

for r, (lp, pp) in enumerate(batches):
    for c, img in enumerate([lp, pp]):
        ax = axes[r, c]
        ax.imshow(np.array(Image.open(img)))
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_color(FRAME)
            s.set_linewidth(0.8)
        if r == 0:
            ax.set_title(col_titles[c], fontsize=13, fontweight="bold", pad=10,
                         color=DARK)
    axes[r, 0].set_ylabel(f"Batch {r + 1}", fontsize=12, fontweight="bold",
                          labelpad=8, color=DARK)

fig.suptitle("YOLOv11n Body Detection: Ground Truth vs. Predictions (scene-level test set)",
             fontsize=13.5, fontweight="bold", y=0.995)
fig.text(0.5, 0.045,
         "mAP@0.5 = 0.860  |  Precision = 0.913  |  Recall = 0.731   "
         "(n = 308 test images, 498 instances)",
         ha="center", fontsize=10, color=DARK)

plt.tight_layout(rect=[0, 0.06, 1, 0.97])
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"fig8_yolo_detections.{ext}", dpi=DPI, bbox_inches="tight")
plt.close(fig)
print("fig8_yolo_detections regenerated (2 batches, side-by-side GT vs predictions).")
