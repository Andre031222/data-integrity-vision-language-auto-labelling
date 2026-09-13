"""Regenerate honest classifier figures (fig4, fig7, fig9) from the raw saved predictions --."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, precision_recall_curve, roc_auc_score,
                             average_precision_score, confusion_matrix, f1_score,
                             accuracy_score)

ROOT = Path(__file__).resolve().parent.parent
OUT_FIG = ROOT / "paper" / "figures"
OUTPUTS = ROOT / "outputs" / "figures"
DPI = 300
plt.rcParams.update({"font.size": 12, "savefig.dpi": DPI})

pred = json.loads((OUTPUTS / "classifier_eyes_honest_predictions.json").read_text())
y = np.array(pred["y_true_anomaly"]); p = np.array(pred["p_anomaly"])
thr = pred["threshold"]

auc = roc_auc_score(y, p)
ap = average_precision_score(y, p)
yp = (p >= thr).astype(int)
cm = confusion_matrix(y, yp, labels=[0, 1])          # rows=true [normal,anomaly]
f1a = f1_score(y, yp, pos_label=1, zero_division=0)
acc = accuracy_score(y, yp)

# bootstrap 95% CI for AUC (fixed seed, deterministic)
rng = np.random.default_rng(42)
boot = []
for _ in range(1000):
    idx = rng.integers(0, len(y), len(y))
    if len(np.unique(y[idx])) < 2:
        continue
    boot.append(roc_auc_score(y[idx], p[idx]))
ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
prevalence = y.mean()

print(f"CANONICAL  AUC={auc:.4f}  AP={ap:.4f}  F1a={f1a:.4f}  Acc={acc:.4f}  "
      f"CI[{ci_lo:.3f},{ci_hi:.3f}]  CM={cm.tolist()}")

BLUE, GREEN, ORANGE, TEAL, RED, GREY = ("#2166AC", "#1B9E77", "#E6820E",
                                        "#17A2A2", "#D62728", "#7F7F7F")

# ---- fig4: honest test-set metrics bar chart --------------------------------
def fig4():
    labels = ["AUC-ROC", "Avg. Precision", "F1 (anomaly)", "Accuracy"]
    vals = [auc, ap, f1a, acc]
    colors = [BLUE, GREEN, ORANGE, TEAL]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    yb = np.arange(len(labels))[::-1]
    ax.barh(yb, vals, color=colors, height=0.62)
    ax.axvline(0.5, color=RED, ls="--", lw=1.6)
    ax.text(0.5, len(labels) - 0.35, "chance\n(AUC=0.5)", color=RED,
            ha="center", va="top", fontsize=10)
    for yi, v in zip(yb, vals):
        ax.text(v + 0.015, yi, f"{v:.3f}", va="center", fontweight="bold", fontsize=13)
    ax.set_yticks(yb); ax.set_yticklabels(labels)
    ax.set_xlim(0, 1.12); ax.set_xlabel("Score")
    ax.set_title("EfficientNet-B2 Ocular Classifier -- Honest Test Set\n"
                 f"(n={len(y)}, group-aware split; chance-level performance)",
                 fontweight="bold")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_FIG / f"fig4_classifier.{ext}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

# ---- fig7: ROC + PR ---------------------------------------------------------
def fig7():
    fpr, tpr, _ = roc_curve(y, p)
    prec, rec, _ = precision_recall_curve(y, p)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.2))
    a1.plot(fpr, tpr, color=RED, lw=2.2,
            label=f"AUC = {auc:.3f} (95% CI [{ci_lo:.3f}, {ci_hi:.3f}])")
    a1.fill_between(fpr, tpr, alpha=0.12, color=RED)
    a1.plot([0, 1], [0, 1], color=GREY, ls="--", lw=1.4, label="Chance (AUC = 0.500)")
    a1.set_xlabel("False Positive Rate"); a1.set_ylabel("True Positive Rate")
    a1.set_title("ROC Curve -- Ocular Anomaly Classifier (honest test)", fontweight="bold")
    a1.legend(loc="lower right", fontsize=11, framealpha=0.95)
    a1.set_xlim(0, 1); a1.set_ylim(0, 1.02)
    a2.plot(rec, prec, color=BLUE, lw=2.2, label=f"AP = {ap:.3f}")
    a2.fill_between(rec, prec, alpha=0.12, color=BLUE)
    a2.axhline(prevalence, color=GREY, ls="--", lw=1.4,
               label=f"Baseline (prevalence) = {prevalence:.2f}")
    a2.set_xlabel("Recall"); a2.set_ylabel("Precision")
    a2.set_title("Precision-Recall Curve", fontweight="bold")
    a2.legend(loc="upper right", fontsize=11, framealpha=0.95)
    a2.set_xlim(0, 1); a2.set_ylim(0, 1.02)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_FIG / f"fig7_roc_pr.{ext}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

# ---- fig9: confusion matrices ----------------------------------------------
def fig9():
    cmn = cm / cm.sum(axis=1, keepdims=True)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.8))
    fig.suptitle("EfficientNet-B2 Ocular Anomaly Classifier -- Confusion Matrix (honest test)",
                 fontweight="bold", fontsize=14)
    for ax, mat, title, fmt in ((a1, cm, f"Counts (n={len(y)})", "d"),
                                (a2, cmn, "Row-normalized", ".2f")):
        im = ax.imshow(mat, cmap="Blues", vmin=0)
        for i in range(2):
            for j in range(2):
                val = mat[i, j]
                txt = f"{val:d}" if fmt == "d" else f"{val:.2f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=17,
                        fontweight="bold",
                        color="white" if val > mat.max() * 0.55 else "#1a1a1a")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["normal", "anomaly"]); ax.set_yticklabels(["normal", "anomaly"])
        ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
        ax.set_title(title, fontweight="bold")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_FIG / f"fig9_confusion_matrices.{ext}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

fig4(); fig7(); fig9()

# ---- consolidated canonical metrics JSON -----------------------------------
out = {
    "task": "eyes", "arch": pred.get("arch", "efficientnet_b2"),
    "honest_split": True, "tta": pred.get("tta", True), "threshold": thr,
    "n_test": int(len(y)),
    "test_distribution": {"anomaly": int(y.sum()), "normal": int((1 - y).sum())},
    "accuracy": float(acc), "auc_roc": float(auc),
    "auc_roc_ci_95": [float(ci_lo), float(ci_hi)],
    "average_precision": float(ap), "f1_anomaly": float(f1a),
    "confusion_matrix": cm.tolist(),
}
(OUTPUTS / "classifier_eyes_honest_metrics.json").write_text(json.dumps(out, indent=2))
print("figures + canonical metrics.json regenerated.")
