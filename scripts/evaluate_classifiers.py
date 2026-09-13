"""AlpacaVision AI -- Classifier evaluation with publication-quality statistics."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

SEED = 42
TEST_FRAC = 0.15
VAL_FRAC = 0.15


def _build_split(dataset, manifest_path: Path, reset: bool) -> dict:
    if manifest_path.exists() and not reset:
        return json.loads(manifest_path.read_text())

    rng = np.random.default_rng(SEED)
    all_labels = np.array([dataset.imgs[i][1] for i in range(len(dataset))])
    test_idx, val_idx, train_idx = [], [], []

    for cls in np.unique(all_labels):
        cls_indices = np.where(all_labels == cls)[0]
        rng.shuffle(cls_indices)
        n = len(cls_indices)
        n_test = max(1, int(n * TEST_FRAC))
        n_val = max(1, int(n * VAL_FRAC))
        test_idx.extend(cls_indices[:n_test].tolist())
        val_idx.extend(cls_indices[n_test:n_test + n_val].tolist())
        train_idx.extend(cls_indices[n_test + n_val:].tolist())

    manifest = {
        "seed": SEED, "test_frac": TEST_FRAC, "val_frac": VAL_FRAC,
        "n_train": len(train_idx), "n_val": len(val_idx), "n_test": len(test_idx),
        "train": train_idx, "val": val_idx, "test": test_idx,
        "test_files": [dataset.imgs[i][0] for i in test_idx],
        "class_to_idx": dataset.class_to_idx,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"  Split saved: {manifest_path}")
    return manifest


def _bootstrap_ci(y_true, y_score, metric_fn, n_boot=1000, ci=0.95):
    rng = np.random.default_rng(SEED)
    n = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            scores.append(metric_fn(y_true[idx], y_score[idx]))
        except Exception:
            pass
    arr = np.array(scores)
    alpha = (1 - ci) / 2
    return float(np.mean(arr)), float(np.percentile(arr, alpha * 100)), float(np.percentile(arr, (1 - alpha) * 100))


def _mcnemar(y_true, y_pred_model, y_pred_baseline):
    from scipy.stats import chi2 as chi2_dist
    b = int(((y_pred_model != y_true) & (y_pred_baseline == y_true)).sum())
    c = int(((y_pred_model == y_true) & (y_pred_baseline != y_true)).sum())
    n = b + c
    if n == 0:
        return None, None
    stat = (abs(b - c) - 1) ** 2 / n if n > 25 else (b - c) ** 2 / n
    return float(stat), float(1 - chi2_dist.cdf(stat, df=1))


def evaluate_task(task: str, reset_split: bool):
    import timm
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        classification_report, confusion_matrix, ConfusionMatrixDisplay,
        roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
        accuracy_score, f1_score, precision_score, recall_score,
    )

    model_dir = ROOT / "models" / "classifier"
    candidates = [model_dir / f"{task}_b2", model_dir / task]
    model_dir_task = next((d for d in candidates if (d / "best.pt").exists()), None)
    if model_dir_task is None:
        print(f"[{task}] Model not found. Train first with train_two_stage.py")
        return None

    model_path = model_dir_task / "best.pt"
    arch = (model_dir_task / "arch.txt").read_text().strip() if (model_dir_task / "arch.txt").exists() else "efficientnet_b2"
    out_dir = ROOT / "outputs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    base_ds = datasets.ImageFolder(str(ROOT / "data" / "crops" / task), transform=VAL_TRANSFORM)
    class_names = base_ds.classes
    manifest = _build_split(base_ds, model_dir_task / "split_manifest.json", reset_split)
    test_idx = manifest["test"]
    n_test = len(test_idx)

    test_dist = {cls: sum(1 for i in test_idx if base_ds.imgs[i][1] == base_ds.class_to_idx[cls]) for cls in class_names}
    print(f"\n[{task}] arch={arch} | train={manifest['n_train']} val={manifest['n_val']} test={n_test}")
    print(f"  Test distribution: {test_dist}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(class_names)
    model = timm.create_model(arch, pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device).eval()

    test_ds = Subset(base_ds, test_idx)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_proba = np.array(all_probs)
    anomaly_idx = base_ds.class_to_idx.get("anomaly", 0)
    y_score = y_proba[:, anomaly_idx]
    y_binary = (y_true == anomaly_idx).astype(int)

    report_str = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(f"\n{report_str}")

    auc, auc_lo, auc_hi = _bootstrap_ci(y_binary, y_score, roc_auc_score)
    print(f"  AUC-ROC: {auc:.4f}  95% CI [{auc_lo:.4f}, {auc_hi:.4f}]")

    f1, f1_lo, f1_hi = _bootstrap_ci(
        y_binary, y_score,
        lambda yt, ys: f1_score(yt, (ys >= 0.5).astype(int), pos_label=1, zero_division=0),
    )
    print(f"  F1 (anomaly): {f1:.4f}  95% CI [{f1_lo:.4f}, {f1_hi:.4f}]")

    majority = np.full_like(y_true, np.bincount(y_true).argmax())
    chi2_stat, p_val = _mcnemar(y_true, y_pred, majority)
    if chi2_stat is not None:
        print(f"  McNemar vs majority: chi2={chi2_stat:.3f}, p={p_val:.4f}")

    fpr, tpr, _ = roc_curve(y_binary, y_score)
    precision_curve, recall_curve, _ = precision_recall_curve(y_binary, y_score)
    ap = average_precision_score(y_binary, y_score)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(fpr, tpr, color="#3b82f6", lw=2,
                 label=f"AUC={auc:.3f} [{auc_lo:.3f},{auc_hi:.3f}]")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].fill_between(fpr, tpr, alpha=0.1, color="#3b82f6")
    axes[0].set(xlabel="False Positive Rate", ylabel="True Positive Rate",
                title=f"ROC Curve -- {task} (n={n_test})")
    axes[0].legend(loc="lower right", fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(recall_curve, precision_curve, color="#22c55e", lw=2, label=f"AP={ap:.3f}")
    axes[1].axhline(y_binary.mean(), color="gray", ls="--", lw=1, label=f"Baseline ({y_binary.mean():.2f})")
    axes[1].set(xlabel="Recall", ylabel="Precision", title=f"Precision-Recall -- {task}")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    roc_path = out_dir / f"classifier_{task}_roc_pr.png"
    plt.savefig(roc_path, dpi=150, bbox_inches="tight")
    plt.close()

    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, data, fmt, title in [
        (axes[0], cm, "d", f"Confusion Matrix (n={n_test})"),
        (axes[1], cm_norm, ".2f", "Normalized Confusion Matrix"),
    ]:
        ConfusionMatrixDisplay(confusion_matrix=data, display_labels=class_names).plot(
            ax=ax, colorbar=False, cmap="Blues", values_format=fmt
        )
        ax.set_title(f"{title} -- {task}", pad=10)
    plt.tight_layout()
    cm_path = out_dir / f"classifier_{task}_confusion.png"
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()

    history_path = model_dir_task / "history.json"
    if history_path.exists():
        hist = json.loads(history_path.read_text())
        epochs = [h["epoch"] for h in hist]
        best_ep = max(hist, key=lambda h: h["val_f1_anomaly"])
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(epochs, [h["train_acc"] for h in hist], label="Train Acc", color="#22c55e")
        ax.plot(epochs, [h["val_acc"] for h in hist], label="Val Acc", color="#60a5fa")
        ax.plot(epochs, [h["val_f1_anomaly"] for h in hist],
                label="Val F1 (anomaly)", color="#f59e0b", ls="--")
        ax.axvline(best_ep["epoch"], color="red", ls=":", lw=1, alpha=0.7,
                   label=f"Best F1={best_ep['val_f1_anomaly']:.3f} (ep{best_ep['epoch']})")
        ax.set(xlabel="Epoch", ylabel="Score", title=f"Training History -- {task} ({arch})")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        curves_path = out_dir / f"classifier_{task}_training_curves.png"
        plt.savefig(curves_path, dpi=150, bbox_inches="tight")
        plt.close()

    metrics_dict = {
        "task": task, "arch": arch, "model_path": str(model_path),
        "n_test": n_test, "test_distribution": test_dist,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_anomaly": f1, "f1_anomaly_ci_95": [f1_lo, f1_hi],
        "auc_roc": auc, "auc_roc_ci_95": [auc_lo, auc_hi],
        "average_precision": float(ap),
        "mcnemar_vs_majority": {"chi2": chi2_stat, "p_value": p_val},
        "per_class": {
            cls: {
                "precision": float(precision_score(y_true, y_pred, pos_label=i, average="binary", zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, pos_label=i, average="binary", zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, pos_label=i, average="binary", zero_division=0)),
                "support": int((y_true == i).sum()),
            }
            for i, cls in enumerate(class_names)
        },
    }

    (out_dir / f"classifier_{task}_metrics.json").write_text(json.dumps(metrics_dict, indent=2))

    report_txt = "\n".join([
        f"AlpacaVision AI -- EfficientNet ({arch}) Evaluation",
        f"Task: {task} | Split: {manifest['n_train']}/{manifest['n_val']}/{n_test}",
        f"Test distribution: {test_dist}", "",
        report_str,
        f"AUC-ROC:          {auc:.4f}  95% CI [{auc_lo:.4f}, {auc_hi:.4f}]",
        f"Average Precision: {ap:.4f}",
        f"F1 (anomaly):     {f1:.4f}  95% CI [{f1_lo:.4f}, {f1_hi:.4f}]", "",
        f"McNemar vs majority: chi2={chi2_stat:.3f} p={p_val:.4f}" if chi2_stat else "McNemar: N/A",
    ])
    (out_dir / f"classifier_{task}_eval_report.txt").write_text(report_txt, encoding="utf-8")

    print(f"  Saved: {roc_path.name}, {cm_path.name}")
    print(f"  Saved: classifier_{task}_metrics.json, classifier_{task}_eval_report.txt")
    return metrics_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["eyes", "legs", "both"], default="both")
    parser.add_argument("--reset-split", action="store_true")
    args = parser.parse_args()

    missing = [p for p in ["timm", "sklearn", "matplotlib", "scipy"]
               if __import__("importlib").util.find_spec(p) is None]
    if missing:
        print(f"Missing: {missing}. Run: pip install {' '.join(missing)}")
        sys.exit(1)

    tasks = ["eyes", "legs"] if args.task == "both" else [args.task]
    results = {t: r for t in tasks if (r := evaluate_task(t, args.reset_split))}

    if results:
        summary = ROOT / "outputs" / "figures" / "classifier_summary.json"
        summary.write_text(json.dumps(results, indent=2))
        print(f"\nSummary: {summary}")


if __name__ == "__main__":
    main()
