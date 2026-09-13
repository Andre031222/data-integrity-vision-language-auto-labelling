"""AlpacaVision AI -- Publication-quality evaluation metrics."""

import json
from pathlib import Path

import numpy as np


def evaluate_classifier(y_true: list, y_pred: list, y_proba: list,
                        class_names: list) -> dict:
    """Accuracy, macro F1, per-class P/R/F1, AUC-ROC."""
    from sklearn.metrics import (
        accuracy_score, classification_report, f1_score, roc_auc_score,
    )

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_proba = np.array(y_proba)

    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0,
    )

    auc = None
    try:
        if len(class_names) == 2:
            auc = roc_auc_score(y_true, y_proba[:, 1])
        else:
            auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
    except Exception:
        pass

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "auc_roc": float(auc) if auc is not None else None,
        "per_class": {name: report[name] for name in class_names if name in report},
    }


def bootstrap_ci(y_true: np.ndarray, y_score: np.ndarray, metric_fn,
                 n_boot: int = 1000, ci: float = 0.95,
                 seed: int = 42) -> tuple[float, float, float]:
    """Percentile bootstrap confidence interval."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            scores.append(metric_fn(y_true[idx], y_score[idx]))
        except Exception:
            pass
    arr = np.array(scores)
    alpha = (1.0 - ci) / 2.0
    return (
        float(np.mean(arr)),
        float(np.percentile(arr, alpha * 100)),
        float(np.percentile(arr, (1.0 - alpha) * 100)),
    )


def mcnemar_test(y_true: np.ndarray, y_pred_a: np.ndarray,
                 y_pred_b: np.ndarray) -> tuple[float | None, float | None]:
    """McNemar test: model A vs model B (or majority baseline)."""
    from scipy.stats import chi2 as chi2_dist

    y_true = np.array(y_true)
    y_pred_a = np.array(y_pred_a)
    y_pred_b = np.array(y_pred_b)

    b = int(((y_pred_a != y_true) & (y_pred_b == y_true)).sum())
    c = int(((y_pred_a == y_true) & (y_pred_b != y_true)).sum())
    n = b + c
    if n == 0:
        return None, None

    stat = (b - c) ** 2 / n if n <= 25 else (abs(b - c) - 1.0) ** 2 / n
    return float(stat), float(1.0 - chi2_dist.cdf(stat, df=1))


def print_classification_report(metrics: dict) -> None:
    print(f"\n{'='*52}")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  F1 Macro : {metrics['f1_macro']:.4f}")
    if metrics.get("auc_roc") is not None:
        print(f"  AUC-ROC  : {metrics['auc_roc']:.4f}")
    print("\n  Per class:")
    for cls, vals in metrics.get("per_class", {}).items():
        print(
            f"    {cls:25s}  P={vals['precision']:.3f} "
            f"R={vals['recall']:.3f} F1={vals['f1-score']:.3f} "
            f"n={int(vals.get('support', 0))}"
        )
    print("=" * 52)


def save_metrics(metrics: dict, output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"Metrics saved: {output_path}")
