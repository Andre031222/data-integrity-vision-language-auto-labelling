"""Train the ocular-anomaly classifier under the group-aware protocol."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from src.data.group_split import build_group_split, assert_no_leakage  # noqa: E402

try:
    import timm
except ImportError:
    raise ImportError("pip install timm")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

TRAIN_TF = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
EVAL_TF = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class ListDataset(Dataset):
    """Dataset built from a list of (path, label_idx) pairs."""
    def __init__(self, items, transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=1.5):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=alpha, reduction="none")

    def forward(self, logits, labels):
        ce = self.ce(logits, labels)
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def make_balanced_sampler(labels, num_classes, gen):
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    weights = [1.0 / counts[l] for l in labels]
    return WeightedRandomSampler(weights, len(labels), replacement=True, generator=gen), counts


@torch.no_grad()
def predict_probs(model, loader, device, anomaly_idx, tta=False):
    model.eval()
    probs, labels = [], []
    for imgs, lbls in loader:
        imgs = imgs.to(device)
        p = torch.softmax(model(imgs), dim=1)
        if tta:
            p = p + torch.softmax(model(torch.flip(imgs, dims=[3])), dim=1)
            p = p / 2
        probs.extend(p[:, anomaly_idx].cpu().numpy())
        labels.extend(lbls.numpy())
    return np.array(labels), np.array(probs)


def train_loop(model, train_items, val_items, classes, device, epochs, batch_size,
               lr, patience, gamma, label):
    num_classes = len(classes)
    anomaly_idx = classes.index("anomaly") if "anomaly" in classes else 0
    gen = torch.Generator().manual_seed(SEED)

    train_labels = [l for _, l in train_items]
    sampler, counts = make_balanced_sampler(train_labels, num_classes, gen)
    train_loader = DataLoader(ListDataset(train_items, TRAIN_TF), batch_size=batch_size,
                              sampler=sampler, num_workers=0)
    val_loader = DataLoader(ListDataset(val_items, EVAL_TF), batch_size=batch_size,
                            shuffle=False, num_workers=0)

    alpha = torch.tensor([len(train_labels) / (num_classes * c) for c in counts],
                         dtype=torch.float).to(device)
    criterion = FocalLoss(alpha=alpha, gamma=gamma)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"\n{'='*64}\n  {label}\n  classes={classes} train={len(train_items)} val={len(val_items)}"
          f"\n  train dist={dict(zip(classes, counts.astype(int)))}\n{'='*64}")

    best_f1, no_improve, best_state, history = -1.0, 0, None, []
    for epoch in range(1, epochs + 1):
        model.train()
        correct = 0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, lbls)
            loss.backward()
            optimizer.step()
            correct += (logits.detach().argmax(1) == lbls).sum().item()
        scheduler.step()
        train_acc = correct / len(train_items)

        y_val, p_val = predict_probs(model, val_loader, device, anomaly_idx)
        pred = (p_val >= 0.5).astype(int)
        yb = (y_val == anomaly_idx).astype(int)
        tp = int(((pred == 1) & (yb == 1)).sum())
        fp = int(((pred == 1) & (yb == 0)).sum())
        fn = int(((pred == 0) & (yb == 1)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        val_acc = float((pred == yb).mean())
        history.append({"epoch": epoch, "train_acc": round(train_acc, 4),
                        "val_acc": round(val_acc, 4), "val_f1_anomaly": round(f1, 4),
                        "val_precision": round(prec, 4), "val_recall": round(rec, 4)})
        print(f"Ep {epoch:3d} | train={train_acc:.3f} | val_acc={val_acc:.3f} | "
              f"F1={f1:.3f} P={prec:.3f} R={rec:.3f}")

        if f1 > best_f1:
            best_f1, no_improve = f1, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping (sin mejora {patience} epochs)")
                break

    if best_state:
        model.load_state_dict(best_state)
    return model, history, best_f1


def tune_threshold(model, val_items, classes, device, anomaly_idx, tta):
    """Pick the threshold maximising macro F1 on the validation split."""
    val_loader = DataLoader(ListDataset(val_items, EVAL_TF), batch_size=32, shuffle=False)
    y, p = predict_probs(model, val_loader, device, anomaly_idx, tta=tta)
    yb = (y == anomaly_idx).astype(int)
    best_t, best_f1m = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        pred = (p >= t).astype(int)
        f1m = _f1_macro(yb, pred)
        if f1m > best_f1m:
            best_f1m, best_t = f1m, float(t)
    return best_t, best_f1m


def _f1_macro(yb, pred):
    f1s = []
    for c in (0, 1):
        tp = int(((pred == c) & (yb == c)).sum())
        fp = int(((pred == c) & (yb != c)).sum())
        fn = int(((pred != c) & (yb == c)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s))


def bootstrap_ci(y, p, fn, n_boot=2000, ci=0.95):
    rng = np.random.default_rng(SEED)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            vals.append(fn(y[idx], p[idx]))
        except Exception:
            pass
    a = (1 - ci) / 2
    return float(np.mean(vals)), float(np.percentile(vals, a * 100)), float(np.percentile(vals, (1 - a) * 100))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="efficientnet_b2")
    ap.add_argument("--skip_stage1", action="store_true")
    ap.add_argument("--epochs1", type=int, default=25)
    ap.add_argument("--epochs2", type=int, default=60)
    ap.add_argument("--batch1", type=int, default=32)
    ap.add_argument("--batch2", type=int, default=16)
    ap.add_argument("--gamma", type=float, default=1.5)
    ap.add_argument("--tta", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | arch: {args.arch}")

    out_dir = ROOT / "models" / "classifier" / "eyes_b2_honest"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = ROOT / "outputs" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    eye_disease_dir = ROOT / "data" / "raw" / "eye_disease"
    alpaca_dir = ROOT / "data" / "crops" / "eyes"

    model = timm.create_model(args.arch, pretrained=True, num_classes=2).to(device)

    history_all = []

    # ---- STAGE 1: fundus humano ----
    if not args.skip_stage1:
        fundus = ImageFolder(str(eye_disease_dir))
        classes1 = fundus.classes
        items1 = [(p, l) for p, l in fundus.imgs]
        rng = np.random.default_rng(SEED)
        rng.shuffle(items1)
        n_val1 = int(len(items1) * 0.2)
        val1, train1 = items1[:n_val1], items1[n_val1:]
        print("\n>>> STAGE 1: pre-entrenamiento en fundus humano")
        model, h1, f1_s1 = train_loop(model, train1, val1, classes1, device,
                                      args.epochs1, args.batch1, 5e-4, 8, args.gamma,
                                      f"Stage 1 -- fundus [{args.arch}]")
        torch.save(model.state_dict(), out_dir / "stage1_pretrain.pt")
        history_all += [{"stage": 1, **h} for h in h1]
    else:
        sp = out_dir / "stage1_pretrain.pt"
        if sp.exists():
            model.load_state_dict(torch.load(sp, map_location=device, weights_only=True))
            print("Stage 1 cargado de stage1_pretrain.pt")

    # ---- STAGE 2: alpaca con split honesto ----
    split = build_group_split(alpaca_dir, seed=SEED, test_frac=0.15, val_frac=0.15,
                              originals_only_eval=True)
    assert_no_leakage(split)
    classes = split["classes"]
    anomaly_idx = classes.index("anomaly") if "anomaly" in classes else 0
    print("\n>>> Split honesto (group-aware):")
    print(json.dumps(split["per_class"], indent=2))

    print("\n>>> STAGE 2: fine-tune en crops de alpaca (split honesto)")
    model, h2, f1_s2 = train_loop(model, split["train"], split["val"], classes, device,
                                  args.epochs2, args.batch2, 1e-4, 15, args.gamma,
                                  f"Stage 2 -- alpaca honesto [{args.arch}]")
    history_all += [{"stage": 2, **h} for h in h2]

    # ---- Threshold tuning en val ----
    thr, f1m_val = tune_threshold(model, split["val"], classes, device, anomaly_idx, args.tta)
    print(f"\nUmbral optimo (val, F1 macro={f1m_val:.3f}): {thr:.3f}")

    # ---- Evaluacion honesta en test ----
    test_loader = DataLoader(ListDataset(split["test"], EVAL_TF), batch_size=32, shuffle=False)
    y, p = predict_probs(model, test_loader, device, anomaly_idx, tta=args.tta)
    yb = (y == anomaly_idx).astype(int)
    pred = (p >= thr).astype(int)

    from sklearn.metrics import (roc_auc_score, average_precision_score,
                                 f1_score, precision_score, recall_score,
                                 accuracy_score, confusion_matrix)

    auc, auc_lo, auc_hi = bootstrap_ci(yb, p, roc_auc_score)
    ap_, _, _ = bootstrap_ci(yb, p, average_precision_score)
    f1a, f1a_lo, f1a_hi = bootstrap_ci(yb, p, lambda t, s: f1_score(t, (s >= thr).astype(int), zero_division=0))

    # McNemar vs mayoria
    from scipy.stats import chi2 as chi2_dist
    majority = np.full_like(y, np.bincount(y).argmax())
    pred_full = np.where(pred == 1, anomaly_idx, 1 - anomaly_idx)
    b = int(((pred_full != y) & (majority == y)).sum())
    c = int(((pred_full == y) & (majority != y)).sum())
    nbc = b + c
    if nbc:
        stat = (abs(b - c) - 1) ** 2 / nbc if nbc > 25 else (b - c) ** 2 / nbc
        pmc = float(1 - chi2_dist.cdf(stat, df=1))
    else:
        stat, pmc = None, None

    test_dist = {cls: int((y == classes.index(cls)).sum()) for cls in classes}
    metrics = {
        "task": "eyes", "arch": args.arch, "honest_split": True, "tta": args.tta,
        "threshold": thr, "n_test": int(len(y)), "test_distribution": test_dist,
        "accuracy": float(accuracy_score(yb, pred)),
        "auc_roc": auc, "auc_roc_ci_95": [auc_lo, auc_hi],
        "average_precision": ap_,
        "f1_anomaly": f1a, "f1_anomaly_ci_95": [f1a_lo, f1a_hi],
        "mcnemar_vs_majority": {"chi2": stat, "p_value": pmc},
        "per_class": {
            "anomaly": {
                "precision": float(precision_score(yb, pred, pos_label=1, zero_division=0)),
                "recall": float(recall_score(yb, pred, pos_label=1, zero_division=0)),
                "f1": float(f1_score(yb, pred, pos_label=1, zero_division=0)),
                "support": int((yb == 1).sum())},
            "normal": {
                "precision": float(precision_score(yb, pred, pos_label=0, zero_division=0)),
                "recall": float(recall_score(yb, pred, pos_label=0, zero_division=0)),
                "f1": float(f1_score(yb, pred, pos_label=0, zero_division=0)),
                "support": int((yb == 0).sum())},
        },
        "confusion_matrix": confusion_matrix(yb, pred).tolist(),
    }

    # ---- Guardar artefactos ----
    torch.save(model.state_dict(), out_dir / "best.pt")
    torch.save(model.state_dict(), out_dir / "last.pt")
    (out_dir / "arch.txt").write_text(args.arch)
    (out_dir / "class_names.json").write_text(json.dumps(classes))
    (out_dir / "threshold.json").write_text(json.dumps({"threshold": thr, "tta": args.tta}))
    (out_dir / "history.json").write_text(json.dumps(history_all, indent=2))
    manifest = {k: split[k] for k in ("seed", "test_frac", "val_frac", "originals_only_eval",
                                      "classes", "class_to_idx", "n_train", "n_val", "n_test", "per_class")}
    manifest["test_files"] = [pth for pth, _ in split["test"]]
    (out_dir / "split_manifest_honest.json").write_text(json.dumps(manifest, indent=2))
    (fig_dir / "classifier_eyes_honest_metrics.json").write_text(json.dumps(metrics, indent=2))

    report = "\n".join([
        f"AlpacaVision AI -- Clasificador de ojos (HONESTO, sin leakage) [{args.arch}]",
        f"Split group-aware | train={split['n_train']} val={split['n_val']} test={split['n_test']}",
        f"Test distribution (solo originales): {test_dist}",
        f"Umbral de decision: {thr:.3f} | TTA: {args.tta}", "",
        f"Accuracy:         {metrics['accuracy']:.4f}",
        f"AUC-ROC:          {auc:.4f}  95% CI [{auc_lo:.4f}, {auc_hi:.4f}]",
        f"Average Precision: {ap_:.4f}",
        f"F1 (anomaly):     {f1a:.4f}  95% CI [{f1a_lo:.4f}, {f1a_hi:.4f}]",
        f"Anomaly  P/R/F1:  {metrics['per_class']['anomaly']['precision']:.3f} / "
        f"{metrics['per_class']['anomaly']['recall']:.3f} / {metrics['per_class']['anomaly']['f1']:.3f}",
        f"Normal   P/R/F1:  {metrics['per_class']['normal']['precision']:.3f} / "
        f"{metrics['per_class']['normal']['recall']:.3f} / {metrics['per_class']['normal']['f1']:.3f}",
        f"McNemar vs majority: chi2={stat:.3f} p={pmc:.4f}" if stat else "McNemar: N/A",
        f"Confusion (filas=real [normal,anomaly]): {metrics['confusion_matrix']}",
    ])
    (fig_dir / "classifier_eyes_honest_report.txt").write_text(report, encoding="utf-8")

    print(f"\n{'='*64}\n{report}\n{'='*64}")
    print(f"Guardado en {out_dir} y {fig_dir}/classifier_eyes_honest_*")


if __name__ == "__main__":
    main()
