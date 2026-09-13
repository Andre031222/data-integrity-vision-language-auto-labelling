"""AlpacaVision AI -- Two-stage transfer learning for ocular anomaly classifier."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

try:
    import timm
except ImportError:
    raise ImportError("pip install timm")

ARCH = "efficientnet_b2"
SEED = 42

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class _SubsetWithTransform(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        from PIL import Image
        path, label = self.subset.dataset.imgs[self.subset.indices[idx]]
        return self.transform(Image.open(path).convert("RGB")), label


class _FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=alpha, reduction="none")

    def forward(self, logits, labels):
        ce = self.ce(logits, labels)
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def run_training(data_dir: Path, model: nn.Module, device: torch.device,
                 epochs: int, batch_size: int, lr: float, patience: int,
                 val_split: float = 0.20, label: str = "", gamma: float = 2.0) -> tuple:
    base_ds = ImageFolder(str(data_dir))
    num_classes = len(base_ds.classes)
    anomaly_idx = base_ds.class_to_idx.get("anomaly", 0)

    gen = torch.Generator().manual_seed(SEED)
    idx = torch.randperm(len(base_ds), generator=gen).tolist()
    n_val = int(len(base_ds) * val_split)
    n_train = len(base_ds) - n_val
    train_idx, val_idx = idx[:n_train], idx[n_train:]

    train_ds = _SubsetWithTransform(Subset(base_ds, train_idx), TRAIN_TRANSFORM)
    val_ds = _SubsetWithTransform(Subset(base_ds, val_idx), VAL_TRANSFORM)

    train_labels = [base_ds.imgs[i][1] for i in train_idx]
    class_counts = np.bincount(train_labels, minlength=num_classes).astype(float)
    sampler = WeightedRandomSampler(
        [1.0 / class_counts[l] for l in train_labels],
        len(train_labels), replacement=True, generator=gen,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    alpha = torch.tensor([len(train_labels) / (num_classes * c) for c in class_counts], dtype=torch.float).to(device)
    criterion = _FocalLoss(alpha=alpha, gamma=gamma)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  Classes: {base_ds.classes} | Train: {n_train} | Val: {n_val}")
    print(f"  Distribution: { {base_ds.classes[i]: int(class_counts[i]) for i in range(num_classes)} }")
    print(f"{'='*60}")

    best_f1, no_improve = 0.0, 0
    best_state = None
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        correct = 0
        for imgs, labels in tqdm(train_loader, desc=f"Ep {epoch}/{epochs}", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            correct += (logits.detach().argmax(1) == labels).sum().item()
        scheduler.step()
        train_acc = correct / n_train

        model.eval()
        preds_all, labels_all = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                preds_all.extend(model(imgs.to(device)).argmax(1).cpu().numpy())
                labels_all.extend(labels.numpy())

        val_acc = sum(p == l for p, l in zip(preds_all, labels_all)) / n_val
        tp = sum(1 for p, l in zip(preds_all, labels_all) if p == anomaly_idx and l == anomaly_idx)
        fp = sum(1 for p, l in zip(preds_all, labels_all) if p == anomaly_idx and l != anomaly_idx)
        fn = sum(1 for p, l in zip(preds_all, labels_all) if p != anomaly_idx and l == anomaly_idx)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        history.append({
            "epoch": epoch, "train_acc": round(train_acc, 4), "val_acc": round(val_acc, 4),
            "val_f1_anomaly": round(f1, 4), "val_precision": round(prec, 4), "val_recall": round(rec, 4),
        })
        print(f"Ep {epoch:3d} | train={train_acc:.3f} | val={val_acc:.3f} | F1={f1:.3f} | P={prec:.3f} R={rec:.3f}")

        if f1 > best_f1:
            best_f1 = f1
            no_improve = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping: no F1 improvement for {patience} epochs")
                break

    if best_state:
        model.load_state_dict(best_state)
    return model, history, best_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_stage1", action="store_true")
    parser.add_argument("--device", default="0")
    parser.add_argument("--gamma", type=float, default=2.0,
                        help="Focal loss gamma (default 2.0; try 1.0 to improve normal-class recall)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ROOT / "models" / "classifier" / "eyes_b2"
    out_dir.mkdir(parents=True, exist_ok=True)

    eye_disease_dir = ROOT / "data" / "raw" / "eye_disease"
    alpaca_dir = ROOT / "data" / "crops" / "eyes"

    if not eye_disease_dir.exists() or not any(eye_disease_dir.rglob("*.jpg")):
        print("ERROR: data/raw/eye_disease/ not found.")
        sys.exit(1)
    if not alpaca_dir.exists():
        print("ERROR: data/crops/eyes/ not found.")
        sys.exit(1)

    model = timm.create_model(ARCH, pretrained=True, num_classes=2)
    model = model.to(device)

    history_all = []
    f1_s1 = None

    if not args.skip_stage1:
        print("\n>>> STAGE 1: Pre-training on human fundus eye disease dataset")
        model, hist1, f1_s1 = run_training(
            data_dir=eye_disease_dir, model=model, device=device,
            epochs=30, batch_size=32, lr=5e-4, patience=8,
            label=f"Stage 1 -- Eye Disease (fundus, 937 anomaly + 959 normal) [{ARCH}]",
            gamma=args.gamma,
        )
        torch.save(model.state_dict(), out_dir / "stage1_pretrain.pt")
        history_all += [{"stage": 1, **h} for h in hist1]
        print(f"  Stage 1 complete. Best F1: {f1_s1:.4f}")
    else:
        stage1_path = out_dir / "stage1_pretrain.pt"
        if stage1_path.exists():
            model.load_state_dict(torch.load(stage1_path, map_location=device, weights_only=True))
            print("  Stage 1 loaded from stage1_pretrain.pt")

    print("\n>>> STAGE 2: Fine-tuning on alpaca eye crops")
    model, hist2, f1_s2 = run_training(
        data_dir=alpaca_dir, model=model, device=device,
        epochs=60, batch_size=16, lr=1e-4, patience=15,
        label=f"Stage 2 -- Alpaca crops fine-tune [{ARCH}] gamma={args.gamma}",
        gamma=args.gamma,
    )

    torch.save(model.state_dict(), out_dir / "best.pt")
    torch.save(model.state_dict(), out_dir / "last.pt")

    base_ds = ImageFolder(str(alpaca_dir))
    (out_dir / "class_names.json").write_text(json.dumps(base_ds.classes))
    (out_dir / "arch.txt").write_text(ARCH)

    history_all += [{"stage": 2, **h} for h in hist2]
    (out_dir / "history.json").write_text(json.dumps(history_all, indent=2))

    print(f"\n{'='*60}")
    print(f"  TWO-STAGE TRAINING COMPLETE")
    print(f"  Stage 1 (eye disease pretrain): F1 = {f1_s1:.4f}" if f1_s1 else "  Stage 1: skipped")
    print(f"  Stage 2 (alpaca fine-tune):     F1 = {f1_s2:.4f}")
    print(f"  Saved to: {out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
