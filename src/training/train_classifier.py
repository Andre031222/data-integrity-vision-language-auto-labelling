"""AlpacaVision AI -- Entrenamiento del Clasificador EfficientNet-B0 (timm)."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms
from tqdm import tqdm

try:
    import timm
except ImportError:
    raise ImportError("Instalar: pip install timm")


TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class SubsetWithTransform(torch.utils.data.Dataset):
    """Subset applying its own transform, independent of the base dataset."""
    def __init__(self, subset: Subset, transform):
        self.subset    = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset.dataset.imgs[self.subset.indices[idx]]
        from PIL import Image
        img = Image.open(img).convert("RGB")
        return self.transform(img), label


def train_classifier(
    task: str,
    data_dir: str,
    output_dir: str,
    epochs: int = 80,
    batch_size: int = 16,
    lr: float = 3e-4,
    patience: int = 15,
    val_split: float = 0.20,
    seed: int = 42,
    device: str = "cpu",
    arch: str = "efficientnet_b0",
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # base dataset, used only to obtain indices and class names
    base_ds = datasets.ImageFolder(data_dir)
    num_classes = len(base_ds.classes)
    n_val   = int(len(base_ds) * val_split)
    n_train = len(base_ds) - n_val

    generator = torch.Generator().manual_seed(seed)
    indices   = torch.randperm(len(base_ds), generator=generator).tolist()
    train_idx = indices[:n_train]
    val_idx   = indices[n_train:]

    train_subset = Subset(base_ds, train_idx)
    val_subset   = Subset(base_ds, val_idx)

    train_ds = SubsetWithTransform(train_subset, TRAIN_TRANSFORM)
    val_ds   = SubsetWithTransform(val_subset,   VAL_TRANSFORM)

    print(f"Tarea: {task} | Clases: {base_ds.classes} | Train: {n_train} | Val: {n_val}")

    # class counts in train, for balancing
    train_labels = [base_ds.imgs[i][1] for i in train_idx]
    class_counts = np.bincount(train_labels, minlength=num_classes).astype(float)
    print(f"Distribucion train: { {base_ds.classes[i]: int(class_counts[i]) for i in range(num_classes)} }")

    # WeightedRandomSampler balances the classes within each batch
    sample_weights = [1.0 / class_counts[lbl] for lbl in train_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
        generator=generator,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,   num_workers=0)

    # class weights in the loss, a second guard against imbalance
    class_weights = torch.tensor(
        [len(train_labels) / (num_classes * c) for c in class_counts],
        dtype=torch.float
    ).to(device)

    # Modelo
    model = timm.create_model(arch, pretrained=True, num_classes=num_classes)
    model = model.to(device)
    print(f"Arquitectura: {arch}")

    # Focal Loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    class FocalLoss(nn.Module):
        def __init__(self, alpha, gamma=2.0):
            super().__init__()
            self.alpha = alpha      # weight per class
            self.gamma = gamma      # factor de enfoque (2.0 estandar)
            self.ce    = nn.CrossEntropyLoss(weight=alpha, reduction="none")

        def forward(self, logits, labels):
            ce_loss = self.ce(logits, labels)
            pt      = torch.exp(-ce_loss)
            return ((1 - pt) ** self.gamma * ce_loss).mean()

    criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_f1  = 0.0
    best_val_acc = 0.0
    no_improve   = 0
    history      = []

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        train_correct = 0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_correct += (logits.argmax(1) == labels).sum().item()

        scheduler.step()
        train_acc = train_correct / n_train

        # --- Val ---
        model.eval()
        val_correct = 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                preds  = logits.argmax(1)
                val_correct += (preds == labels).sum().item()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_acc = val_correct / n_val

        # F1 of the anomaly class, index 0 under ImageFolder's alphabetical ordering
        anomaly_idx = base_ds.class_to_idx.get("anomaly", 0)
        tp = sum(1 for p, l in zip(all_preds, all_labels) if p == anomaly_idx and l == anomaly_idx)
        fp = sum(1 for p, l in zip(all_preds, all_labels) if p == anomaly_idx and l != anomaly_idx)
        fn = sum(1 for p, l in zip(all_preds, all_labels) if p != anomaly_idx and l == anomaly_idx)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        history.append({
            "epoch": epoch, "train_acc": round(train_acc, 4),
            "val_acc": round(val_acc, 4), "val_f1_anomaly": round(f1, 4),
            "val_precision": round(prec, 4), "val_recall": round(rec, 4),
        })
        print(f"Epoch {epoch:3d} | train={train_acc:.3f} | val_acc={val_acc:.3f} "
              f"| F1_anomaly={f1:.3f} | P={prec:.3f} R={rec:.3f}")

        # keep the best checkpoint by F1, not by accuracy
        if f1 > best_val_f1 or (f1 == best_val_f1 and val_acc > best_val_acc):
            best_val_f1  = f1
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_path / "best.pt")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping en epoch {epoch} (sin mejora F1 por {patience} epochs)")
                break

    torch.save(model.state_dict(), output_path / "last.pt")
    (output_path / "class_names.json").write_text(json.dumps(base_ds.classes))
    (output_path / "history.json").write_text(json.dumps(history, indent=2))
    (output_path / "arch.txt").write_text(arch)

    print(f"\nClasificador guardado en: {output_dir}")
    print(f"Mejor val_acc: {best_val_acc:.4f} | Mejor F1_anomaly: {best_val_f1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",     type=str,   required=True, choices=["eyes", "legs"])
    parser.add_argument("--data_dir", type=str,   required=True)
    parser.add_argument("--output",   type=str,   required=True)
    parser.add_argument("--epochs",   type=int,   default=80)
    parser.add_argument("--batch",    type=int,   default=16)
    parser.add_argument("--lr",       type=float, default=3e-4)
    parser.add_argument("--patience", type=int,   default=15)
    parser.add_argument("--device",   type=str,   default="cpu")
    parser.add_argument("--arch",     type=str,   default="efficientnet_b0",
                        help="Arquitectura timm (default: efficientnet_b0)")
    args = parser.parse_args()

    train_classifier(
        task=args.task,
        data_dir=args.data_dir,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        patience=args.patience,
        device=args.device,
        arch=args.arch,
    )
