#!/usr/bin/env python3
"""Repeat the published stage-2 ocular protocol over several seeds, unchanged."""
import argparse, json, statistics, sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.group_split import build_group_split, assert_no_leakage  # noqa: E402

CROPS = Path("data/crops/eyes")
STAGE1 = Path("models/classifier/eyes_b2_honest/stage1_pretrain.pt")

# Exactly the augmentation of the published run.
TRAIN_TF_STRONG = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
# Milder variant, used to test how much the geometric jitter costs.
TRAIN_TF_MILD = transforms.Compose([
    transforms.Resize(256), transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(), transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
EVAL_TF = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class ListDataset(Dataset):
    def __init__(self, items, tf):
        self.items, self.tf = items, tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        return self.tf(Image.open(p).convert("RGB")), y


class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=1.5):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=alpha, reduction="none")

    def forward(self, logits, labels):
        ce = self.ce(logits, labels)
        return ((1 - torch.exp(-ce)) ** self.gamma * ce).mean()


@torch.no_grad()
def probs(model, items, device, anomaly_idx, tta=False):
    model.eval()
    ys, ps = [], []
    for xb, yb in DataLoader(ListDataset(items, EVAL_TF), batch_size=32):
        xb = xb.to(device)
        p = torch.softmax(model(xb), 1)
        if tta:
            p = (p + torch.softmax(model(torch.flip(xb, dims=[3])), 1)) / 2
        ps.append(p[:, anomaly_idx].cpu().numpy())
        ys.append(yb.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def one_seed(seed, epochs, patience, gamma, device, aug='strong', tta=True):
    split = build_group_split(CROPS, seed=seed, test_frac=0.15, val_frac=0.15,
                              originals_only_eval=True)
    assert_no_leakage(split)
    classes = split["classes"]
    ai = classes.index("anomaly")

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = timm.create_model("efficientnet_b2", pretrained=False, num_classes=2)
    model.load_state_dict(torch.load(STAGE1, map_location="cpu", weights_only=True))
    model.to(device)

    labels = [l for _, l in split["train"]]
    counts = np.bincount(labels, minlength=2).astype(float)
    gen = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler([1.0 / counts[l] for l in labels], len(labels),
                                    replacement=True, generator=gen)
    tf = TRAIN_TF_STRONG if aug == 'strong' else TRAIN_TF_MILD
    loader = DataLoader(ListDataset(split["train"], tf), batch_size=16,
                        sampler=sampler, num_workers=0)
    alpha = torch.tensor([len(labels) / (2 * c) for c in counts],
                         dtype=torch.float, device=device)
    crit = FocalLoss(alpha, gamma)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_f1, no_improve, best_state = -1.0, 0, None
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            crit(model(xb.to(device)), yb.to(device)).backward()
            opt.step()
        sched.step()
        y, p = probs(model, split["val"], device, ai, tta)
        f1 = f1_score((y == ai).astype(int), (p >= 0.5).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, no_improve = f1, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    model.load_state_dict(best_state)

    # Threshold chosen on validation by macro F1, exactly as in the published run.
    y, p = probs(model, split["val"], device, ai, tta)
    yb = (y == ai).astype(int)
    thr = max(np.linspace(0.05, 0.95, 91),
              key=lambda t: f1_score(yb, (p >= t).astype(int), average="macro",
                                     zero_division=0))
    y, p = probs(model, split["test"], device, ai, tta)
    yb = (y == ai).astype(int)
    pred = (p >= thr).astype(int)
    return {"seed": seed, "aug": aug, "tta": tta, "threshold": round(float(thr), 3),
            "n_test": int(len(yb)), "n_test_anomaly": int(yb.sum()),
            "auc_roc": round(float(roc_auc_score(yb, p)), 4),
            "f1_anomaly": round(float(f1_score(yb, pred, zero_division=0)), 4),
            "accuracy": round(float(accuracy_score(yb, pred)), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 0, 1, 2])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--gamma", type=float, default=1.5)
    ap.add_argument("--aug", choices=("strong", "mild"), default="strong",
                    help="strong reproduces the published augmentation")
    ap.add_argument("--no-tta", dest="tta", action="store_false",
                    help="the published run used test-time horizontal flipping")
    ap.add_argument("--out", default="outputs/figures/classifier_seed_sweep.json")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    runs = []
    for s in a.seeds:
        r = one_seed(s, a.epochs, a.patience, a.gamma, device, a.aug, a.tta)
        runs.append(r)
        print(f"[seed {s}] {r}", flush=True)
        aucs = [x["auc_roc"] for x in runs]
        Path(a.out).write_text(json.dumps(
            {"protocol": f"stage-2 protocol, aug={a.aug}, tta={a.tta}",
             "per_seed": runs,
             "auc_mean": round(statistics.mean(aucs), 4),
             "auc_std": round(statistics.pstdev(aucs), 4) if len(aucs) > 1 else 0.0,
             "auc_min": min(aucs), "auc_max": max(aucs)}, indent=2))

    aucs = [x["auc_roc"] for x in runs]
    print(f"\nAUC-ROC across seeds: mean {statistics.mean(aucs):.4f} "
          f"+/- {statistics.pstdev(aucs):.4f}  range [{min(aucs):.3f}, {max(aucs):.3f}]")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
