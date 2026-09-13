#!/usr/bin/env python3
"""Evaluate the stage-1 fundus checkpoint with AUC-ROC, comparable to the alpaca stage."""
import json
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

SEED = 42
EVAL_TF = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
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


def bootstrap_ci(y, p, n=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx])) < 2:
            continue
        vals.append(roc_auc_score(y[idx], p[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    root = Path("data/raw/eye_disease")
    out = Path("models/classifier/eyes_b2_honest")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Reproduce the stage-1 split exactly as scripts/train_eyes_honest.py drew it.
    fundus = ImageFolder(str(root))
    items = [(p, l) for p, l in fundus.imgs]
    np.random.default_rng(SEED).shuffle(items)
    val = items[: int(len(items) * 0.2)]
    anomaly_idx = fundus.classes.index("anomaly")
    print(f"classes {fundus.classes} | total {len(items)} | held-out {len(val)}")

    model = timm.create_model("efficientnet_b2", pretrained=False, num_classes=2)
    model.load_state_dict(torch.load(out / "stage1_pretrain.pt", map_location="cpu",
                                     weights_only=True))
    model.to(device).eval()

    ys, ps = [], []
    with torch.no_grad():
        for xb, yb in DataLoader(ListDataset(val, EVAL_TF), batch_size=32):
            prob = torch.softmax(model(xb.to(device)), 1)[:, anomaly_idx]
            ps.append(prob.cpu().numpy())
            ys.append(yb.numpy())
    y = np.concatenate(ys) == anomaly_idx
    p = np.concatenate(ps)

    auc = roc_auc_score(y, p)
    lo, hi = bootstrap_ci(y, p)
    pred = p >= 0.5
    res = {"n": int(len(y)), "n_anomaly": int(y.sum()),
           "auc_roc": round(float(auc), 4), "auc_ci95": [round(lo, 4), round(hi, 4)],
           "f1_anomaly": round(float(f1_score(y, pred)), 4),
           "accuracy": round(float(accuracy_score(y, pred)), 4)}
    print(json.dumps(res, indent=2))
    Path("outputs/figures/stage1_fundus_metrics.json").write_text(json.dumps(res, indent=2))
    print("wrote outputs/figures/stage1_fundus_metrics.json")


if __name__ == "__main__":
    main()
