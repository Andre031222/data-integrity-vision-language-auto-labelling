#!/usr/bin/env python3
"""Isolate how much of the classifier's apparent skill came from augmentation leakage."""
import argparse, json, re, statistics, warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

CROPS = Path("data/crops/eyes")
STAGE1 = Path("models/classifier/eyes_b2_honest/stage1_pretrain.pt")
CLASSES = ("anomaly", "normal")

TRAIN_TF = transforms.Compose([
    transforms.Resize(256), transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(), transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
EVAL_TF = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class CropSet(Dataset):
    def __init__(self, items, tf):
        self.items, self.tf = items, tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        return self.tf(Image.open(p).convert("RGB")), y


def source_of(name):
    """Strip the aug_NN_ prefix and the crop suffix to recover the source image."""
    return re.sub(r"^aug_\d+_", "", name).split("_cls2_")[0]


def load_items():
    items = []
    for idx, cls in enumerate(CLASSES):
        for f in sorted((CROPS / cls).iterdir()):
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                items.append({"path": f, "y": idx, "name": f.name,
                              "group": source_of(f.name),
                              "aug": f.name.startswith("aug_")})
    return items


def split_random(items, seed):
    """File-level split: augmented copies of one image can land on both sides."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(items))
    n_tr, n_va = int(0.70 * len(idx)), int(0.15 * len(idx))
    take = lambda s: [items[i] for i in s]
    return take(idx[:n_tr]), take(idx[n_tr:n_tr + n_va]), take(idx[n_tr + n_va:])


def split_group(items, seed):
    """Group-aware split; validation and test hold only unaugmented originals."""
    groups = defaultdict(list)
    for it in items:
        groups[it["group"]].append(it)
    keys = sorted(groups)
    rng = np.random.default_rng(seed)
    keys = [keys[i] for i in rng.permutation(len(keys))]
    n_tr, n_va = int(0.70 * len(keys)), int(0.15 * len(keys))
    part = lambda ks, originals_only: [it for k in ks for it in groups[k]
                                       if not (originals_only and it["aug"])]
    return (part(keys[:n_tr], False),
            part(keys[n_tr:n_tr + n_va], True),
            part(keys[n_tr + n_va:], True))


def run(train, val, test, seed, epochs, device, select='auc'):
    torch.manual_seed(seed)
    model = timm.create_model("efficientnet_b2", pretrained=False, num_classes=2)
    if STAGE1.exists():
        model.load_state_dict(torch.load(STAGE1, map_location="cpu", weights_only=True))
    model.to(device)

    tr = [(it["path"], it["y"]) for it in train]
    counts = np.bincount([y for _, y in tr], minlength=2)
    w = [1.0 / counts[y] for _, y in tr]
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(CropSet(tr, TRAIN_TF), batch_size=16, num_workers=0,
                        sampler=WeightedRandomSampler(w, len(w), generator=gen))
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()

    def probs(split):
        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for xb, yb in DataLoader(CropSet([(i["path"], i["y"]) for i in split],
                                             EVAL_TF), batch_size=32):
                ps.append(torch.softmax(model(xb.to(device)), 1)[:, 0].cpu().numpy())
                ys.append(yb.numpy())
        return np.concatenate(ys) == 0, np.concatenate(ps)

    best, best_state = -1.0, None
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            lossf(model(xb.to(device)), yb.to(device)).backward()
            opt.step()
        y, p = probs(val)
        if select == "auc":
            score = roc_auc_score(y, p) if len(set(y)) > 1 else 0.5
        else:
            score = f1_score(y, p >= 0.5, average="macro", zero_division=0)
        if score > best:
            best, best_state = score, {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    y, p = probs(test)
    pred = p >= 0.5
    return {"auc_roc": round(float(roc_auc_score(y, p)), 4),
            "f1_anomaly": round(float(f1_score(y, pred, zero_division=0)), 4),
            "accuracy": round(float(accuracy_score(y, pred)), 4),
            "n_test": int(len(y)), "n_test_anomaly": int(y.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--select", choices=("auc", "f1"), default="auc",
                    help="validation criterion used to keep the best epoch")
    ap.add_argument("--out", default="outputs/figures/classifier_leakage_ablation.json")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    items = load_items()
    print(f"crops: {len(items)} | augmented: {sum(i['aug'] for i in items)} | "
          f"source groups: {len({i['group'] for i in items})} | device: {device}")

    out = {}
    for cond, splitter in (("file_level_split", split_random),
                           ("group_aware_split", split_group)):
        runs = []
        for seed in a.seeds:
            tr, va, te = splitter(items, seed)
            r = run(tr, va, te, seed, a.epochs, device, a.select)
            runs.append(r)
            print(f"[{cond}] seed {seed}: {r}", flush=True)
        out[cond] = {"per_seed": runs,
                     "auc_mean": round(statistics.mean(r["auc_roc"] for r in runs), 4),
                     "auc_std": round(statistics.pstdev([r["auc_roc"] for r in runs]), 4)}
        Path(a.out).write_text(json.dumps(out, indent=2))

    print("\ncondition                 AUC-ROC")
    for k, v in out.items():
        print(f"  {k:<22} {v['auc_mean']:.4f} +/- {v['auc_std']:.4f}")
    gap = out["file_level_split"]["auc_mean"] - out["group_aware_split"]["auc_mean"]
    print(f"\nattributable to augmentation leakage: {gap:+.4f} AUC")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
