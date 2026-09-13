"""AlpacaVision AI -- Generate publication-quality Grad-CAM panel (Figure 5)."""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import datasets, transforms

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation.gradcam import GradCAM

MODEL_DIR = ROOT / "models" / "classifier" / "eyes_b2"
CROPS_DIR = ROOT / "data" / "crops" / "eyes"
MANIFEST_PATH = MODEL_DIR / "split_manifest.json"
FIGURE_OUT = ROOT / "paper" / "figures"

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

_PANEL = {
    "TP": {"title": "True Positive (TP)", "sub": "Anomaly correctly detected", "color": "#16a34a"},
    "TN": {"title": "True Negative (TN)", "sub": "Normal correctly identified", "color": "#2563eb"},
    "FP": {"title": "False Positive (FP)", "sub": "Normal misclassified as anomaly", "color": "#d97706"},
    "FN": {"title": "False Negative (FN)", "sub": "Anomaly missed by classifier",  "color": "#dc2626"},
}


def main():
    import timm
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not (MODEL_DIR / "best.pt").exists():
        print(f"ERROR: Model not found: {MODEL_DIR / 'best.pt'}")
        print("  Train first: python scripts/train_two_stage.py")
        sys.exit(1)
    if not MANIFEST_PATH.exists():
        print(f"ERROR: Manifest not found: {MANIFEST_PATH}")
        print("  Run: python scripts/evaluate_classifiers.py --task eyes")
        sys.exit(1)

    arch = (MODEL_DIR / "arch.txt").read_text().strip() if (MODEL_DIR / "arch.txt").exists() else "efficientnet_b2"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Arch: {arch}")

    model = timm.create_model(arch, pretrained=False, num_classes=2)
    model.load_state_dict(torch.load(MODEL_DIR / "best.pt", map_location=device, weights_only=True))
    model = model.to(device).eval()

    base_ds = datasets.ImageFolder(str(CROPS_DIR))
    anomaly_idx = base_ds.class_to_idx.get("anomaly", 0)
    manifest = json.loads(MANIFEST_PATH.read_text())
    test_idx = manifest["test"]
    print(f"Test set: {len(test_idx)} samples | class_to_idx={base_ds.class_to_idx}")

    # Inference pass (no gradient tracking needed here)
    records = []
    with torch.no_grad():
        for i in test_idx:
            img_path, label = base_ds.imgs[i]
            pil = Image.open(img_path).convert("RGB")
            tensor = VAL_TRANSFORM(pil).unsqueeze(0).to(device)
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            pred = int(logits.argmax(1).item())
            records.append((img_path, label, pred, float(probs[anomaly_idx])))

    # Bucket into quadrants
    buckets = {"TP": [], "TN": [], "FP": [], "FN": []}
    for rec in records:
        _, true, pred, ap = rec
        t_anom = (true == anomaly_idx)
        p_anom = (pred == anomaly_idx)
        key = ("TP" if t_anom and p_anom else
               "TN" if not t_anom and not p_anom else
               "FP" if not t_anom and p_anom else "FN")
        buckets[key].append(rec)

    for k, v in buckets.items():
        print(f"  {k}: {len(v)} examples")

    def pick_best(items, quadrant):
        if not items:
            return None
        # TP/FP: highest anomaly probability; TN/FN: lowest anomaly probability
        reverse = quadrant in ("TP", "FP")
        return sorted(items, key=lambda x: x[3], reverse=reverse)[0]

    selected = {k: pick_best(v, k) for k, v in buckets.items()}

    # Grad-CAM generation (needs gradient flow -- outside torch.no_grad context)
    overlays = {}
    for qkey, rec in selected.items():
        if rec is None:
            print(f"  WARNING: No {qkey} example available -- placeholder will be shown")
            continue
        img_path, true, pred, ap = rec
        print(f"  Grad-CAM {qkey}: {Path(img_path).name} (anomaly_prob={ap:.3f})")

        pil_orig = Image.open(img_path).convert("RGB").resize((224, 224))
        orig_rgb = np.array(pil_orig)

        tensor = VAL_TRANSFORM(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)

        gcam = GradCAM(model)
        with torch.enable_grad():
            heatmap = gcam(tensor, target_class=pred)
        gcam.remove_hooks()

        heatmap_u8 = (cv2.resize(heatmap, (224, 224)) * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        overlay = np.clip(
            orig_rgb.astype(float) * 0.55 + heatmap_rgb.astype(float) * 0.45,
            0, 255
        ).astype(np.uint8)
        overlays[qkey] = (overlay, ap)

    # Compose 2x2 panel
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.patch.set_facecolor("white")
    layout = [("TP", axes[0, 0]), ("TN", axes[0, 1]), ("FP", axes[1, 0]), ("FN", axes[1, 1])]

    for qkey, ax in layout:
        info = _PANEL[qkey]
        if qkey in overlays:
            img_arr, ap = overlays[qkey]
            ax.imshow(img_arr)
            conf_str = f"P(anomaly) = {ap:.3f}"
        else:
            placeholder = np.full((224, 224, 3), 210, dtype=np.uint8)
            ax.imshow(placeholder)
            ax.text(112, 112, "No example\nin test set",
                    ha="center", va="center", fontsize=11, color="#6b7280")
            conf_str = "No example in test set"

        ax.set_title(info["title"], fontsize=11, fontweight="bold",
                     color=info["color"], pad=5)
        ax.set_xlabel(f"{info['sub']}\n{conf_str}", fontsize=9, color="#374151")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(info["color"])
            spine.set_linewidth(2.5)

    fig.suptitle(
        "Grad-CAM Gradient-Weighted Class Activation Maps\n"
        "EfficientNet-B2 Ocular Anomaly Classifier -- Test Set Examples",
        fontsize=13, fontweight="bold", y=1.02
    )
    plt.tight_layout(pad=1.5)

    FIGURE_OUT.mkdir(parents=True, exist_ok=True)
    out_png = FIGURE_OUT / "fig5_gradcam.png"
    out_pdf = FIGURE_OUT / "fig5_gradcam.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"\nSaved: {out_png}")
    print(f"Saved: {out_pdf}")
    print("\nNext steps:")
    print("  1. Check paper/figures/fig5_gradcam.png visually")
    print("  2. Recompile: cd paper/manuscript && pdflatex preview.tex && bibtex preview && pdflatex preview.tex && pdflatex preview.tex")


if __name__ == "__main__":
    main()
