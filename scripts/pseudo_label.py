"""AlpacaVision AI -- Pseudo-etiquetado de crops no etiquetados."""

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

try:
    import timm
except ImportError:
    raise ImportError("pip install timm")

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def run_pseudo_label(task: str, threshold: float, dry_run: bool):
    from PIL import Image

    model_dir = ROOT / "models" / "classifier" / task
    model_path = model_dir / "best.pt"
    class_names_path = model_dir / "class_names.json"

    if not model_path.exists():
        print(f"ERROR: Modelo no encontrado en {model_path}")
        sys.exit(1)

    class_names = json.loads(class_names_path.read_text())
    # ImageFolder ordena alphabetically: anomaly=0, normal=1
    anomaly_idx = class_names.index("anomaly")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=len(class_names))
    model.load_state_dict(torch.load(str(model_path), map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    unlabeled_dir = ROOT / "data" / "unlabeled" / task
    anomaly_dest  = ROOT / "data" / "crops" / task / "anomaly"
    anomaly_dest.mkdir(parents=True, exist_ok=True)

    if not unlabeled_dir.exists() or not any(unlabeled_dir.glob("*.jpg")):
        print(f"[{task}] Sin imagenes en {unlabeled_dir}")
        return

    images = sorted(unlabeled_dir.glob("*.jpg")) + sorted(unlabeled_dir.glob("*.png"))
    print(f"\n[{task}] Pseudo-etiquetando {len(images)} imagenes con threshold={threshold:.0%}")

    counts = {"anomaly_high_conf": 0, "normal": 0, "anomaly_low_conf": 0}
    new_anomalies = []

    with torch.no_grad():
        for img_path in tqdm(images, desc=f"Clasificando {task}"):
            try:
                img = Image.open(str(img_path)).convert("RGB")
            except Exception:
                continue

            tensor = VAL_TRANSFORM(img).unsqueeze(0).to(device)
            logits = model(tensor)
            probs  = F.softmax(logits, dim=1)[0]
            anomaly_prob = probs[anomaly_idx].item()

            if anomaly_prob >= threshold:
                counts["anomaly_high_conf"] += 1
                new_anomalies.append((img_path, anomaly_prob))
            elif anomaly_prob >= 0.5:
                counts["anomaly_low_conf"] += 1
            else:
                counts["normal"] += 1

    print(f"\n  Anomalia alta conf (>={threshold:.0%}): {counts['anomaly_high_conf']}")
    print(f"  Anomalia baja conf (50-{threshold:.0%}):  {counts['anomaly_low_conf']}")
    print(f"  Normal:                          {counts['normal']}")

    if not new_anomalies:
        print("\n  No se encontraron nuevas anomalias con alta confianza.")
        return

    print(f"\n  Top 20 por confianza:")
    for p, conf in sorted(new_anomalies, key=lambda x: -x[1])[:20]:
        print(f"    {p.name}: {conf:.3f}")

    if not dry_run:
        copied = 0
        for img_path, conf in new_anomalies:
            dest = anomaly_dest / f"pseudo_{img_path.stem}_c{conf:.2f}.jpg"
            if not dest.exists():
                shutil.copy2(str(img_path), str(dest))
                copied += 1
        print(f"\n  Copiadas {copied} nuevas anomalias a {anomaly_dest}")
        total = len(list(anomaly_dest.glob("*.jpg")))
        normal_total = len(list((ROOT / "data" / "crops" / task / "normal").glob("*.jpg")))
        print(f"  Total anomaly: {total} | Normal: {normal_total} | Ratio: {normal_total/total:.2f}:1")
    else:
        print("\n  [DRY RUN] No se copiaron archivos. Usa sin --dry_run para aplicar.")


def main():
    parser = argparse.ArgumentParser(description="Pseudo-etiquetado con clasificador entrenado")
    parser.add_argument("--task",      choices=["eyes", "legs", "both"], default="eyes")
    parser.add_argument("--threshold", type=float, default=0.85,
                        help="Confianza minima para aceptar como anomalia (default: 0.85)")
    parser.add_argument("--dry_run",   action="store_true",
                        help="Solo mostrar estadisticas, no copiar archivos")
    args = parser.parse_args()

    tasks = ["eyes", "legs"] if args.task == "both" else [args.task]
    for task in tasks:
        run_pseudo_label(task, args.threshold, args.dry_run)


if __name__ == "__main__":
    main()
