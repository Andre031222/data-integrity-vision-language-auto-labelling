"""AlpacaVision AI -- Entrenamiento del Detector YOLOv11."""

import argparse
from pathlib import Path

import yaml


def train_detector(config_path: str, resume: bool = False):
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("Instalar: pip install ultralytics")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_path = config.pop("model")
    run_dir = Path(config.get("project", "outputs/training_runs"))
    run_name = config.get("name", "detector_v1")

    print(f"Entrenamiento detector YOLOv11")
    print(f"  Modelo base:  {model_path}")
    print(f"  Dataset:      {config.get('data')}")
    print(f"  Epochs:       {config.get('epochs')}")
    print(f"  Batch:        {config.get('batch')}")
    print(f"  Device:       {config.get('device')}")

    if resume:
        last_pt = run_dir / run_name / "weights" / "last.pt"
        if last_pt.exists():
            print(f"  Reanudando desde: {last_pt}")
            model = YOLO(str(last_pt))
        else:
            print("  Checkpoint no encontrado, iniciando desde cero.")
            model = YOLO(model_path)
    else:
        model = YOLO(model_path)

    results = model.train(**config)

    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\nEntrenamiento finalizado.")
    print(f"  Mejor modelo: {best_pt}")
    try:
        print(f"  mAP@0.5:      {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.4f}")
        print(f"  mAP@0.5:0.95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
    except Exception:
        pass

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/train_detector.yaml")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    train_detector(config_path=args.config, resume=args.resume)
