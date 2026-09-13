"""Evaluate a YOLO detector on one split, single-process for Windows safety."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default="config/dataset_stage1_clean.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.weights)
    r = model.val(data=args.data, split=args.split, device=args.device,
                  workers=0, verbose=False, plots=True)
    out = {
        "weights": args.weights, "data": args.data, "split": args.split,
        "mAP50": float(r.box.map50), "mAP50_95": float(r.box.map),
        "precision": float(r.box.mp), "recall": float(r.box.mr),
    }
    print(json.dumps(out, indent=2))
    dst = ROOT / "outputs" / "figures" / f"detector_clean_{args.split}_metrics.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=2))
    print(f"Guardado: {dst}")


if __name__ == "__main__":
    main()
