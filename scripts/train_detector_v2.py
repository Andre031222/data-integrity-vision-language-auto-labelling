#!/usr/bin/env python3
"""Train and evaluate the detector on dataset v2 over several seeds."""
import argparse, json, statistics, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
from ultralytics import YOLO


def run(weights, data, seed, epochs, imgsz, batch, device, project, name):
    model = YOLO(weights)
    model.train(data=data, epochs=epochs, imgsz=imgsz, batch=batch, device=device,
                seed=seed, optimizer="AdamW", lr0=1e-3, weight_decay=5e-4,
                patience=20, workers=4, project=project, name=name,
                exist_ok=True, verbose=False, plots=False)
    res = model.val(data=data, split="test", imgsz=imgsz, batch=batch, device=device,
                    project=project, name=f"{name}_test", exist_ok=True,
                    verbose=False, plots=False)
    b = res.box
    return {"mAP50": round(float(b.map50), 4), "mAP50_95": round(float(b.map), 4),
            "precision": round(float(b.mp), 4), "recall": round(float(b.mr), 4)}


def summarise(runs):
    keys = runs[0].keys()
    return {k: {"mean": round(statistics.mean(r[k] for r in runs), 4),
                "std": round(statistics.pstdev([r[k] for r in runs]), 4) if len(runs) > 1 else 0.0,
                "values": [r[k] for r in runs]} for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/annotated_v3_final/data.yaml")
    ap.add_argument("--weights", default="models/pretrained/yolo11n.pt")
    ap.add_argument("--tag", default="v2_n")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0")
    ap.add_argument("--project", default="outputs/training_runs_v2")
    ap.add_argument("--out", default="outputs/figures/detector_v2_metrics.json")
    a = ap.parse_args()

    results = {}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for seed in a.seeds:
        name = f"{a.tag}_seed{seed}"
        print(f"[train] {name}", flush=True)
        results[str(seed)] = run(a.weights, a.data, seed, a.epochs, a.imgsz,
                                 a.batch, a.device, a.project, name)
        print(f"[done ] {name}: {results[str(seed)]}", flush=True)
        payload = {"dataset": a.data, "weights": a.weights, "epochs": a.epochs,
                   "per_seed": results, "summary": summarise(list(results.values()))}
        out.write_text(json.dumps(payload, indent=2))

    print("\nsummary over seeds", a.seeds)
    for k, v in payload["summary"].items():
        print(f"  {k:<10} {v['mean']:.4f} +/- {v['std']:.4f}   {v['values']}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
