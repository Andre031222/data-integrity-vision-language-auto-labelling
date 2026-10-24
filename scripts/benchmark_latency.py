#!/usr/bin/env python3
"""Measure single-image inference latency of the detectors on CPU and GPU, PyTorch and ONNX."""
import argparse, json, platform, statistics, time
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO


def timed(fn, warmup, reps):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(reps):
        start = time.perf_counter()
        fn()
        times.append(1000 * (time.perf_counter() - start))
    return {"median_ms": round(statistics.median(times), 2),
            "p90_ms": round(float(np.percentile(times, 90)), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", nargs="+", required=True)
    ap.add_argument("--images", default="data/v3.1/s6photoid/images/test")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default="outputs/figures/v31/latency.json")
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    images = sorted(Path(a.images).iterdir())[: a.reps]
    frames = [str(p) for p in images]
    result = {"cpu": platform.processor() or platform.machine(), "cpu_threads": a.threads,
              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
              "imgsz": a.imgsz, "batch": 1, "models": {}}

    for w in a.weights:
        name = Path(w).parts[-3] if "weights" in Path(w).parts else Path(w).stem
        entry = {"size_mb": round(Path(w).stat().st_size / 2 ** 20, 2)}
        model = YOLO(w)
        it = iter(frames * 10)
        for dev in ("cpu", "0") if torch.cuda.is_available() else ("cpu",):
            entry[f"pytorch_{'cpu' if dev == 'cpu' else 'gpu'}"] = timed(
                lambda: model.predict(next(it), imgsz=a.imgsz, device=dev, verbose=False),
                a.warmup, a.reps)
        onnx_path = model.export(format="onnx", imgsz=a.imgsz, simplify=True, verbose=False)
        onnx = YOLO(onnx_path, task="detect")
        entry["onnx_size_mb"] = round(Path(onnx_path).stat().st_size / 2 ** 20, 2)
        entry["onnx_cpu"] = timed(
            lambda: onnx.predict(next(it), imgsz=a.imgsz, device="cpu", verbose=False),
            a.warmup, a.reps)
        result["models"][name] = entry
        print(name, json.dumps(entry))

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=2))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
