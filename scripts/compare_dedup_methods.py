#!/usr/bin/env python3
"""Compare dihedral pHash with CLIP embeddings for near-duplicate detection: cost and recall."""
import argparse, json, platform, sys, time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyse_contamination import cross_distance, list_split
from build_dataset_v2 import TRANSFORMS, _bits, photo_id
import imagehash


def phash_all(records):
    start = time.perf_counter()
    out = np.stack([np.stack([_bits(int(str(imagehash.phash(fn(Image.open(r["path"]).convert("RGB")))), 16))
                              for fn in TRANSFORMS]) for r in records])
    return out, time.perf_counter() - start


def clip_all(records, device, batch):
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    model = model.to(device).eval()
    feats = []
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        for k in range(0, len(records), batch):
            x = torch.stack([preprocess(Image.open(r["path"]).convert("RGB"))
                             for r in records[k:k + batch]]).to(device)
            f = model.encode_image(x)
            feats.append(torch.nn.functional.normalize(f.float(), dim=1).cpu())
    if device == "cuda":
        torch.cuda.synchronize()
    return torch.cat(feats).numpy(), time.perf_counter() - start


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/v3.1/s6photoid")
    ap.add_argument("--tau", type=int, default=6)
    ap.add_argument("--devices", nargs="+", default=["cpu"])
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", default="outputs/figures/v31/dedup_method_comparison.json")
    a = ap.parse_args()

    records = [r for sp in ("train", "val", "test") for r in list_split(a.root, sp)]
    n = len(records)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    bits, t_hash = phash_all(records)
    t0 = time.perf_counter()
    d = cross_distance(bits, bits)
    t_hash_pairs = time.perf_counter() - t0

    pid = np.array([photo_id(r["name"]) or f"none-{i}" for i, r in enumerate(records)])
    iu = np.triu_indices(n, 1)
    same_photo = pid[iu[0]] == pid[iu[1]]
    hash_dup = d[iu] <= a.tau

    result = {"images": n, "pairs": int(len(iu[0])), "photo_id_pairs": int(same_photo.sum()),
              "cpu": platform.processor() or platform.machine(), "torch_threads": torch.get_num_threads(),
              "phash_dihedral": {"seconds_total": round(t_hash, 2),
                                 "ms_per_image": round(1000 * t_hash / n, 2),
                                 "seconds_all_pairs": round(t_hash_pairs, 3),
                                 "bytes_per_image": 8 * len(TRANSFORMS),
                                 "photo_id_pairs_found": int((hash_dup & same_photo).sum()),
                                 "other_pairs_flagged": int((hash_dup & ~same_photo).sum())},
              "clip_vit_b32": {}}

    emb = None
    for dev in a.devices:
        if dev == "cuda" and not torch.cuda.is_available():
            continue
        emb, t_clip = clip_all(records, dev, a.batch)
        result["clip_vit_b32"][dev] = {"seconds_total": round(t_clip, 2),
                                       "ms_per_image": round(1000 * t_clip / n, 2)}
        if dev == "cuda":
            result["clip_vit_b32"][dev]["gpu"] = torch.cuda.get_device_name(0)

    sim = (emb @ emb.T)[iu]
    result["clip_vit_b32"]["bytes_per_image"] = 4 * emb.shape[1]
    sweep = {}
    for thr in (0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98):
        flag = sim >= thr
        sweep[str(thr)] = {"photo_id_pairs_found": int((flag & same_photo).sum()),
                           "other_pairs_flagged": int((flag & ~same_photo).sum()),
                           "photo_id_pairs_missed_by_hash_found": int((flag & same_photo & ~hash_dup).sum())}
    result["clip_vit_b32"]["threshold_sweep"] = sweep
    result["photo_id_pairs_missed_by_hash"] = int((same_photo & ~hash_dup).sum())

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
