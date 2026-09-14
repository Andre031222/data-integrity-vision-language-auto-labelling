#!/usr/bin/env python3
"""Subsample a training set to a target size, to separate contamination from data volume."""
import argparse, json, random, shutil
from pathlib import Path

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/annotated_v3_md5")
    ap.add_argument("--out", default="data/annotated_size_control")
    ap.add_argument("--n-train", type=int, default=1022)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    src, out = Path(a.src), Path(a.out)
    shutil.rmtree(out, ignore_errors=True)
    rng = random.Random(a.seed)

    for split in ("train", "val", "test"):
        names = sorted(p.name for p in (src / "images" / split).iterdir()
                       if p.suffix.lower() in EXT)
        if split == "train":
            rng.shuffle(names)
            names = sorted(names[: a.n_train])
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)
        for n in names:
            (out / "images" / split / n).symlink_to((src / "images" / split / n).resolve())
            lbl = src / "labels" / split / (Path(n).stem + ".txt")
            if lbl.exists():
                (out / "labels" / split / lbl.name).symlink_to(lbl.resolve())
        print(f"{split:<6} {len(names)}")

    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\n"
        f"test: images/test\n\nnc: 1\nnames:\n  0: alpaca\n")
    json.dump({"source": a.src, "n_train": a.n_train, "seed": a.seed,
               "purpose": "size-matched control for the stage-5 training set"},
              open(out / "build_manifest.json", "w"), indent=2)
    print(f"\nwrote {out}/data.yaml")


if __name__ == "__main__":
    main()
