#!/usr/bin/env python3
"""Assemble the Zenodo release archive of the dataset from the built split."""
import argparse, csv, hashlib, json, shutil, sys, zipfile
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dataset_v2 import (EXCLUDED_CONTENT, EXCLUDED_SOURCES, collect, content_exclusions,
                              group_by_distance, dihedral_distance, merge_by_photo_id,
                              phash_variants, photo_id)

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scene_ids(records, tau):
    groups = merge_by_photo_id(group_by_distance(dihedral_distance(phash_variants(records)), tau),
                               records)
    label = {}
    for members in groups:
        gid = min(records[i]["name"] for i in members).split(".rf.")[0]
        for i in members:
            label[records[i]["name"]] = gid
    return label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="data/v3.1/s6photoid")
    ap.add_argument("--raw", default="data/annotated_v3")
    ap.add_argument("--version", default="3.1")
    ap.add_argument("--readme", default="docs/release/README_v3.1.md")
    ap.add_argument("--changelog", default="docs/release/CHANGELOG.md")
    ap.add_argument("--out", default="_local/zenodo/zenodo_v3.1")
    a = ap.parse_args()

    split_root, out_root = Path(a.split), Path(a.out)
    name = f"alpacavision-v{a.version}"
    stage = out_root / name
    shutil.rmtree(stage, ignore_errors=True)
    manifest = json.loads((split_root / "build_manifest.json").read_text())

    records = []
    for sp in ("train", "val", "test"):
        for f in sorted((split_root / "images" / sp).iterdir()):
            if f.suffix.lower() not in EXT:
                continue
            records.append({"split": sp, "path": f, "name": f.name,
                            "label": split_root / "labels" / sp / (f.stem + ".txt"),
                            "source": f.name.split("_")[0], "photo": photo_id(f.name)})
    scene = scene_ids(records, manifest["tau"])

    rows = []
    for r in records:
        for kind, src in (("images", r["path"]), ("labels", r["label"])):
            dst = stage / kind / r["split"] / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src.resolve(), dst)
        boxes = sum(1 for line in r["label"].read_text().splitlines() if line.strip())
        rows.append([r["name"], r["split"], r["source"], r["photo"] or "", scene[r["name"]],
                     boxes, sha256(r["path"].resolve())])
    with open(stage / "provenance.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "split", "source_project", "source_photo_id", "photo_group",
                    "boxes", "sha256"])
        w.writerows(rows)

    raw = collect(Path(a.raw))
    kept = {r["name"] for r in records}
    by_hash = defaultdict(list)
    for r in raw:
        by_hash[sha256(r["path"].resolve())].append(r)
    licensed = []
    excluded_rows = []
    for copies in by_hash.values():
        for r in copies:
            if r["name"] in kept:
                continue
            if r["source"] in EXCLUDED_SOURCES:
                excluded_rows.append([r["name"], r["source"], EXCLUDED_SOURCES[r["source"]]])
            elif any(c["name"] in kept for c in copies):
                excluded_rows.append([r["name"], r["source"], "byte-identical copy of a retained file"])
            else:
                licensed.append(r)
    content = content_exclusions(licensed, manifest["tau"])
    for i, r in enumerate(licensed):
        reason = EXCLUDED_CONTENT.get(content.get(i), "byte-identical copy")
        excluded_rows.append([r["name"], r["source"], f"content: {reason}"
                              if i in content else reason])
    with open(stage / "exclusions.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "source_project", "reason"])
        w.writerows(sorted(excluded_rows))

    (stage / "data.yaml").write_text("path: .\ntrain: images/train\nval: images/val\n"
                                     "test: images/test\n\nnc: 1\nnames:\n  0: alpaca\n")
    (stage / "build_manifest.json").write_text(json.dumps(manifest, indent=2))
    shutil.copyfile(a.readme, stage / "README.md")
    shutil.copyfile(a.changelog, stage / "CHANGELOG.md")
    (stage / "LICENSE.txt").write_text(
        f"AlpacaVision detection benchmark v{a.version}\n"
        "Creative Commons Attribution 4.0 International (CC BY 4.0)\n"
        "https://creativecommons.org/licenses/by/4.0/\n\n"
        "Every image derives from a Roboflow Universe project that declares CC BY 4.0.\n"
        "Per-image provenance is listed in provenance.csv.\n")

    archive = out_root / f"{name}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(stage.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(out_root))
    (out_root / f"{name}.zip.sha256").write_text(f"{sha256(archive)}  {archive.name}\n")

    print(f"images     : {len(records)}  {dict(Counter(r['split'] for r in records))}")
    print(f"boxes      : {sum(row[5] for row in rows)}")
    print(f"groups     : {len(set(scene.values()))}")
    print(f"excluded   : {len(excluded_rows)}  "
          f"{dict(Counter(row[2].split(':')[0] for row in excluded_rows))}")
    print(f"archive    : {archive}  ({archive.stat().st_size / 2 ** 20:.1f} MiB)")


if __name__ == "__main__":
    main()
