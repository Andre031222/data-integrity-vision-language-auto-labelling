"""select_annotation_batch.py -------------------------- Selects 300 images (150 iNaturalist +."""

import csv
import random
import shutil
from pathlib import Path

import cv2  # opencv-python-headless

# Configuration
ROOT = Path(__file__).resolve().parent.parent          # project root
DATA_RAW = ROOT / "data" / "raw"

INATURALIST_DIR = DATA_RAW / "inaturalist"
ROBOFLOW_DIR    = DATA_RAW / "roboflow"

OUTPUT_DIR  = ROOT / "data" / "annotation_batch"
MANIFEST    = OUTPUT_DIR / "manifest.csv"

MIN_DIM     = 400        # minimum width AND height in pixels
MIN_BOX_AREA = 0.15      # minimum normalised bounding-box area (w * h)
N_INATURALIST = 150
N_ROBOFLOW    = 150
SEED          = 42

# Helpers

def image_size(path: Path):
    """Return (width, height) or None if the file cannot be read."""
    img = cv2.imread(str(path))
    if img is None:
        return None
    h, w = img.shape[:2]
    return w, h


def largest_box_area(label_path: Path) -> float:
    """Parse a YOLO-format label file and return the area (w * h) of the largest bounding box."""
    if not label_path.exists():
        return 0.0
    max_area = 0.0
    try:
        with label_path.open() as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                w, h = float(parts[3]), float(parts[4])
                area = w * h
                if area > max_area:
                    max_area = area
    except Exception:
        return 0.0
    return max_area


def collect_inaturalist_candidates():
    """Scan data/raw/inaturalist/*.jpg, filter by MIN_DIM, return list of dicts."""
    candidates = []
    for img_path in sorted(INATURALIST_DIR.glob("*.jpg")):
        dims = image_size(img_path)
        if dims is None:
            continue
        w, h = dims
        if w < MIN_DIM or h < MIN_DIM:
            continue
        candidates.append({
            "path":     img_path,
            "filename": img_path.name,
            "source":   "inaturalist",
            "width":    w,
            "height":   h,
            "box_area": "",   # not applicable
        })
    return candidates


def collect_roboflow_candidates():
    """Walk all roboflow subsets (*/train/images/*.jpg), filter by MIN_DIM and MIN_BOX_AREA using the."""
    candidates = []
    for img_path in sorted(ROBOFLOW_DIR.glob("*/train/images/*.jpg")):
        # Derive the corresponding label path
        label_path = (
            img_path.parent.parent  # train/
            / "labels"
            / img_path.with_suffix(".txt").name
        )

        box_area = largest_box_area(label_path)
        if box_area < MIN_BOX_AREA:
            continue

        dims = image_size(img_path)
        if dims is None:
            continue
        w, h = dims
        if w < MIN_DIM or h < MIN_DIM:
            continue

        # Build a unique filename: <dataset_slug>__<original_name>
        dataset_slug = img_path.parts[
            img_path.parts.index("roboflow") + 1
        ]
        unique_name = f"{dataset_slug}__{img_path.name}"

        candidates.append({
            "path":     img_path,
            "filename": unique_name,
            "source":   "roboflow",
            "width":    w,
            "height":   h,
            "box_area": round(box_area, 6),
        })
    return candidates


# Main

def main():
    rng = random.Random(SEED)

    # 1. Collect candidates
    print("Scanning iNaturalist images ...")
    inat_candidates = collect_inaturalist_candidates()
    print(f"  Passed size filter: {len(inat_candidates):,}")

    print("Scanning Roboflow images ...")
    rbf_candidates = collect_roboflow_candidates()
    print(f"  Passed size + box_area filters: {len(rbf_candidates):,}")

    # 2. Sample
    n_inat = min(N_INATURALIST, len(inat_candidates))
    n_rbf  = min(N_ROBOFLOW,    len(rbf_candidates))

    selected_inat = rng.sample(inat_candidates, n_inat)
    selected_rbf  = rng.sample(rbf_candidates,  n_rbf)
    selected      = selected_inat + selected_rbf

    print(f"\nSelected {len(selected_inat)} iNaturalist + {len(selected_rbf)} Roboflow = {len(selected)} total images.")

    # 3. Copy images to output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nCopying images to {OUTPUT_DIR} ...")

    copied = 0
    skipped = 0
    for entry in selected:
        dest = OUTPUT_DIR / entry["filename"]
        if dest.exists():
            skipped += 1
        else:
            shutil.copy2(entry["path"], dest)
            copied += 1

    print(f"  Copied : {copied}")
    print(f"  Already existed (skipped): {skipped}")

    # 4. Write manifest CSV
    fieldnames = ["filename", "source", "width", "height", "box_area"]
    with MANIFEST.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for entry in selected:
            writer.writerow({k: entry[k] for k in fieldnames})

    print(f"\nManifest written to {MANIFEST}")

    # 5. Summary statistics
    print("\n--- Statistics ---")
    print(f"Total selected images      : {len(selected)}")
    print(f"  iNaturalist              : {len(selected_inat)}")
    print(f"  Roboflow                 : {len(selected_rbf)}")

    widths  = [e["width"]  for e in selected]
    heights = [e["height"] for e in selected]
    print(f"Width  -- min: {min(widths)}, max: {max(widths)}, "
          f"avg: {sum(widths)/len(widths):.0f}")
    print(f"Height -- min: {min(heights)}, max: {max(heights)}, "
          f"avg: {sum(heights)/len(heights):.0f}")

    rbf_areas = [e["box_area"] for e in selected_rbf if e["box_area"] != ""]
    if rbf_areas:
        print(f"Roboflow box_area -- min: {min(rbf_areas):.4f}, "
              f"max: {max(rbf_areas):.4f}, "
              f"avg: {sum(rbf_areas)/len(rbf_areas):.4f}")

    inat_candidate_count = len(inat_candidates)
    rbf_candidate_count  = len(rbf_candidates)
    if inat_candidate_count:
        print(f"\niNaturalist sampling ratio : {n_inat}/{inat_candidate_count} "
              f"({100*n_inat/inat_candidate_count:.1f}%)")
    if rbf_candidate_count:
        print(f"Roboflow sampling ratio    : {n_rbf}/{rbf_candidate_count} "
              f"({100*n_rbf/rbf_candidate_count:.1f}%)")

    print("\nDone.")


if __name__ == "__main__":
    main()
