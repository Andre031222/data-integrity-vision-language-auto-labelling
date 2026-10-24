#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
V=data/v3.1
PROJ="$PWD/outputs/training_runs_v31"
OUT=outputs/figures/v31
N=models/pretrained/yolo11n.pt
S=models/pretrained/yolo11s.pt
mkdir -p "$OUT"

while pgrep -f "scripts/run_revision_experiment[s].sh" >/dev/null; do sleep 60; done

done_already() { [ -f "$1" ] && ! grep -q placeholder "$1"; }

run() {
  local tag=$1 data=$2 w=$3; shift 3
  done_already "$OUT/$tag.json" && { echo "[skip] $tag"; return; }
  rm -rf "$PROJ/${tag}_seed"*
  "$PY" scripts/train_detector_v2.py --data "$data/data.yaml" --weights "$w" --tag "$tag" \
      --seeds "$@" --project "$PROJ" --out "$OUT/$tag.json"
}

NT=$("$PY" -c "import json;print(json.load(open('$V/s6photoid/build_manifest.json'))['per_split_images']['train'])")

run s1raw      $V/s1raw      $N 0 1 2
run s2sha      $V/s2sha      $N 0 1 2
run s3src      $V/s3src      $N 0 1 2
run s4phash    $V/s4phash    $N 0 1 2
run s5dihedral $V/s5dihedral $N 0 1 2
run s6photoid  $V/s6photoid  $N 0 1 2
run "s1raw_n$NT" "$V/size_control/s1raw_n$NT" $N 0 1 2
run "s2sha_n$NT" "$V/size_control/s2sha_n$NT" $N 0 1 2
for k in 1 2 3 4; do
  for st in s3src s4phash s5dihedral s6photoid; do
    run "${st}_split$k" "$V/splits/${st}_split$k" $N 0
  done
done
run s6photoid_yolo11s $V/s6photoid $S 0 1 2

"$PY" scripts/eval_stage1_fundus.py
done_already "$OUT/classifier_fixed_split.json" || "$PY" scripts/seed_sweep_classifier.py \
    --seeds 0 1 2 3 4 5 6 7 8 9 --split-seed 42 --out "$OUT/classifier_fixed_split.json"
done_already "$OUT/classifier_split_sweep.json" || "$PY" scripts/seed_sweep_classifier.py \
    --seeds 42 0 1 2 3 4 5 6 7 8 --out "$OUT/classifier_split_sweep.json"
"$PY" scripts/compare_dedup_methods.py --devices cpu cuda
"$PY" scripts/benchmark_latency.py --weights "$PROJ/s6photoid_seed0/weights/best.pt" \
    "$PROJ/s6photoid_yolo11s_seed0/weights/best.pt"
"$PY" scripts/analyse_contamination.py --reference s6photoid \
    --stages "s1raw=$V/s1raw" "s2sha=$V/s2sha" "s3src=$V/s3src" "s4phash=$V/s4phash" \
             "s5dihedral=$V/s5dihedral" "s6photoid=$V/s6photoid" \
             "s1raw_n$NT=$V/size_control/s1raw_n$NT" "s2sha_n$NT=$V/size_control/s2sha_n$NT"
echo "revision runs finished"
