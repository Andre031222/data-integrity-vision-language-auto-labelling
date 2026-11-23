#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
V=data/v3.1
PROJ="$PWD/outputs/training_runs_v31"
OUT=outputs/figures/v31
N=models/pretrained/yolo11n.pt

while pgrep -f "scripts/seed_sweep_classifie[r].py" >/dev/null; do sleep 60; done

done_already() { [ -f "$1" ] && ! grep -q placeholder "$1"; }

run() {
  local tag=$1 data=$2 w=$3; shift 3
  done_already "$OUT/$tag.json" && { echo "[skip] $tag"; return; }
  rm -rf "$PROJ/${tag}_seed"*
  "$PY" scripts/train_detector_v2.py --data "$data/data.yaml" --weights "$w" --tag "$tag" \
      --seeds "$@" --project "$PROJ" --out "$OUT/$tag.json"
}

for ss in 43 44; do
  for st in s1raw s2sha; do
    run "${st}_n1035_ss$ss" "$V/size_control/${st}_n1035_ss$ss" $N 0 1 2
  done
done
echo "r2 size-control runs finished"
