#!/bin/bash
# AlpacaVision AI -- Pipeline de entrenamiento completo

set -e
echo "=========================================="
echo "  AlpacaVision AI -- Pipeline Entrenamiento"
echo "=========================================="

python -c "import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"
python -c "from ultralytics import YOLO; print('Ultralytics OK')"

echo ""
echo "-- PASO 1: Preprocesando datos ------------"
python src/data/preprocess.py \
    --input data/raw/roboflow data/raw/inaturalist data/raw/field \
    --output data/processed

echo ""
echo "-- PASO 2: Dividiendo dataset -------------"
python src/data/split_dataset.py \
    --images data/processed \
    --labels data/annotated_v3/labels_raw \
    --output data/annotated_v3

echo ""
echo "-- PASO 3: Entrenando detector YOLOv11 ----"
python src/training/train_detector.py --config config/train_detector.yaml

echo ""
echo "-- PASO 4: Extrayendo recortes ROI --------"
python src/data/crop_regions.py \
    --detector outputs/training_runs/detector_v1/weights/best.pt \
    --images data/annotated_v3/images/train \
    --output data/crops

echo ""
echo "Recortes extraídos. Ahora etiqueta anomalías en data/crops/ y ejecuta:"
echo "  bash scripts/train_classifiers.sh"
echo ""
echo "=============================================="
echo "  Pipeline de detección completado"
echo "=============================================="
