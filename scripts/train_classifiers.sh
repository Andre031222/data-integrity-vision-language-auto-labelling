#!/bin/bash
# AlpacaVision AI -- Entrenamiento de clasificadores

set -e

echo "-- Clasificador de Ojos -------------------"
python src/training/train_classifier.py \
    --task eyes \
    --data_dir data/crops/eyes \
    --output models/classifier/eyes \
    --epochs 50 \
    --batch 32

echo ""
echo "-- Clasificador de Extremidades -----------"
python src/training/train_classifier.py \
    --task legs \
    --data_dir data/crops/legs \
    --output models/classifier/legs \
    --epochs 50 \
    --batch 32

echo ""
echo "Clasificadores entrenados. Lanzar app:"
echo "  python run_webapp.py"
