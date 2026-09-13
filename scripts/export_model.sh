#!/bin/bash
# AlpacaVision AI -- Exportar modelos a ONNX y TFLite

set -e
BEST_PT="outputs/training_runs/detector_v1/weights/best.pt"

if [ ! -f "$BEST_PT" ]; then
    echo "ERROR: Modelo no encontrado: $BEST_PT"
    exit 1
fi

echo "Exportando detector a ONNX..."
python -c "
from ultralytics import YOLO
model = YOLO('$BEST_PT')
model.export(format='onnx', imgsz=640, simplify=True)
print('ONNX exportado.')
"

echo "Exportando detector a TFLite (int8)..."
python -c "
from ultralytics import YOLO
model = YOLO('$BEST_PT')
model.export(format='tflite', imgsz=640, int8=True)
print('TFLite exportado.')
"

echo "Modelos exportados."
