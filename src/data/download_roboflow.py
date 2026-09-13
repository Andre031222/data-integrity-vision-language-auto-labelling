"""
AlpacaVision AI -- Descarga de datasets de alpacas desde Roboflow Universe.

Uso:
    export ROBOFLOW_API_KEY="tu_api_key"
    python src/data/download_roboflow.py --output data/raw/roboflow
"""

import argparse
import os
from pathlib import Path

try:
    from roboflow import Roboflow
except ImportError:
    raise ImportError("Instalar: pip install roboflow")

# Datasets públicos de alpacas confirmados en Roboflow Universe
ALPACA_DATASETS = [
    {
        "workspace": "objetos-test",
        "project": "alpaca",
        "version": 1,
        "description": "~200 imágenes -- detección de cuerpo completo",
        "url": "https://universe.roboflow.com/objetos-test/alpaca",
    },
    {
        "workspace": "mahnoor",
        "project": "alpaca-detection-ekd4g",
        "version": 1,
        "description": "~44 imágenes -- múltiples formatos",
        "url": "https://universe.roboflow.com/mahnoor/alpaca-detection-ekd4g",
    },
    # find more at https://universe.roboflow.com/search?q=class:alpaca
]


def download_roboflow_datasets(api_key: str, output_dir: str = "data/raw/roboflow", fmt: str = "yolov11"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rf = Roboflow(api_key=api_key)
    print(f"Descargando {len(ALPACA_DATASETS)} datasets | formato={fmt}\n")

    for i, ds in enumerate(ALPACA_DATASETS, 1):
        print(f"[{i}/{len(ALPACA_DATASETS)}] {ds['description']}")
        try:
            project = rf.workspace(ds["workspace"]).project(ds["project"])
            version = project.version(ds["version"])
            version.download(fmt, location=str(output_path / ds["project"]))
            print(f"  OK -> {output_path / ds['project']}\n")
        except Exception as exc:
            print(f"  ERROR: {exc}\n")

    print(f"Roboflow completado. Archivos en: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="data/raw/roboflow")
    parser.add_argument("--format", type=str, default="yolov11",
                        choices=["yolov11", "yolov8", "coco", "voc"])
    args = parser.parse_args()

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise ValueError(
            "Falta ROBOFLOW_API_KEY.\n"
            "Obtenla en: https://app.roboflow.com/settings/api\n"
            "Luego: export ROBOFLOW_API_KEY='tu_key'"
        )

    download_roboflow_datasets(api_key=api_key, output_dir=args.output, fmt=args.format)
