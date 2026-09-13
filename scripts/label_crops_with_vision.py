"""AlpacaVision AI -- Auto-etiquetado de crops con Groq Vision."""

import argparse
import base64
import json
import os
import shutil
import sys
import time
from pathlib import Path

import cv2
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

ROOT = Path(__file__).parent.parent
CROPS_DIR = ROOT / "data" / "crops"
MANIFEST_PATH = ROOT / "data" / "crops" / "vision_label_manifest.json"

EYE_PROMPT = """Eres un veterinario experto en camélidos sudamericanos.
Analiza este RECORTE de la región ocular de una alpaca.
Responde SOLO en json (objeto json):
{
  "has_anomaly": true/false,
  "anomaly_type": "eye_normal"|"eye_cataract"|"eye_opacity"|"eye_discharge"|"eye_inflammation",
  "confidence": 0.0-1.0,
  "description": "descripcion breve en español"
}
Si la imagen es demasiado pequeña, borrosa o no muestra claramente el ojo, responde has_anomaly=false y anomaly_type="eye_normal".
"""

LEG_PROMPT = """Eres un veterinario experto en camélidos sudamericanos.
Analiza este RECORTE de la extremidad (pata) de una alpaca.
Responde SOLO en json (objeto json):
{
  "has_anomaly": true/false,
  "anomaly_type": "leg_normal"|"leg_angular"|"leg_hoof_abnormal"|"leg_swelling"|"leg_posture_abnormal",
  "confidence": 0.0-1.0,
  "description": "descripcion breve en español"
}
Si la imagen es demasiado pequeña, borrosa o no muestra claramente la pata, responde has_anomaly=false y anomaly_type="leg_normal".
"""

VISION_MODEL   = "meta-llama/llama-4-scout-17b-16e-instruct"
FALLBACK_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"  # único modelo vision disponible


def img_to_b64(path: Path) -> str:
    img = cv2.imread(str(path))
    if img is None:
        return None
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode()


def analyze_crop(client, b64: str, prompt: str) -> dict:
    """Analiza un crop con Groq Vision."""
    import re
    last_err = ""
    for model in (VISION_MODEL,):  # solo Scout disponible
        for attempt in range(4):  # hasta 4 reintentos por rate limit
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "image_url",
                             "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                            {"type": "text", "text": prompt},
                        ]
                    }],
                    max_tokens=150,
                    temperature=0.05,
                    response_format={"type": "json_object"},
                )
                data = json.loads(resp.choices[0].message.content)
                data["model"] = model
                data.setdefault("has_anomaly", False)
                data.setdefault("anomaly_type", "unknown")
                data.setdefault("confidence", 0.0)
                data.setdefault("description", "")
                return data
            except Exception as e:
                last_err = str(e)
                # Rate limit: parsear tiempo de espera y dormir
                if "429" in last_err or "rate_limit" in last_err.lower():
                    wait_match = re.search(r"try again in (\d+)m([\d.]+)s", last_err)
                    if wait_match:
                        wait_sec = int(wait_match.group(1)) * 60 + float(wait_match.group(2)) + 5
                    else:
                        wait_sec = 60
                    tqdm.write(f"\n  [rate limit] Esperando {wait_sec:.0f}s antes de reintentar...")
                    time.sleep(wait_sec)
                    continue
                break  # otro tipo de error, no reintentar
    return {"has_anomaly": False, "anomaly_type": "error", "confidence": 0.0,
            "description": last_err, "model": None}


def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {}


def save_manifest(manifest: dict):
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                              encoding="utf-8")


def process_task(task: str, client, resume: bool, min_confidence: float, source_dir: Path = None):
    """Procesa todos los crops de una tarea (eyes o legs)."""
    normal_dir  = source_dir if source_dir else CROPS_DIR / task / "normal"
    anomaly_dir = CROPS_DIR / task / "anomaly"
    anomaly_dir.mkdir(parents=True, exist_ok=True)

    prompt = EYE_PROMPT if task == "eyes" else LEG_PROMPT
    images = sorted(normal_dir.glob("*.jpg")) + sorted(normal_dir.glob("*.png"))

    if not images:
        print(f"  [{task}] Sin imagenes en {normal_dir}")
        return

    manifest = load_manifest()
    task_key = task

    if task_key not in manifest:
        manifest[task_key] = {}

    # Skip already processed entries, but retry errors
    if resume:
        already_done = {k for k, v in manifest[task_key].items()
                        if v.get("anomaly_type") != "error"}
    else:
        already_done = set()
    pending = [p for p in images if p.name not in already_done]

    print(f"\n[{task}] Total: {len(images)} | Ya procesados: {len(already_done)} | Pendientes: {len(pending)}")

    normal_count  = sum(1 for v in manifest[task_key].values() if not v.get("has_anomaly"))
    anomaly_count = sum(1 for v in manifest[task_key].values() if v.get("has_anomaly"))

    for img_path in tqdm(pending, desc=f"Etiquetando {task}"):
        b64 = img_to_b64(img_path)
        if b64 is None:
            manifest[task_key][img_path.name] = {"has_anomaly": False, "error": "no_read"}
            continue

        result = analyze_crop(client, b64, prompt)
        manifest[task_key][img_path.name] = result

        if result.get("has_anomaly") and result.get("confidence", 0) >= min_confidence:
            dest = anomaly_dir / img_path.name
            shutil.move(str(img_path), str(dest))  # MOVER, no copiar: evita el mismo
            # crop en anomaly/ y normal/ a la vez (conflicto de etiqueta)
            anomaly_count += 1
            tqdm.write(f"  ANOMALIA: {img_path.name} -- {result.get('anomaly_type')} "
                       f"({result.get('confidence'):.0%}) -- {result.get('description','')[:60]}")
        else:
            normal_count += 1

        # Guardar cada 10 imagenes
        if len(manifest[task_key]) % 10 == 0:
            save_manifest(manifest)

        # Rate limit: ~15 req/min para vision en tier free
        time.sleep(4.5)

    save_manifest(manifest)
    print(f"\n  [{task}] Resultado: {normal_count} normales | {anomaly_count} con anomalia")
    print(f"  Crops anomaly en: {anomaly_dir}")
    print(f"  Manifest en: {MANIFEST_PATH}")
    return normal_count, anomaly_count


def main():
    parser = argparse.ArgumentParser(
        description="Auto-etiquetar crops con Groq Vision"
    )
    parser.add_argument("--task", choices=["eyes", "legs", "both"], default="both",
                        help="Tarea a procesar (default: both)")
    parser.add_argument("--resume", action="store_true",
                        help="Saltar crops ya procesados en el manifest")
    parser.add_argument("--min-confidence", type=float, default=0.70,
                        help="Confianza minima para clasificar como anomalia (default: 0.70)")
    parser.add_argument("--source_dir", type=str, default=None,
                        help="Directorio fuente alternativo (en vez del default crops/task/normal/)")
    args = parser.parse_args()

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("ERROR: GROQ_API_KEY no configurada en .env")
        sys.exit(1)

    try:
        from groq import Groq
    except ImportError:
        print("ERROR: pip install groq")
        sys.exit(1)

    client = Groq(api_key=api_key)
    tasks  = ["eyes", "legs"] if args.task == "both" else [args.task]

    print("=" * 60)
    print("  AlpacaVision AI -- Auto-etiquetado de crops con Vision")
    print(f"  Modelo: {VISION_MODEL}")
    print(f"  Min confidence: {args.min_confidence:.0%}")
    print(f"  Modo resume: {args.resume}")
    print("=" * 60)

    source_dir = Path(args.source_dir) if args.source_dir else None

    for task in tasks:
        src = source_dir if source_dir else None
        process_task(task, client, resume=args.resume,
                     min_confidence=args.min_confidence, source_dir=src)

    print("\nFinalizado. Proximos pasos:")
    print("  1. Revisar crops en data/crops/eyes/anomaly/ y data/crops/legs/anomaly/")
    print("     Eliminar los que esten mal clasificados (falsos positivos)")
    print("  2. Con >= 50 anomaly por clase, entrenar clasificadores:")
    print("     venv/Scripts/python.exe src/training/train_classifier.py \\")
    print("       --task eyes --data_dir data/crops/eyes --output models/classifier/eyes")
    print("     venv/Scripts/python.exe src/training/train_classifier.py \\")
    print("       --task legs --data_dir data/crops/legs --output models/classifier/legs")


if __name__ == "__main__":
    main()
