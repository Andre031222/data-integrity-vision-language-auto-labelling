"""AlpacaVision AI -- Auto-etiquetado con Grounding DINO (API directa, GPU-optimizado)."""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# -- Configuración --------------------------------------------------------------
IMAGES_DIR  = Path("data/annotation_batch")
OUTPUT_DIR  = Path("data/auto_labels")
BOX_THRESH  = 0.25
TEXT_THRESH = 0.25
IMG_SIZE    = 800  # resolución de entrada para Grounding DINO

CLASS_NAMES = ["alpaca_body", "alpaca_head", "alpaca_eye", "alpaca_leg_front", "alpaca_leg_rear"]

# Prompt único -- Grounding DINO tokeniza cada frase separada por " . "
CAPTION = (
    "alpaca body . alpaca torso . "
    "alpaca head . alpaca face . "
    "alpaca eye . "
    "alpaca front leg . alpaca foreleg . "
    "alpaca rear leg . alpaca hind leg ."
)

# Mapeo: frase detectada -> class_id (coincidencia substring)
PHRASE_MAP = {
    "alpaca body":      0,
    "alpaca torso":     0,
    "alpaca head":      1,
    "alpaca face":      1,
    "alpaca eye":       2,
    "alpaca front leg": 3,
    "alpaca foreleg":   3,
    "alpaca rear leg":  4,
    "alpaca hind leg":  4,
}

MEAN = torch.tensor([0.485, 0.456, 0.406])
STD  = torch.tensor([0.229, 0.224, 0.225])


def load_gd_model(device: str):
    """Carga Grounding DINO SwinT-OGC y lo mueve a device UNA sola vez."""
    import os
    from groundingdino.util.inference import load_model

    cache = Path(os.path.expanduser("~")) / ".cache" / "autodistill" / "groundingdino"
    config_path  = cache / "GroundingDINO_SwinT_OGC.py"
    weights_path = cache / "groundingdino_swint_ogc.pth"

    if not config_path.exists():
        import groundingdino
        config_path = Path(groundingdino.__file__).parent / "config" / "GroundingDINO_SwinT_OGC.py"

    if not weights_path.exists():
        raise FileNotFoundError(
            f"Pesos no encontrados en {weights_path}. "
            "Corre primero: pip install autodistill-grounding-dino"
        )

    model = load_model(str(config_path), str(weights_path), device=device)
    model = model.to(device)
    model.eval()
    return model


def preprocess_image(img_path: Path, device: str) -> tuple:
    """Carga y normaliza imagen para Grounding DINO. Devuelve (tensor, orig_w, orig_h)."""
    from groundingdino.util.inference import load_image
    img_src, img_tensor = load_image(str(img_path))
    orig_h, orig_w = img_src.shape[:2]
    return img_tensor.to(device), orig_w, orig_h


def run_inference(model, img_tensor: torch.Tensor, caption: str, tokenizer) -> tuple:
    """Inferencia directa sin overhead de predict(). Devuelve (boxes_cxcywh, logits, phrases)."""
    from groundingdino.util.misc import clean_state_dict
    from groundingdino.util.vl_utils import create_positive_map_from_span

    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption += "."

    with torch.no_grad():
        outputs = model(img_tensor[None], captions=[caption])

    pred_logits = outputs["pred_logits"].cpu().sigmoid()[0]   # (nq, 256)
    pred_boxes  = outputs["pred_boxes"].cpu()[0]              # (nq, 4) cx,cy,w,h norm

    # Filtrar por box_threshold
    scores = pred_logits.max(dim=1).values
    mask   = scores > BOX_THRESH
    logits = pred_logits[mask]
    boxes  = pred_boxes[mask]

    # Decodificar frases
    tokenized = tokenizer(caption)
    phrases = []
    for logit_row in logits:
        token_ids  = (logit_row > TEXT_THRESH).nonzero(as_tuple=True)[0].tolist()
        tokens     = tokenizer.convert_ids_to_tokens(
            [tokenized["input_ids"][i] for i in token_ids if i < len(tokenized["input_ids"])]
        )
        phrase = tokenizer.convert_tokens_to_string(tokens).strip()
        phrases.append(phrase)

    confs = logits.max(dim=1).values

    return boxes, confs, phrases


def phrase_to_class(phrase: str) -> int:
    """Convierte frase detectada a class_id. Retorna -1 si no coincide."""
    p = phrase.lower().strip()
    # Coincidencia exacta primero
    if p in PHRASE_MAP:
        return PHRASE_MAP[p]
    # Coincidencia substring
    for key, cid in PHRASE_MAP.items():
        if key in p:
            return cid
    return -1


def boxes_to_yolo(boxes_cxcywh, confs, phrases) -> list[str]:
    """Convierte boxes normalizadas [cx,cy,w,h] + frases -> líneas YOLO."""
    lines = []
    for box, conf, phrase in zip(boxes_cxcywh, confs, phrases):
        cls_id = phrase_to_class(phrase)
        if cls_id < 0:
            continue
        cx, cy, w, h = box.tolist()
        if w <= 0 or h <= 0 or not (0 <= cx <= 1) or not (0 <= cy <= 1):
            continue
        lines.append(
            f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
            f"  # conf={float(conf):.2f} phrase={phrase}"
        )
    return lines


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo: {device.upper()}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    images = sorted(list(IMAGES_DIR.glob("*.jpg")) + list(IMAGES_DIR.glob("*.png")))
    print(f"Imágenes totales: {len(images)}")

    # Resume: saltar imágenes que ya tienen labels no vacíos
    pending = []
    skipped = 0
    for img_path in images:
        out_txt = OUTPUT_DIR / (img_path.stem + ".txt")
        if out_txt.exists() and out_txt.stat().st_size > 0:
            skipped += 1
        else:
            pending.append(img_path)

    if skipped:
        print(f"Resume: {skipped} ya etiquetadas (saltadas), {len(pending)} pendientes.")
    else:
        print(f"Imágenes a procesar: {len(pending)}")

    if not pending:
        print("Todas las imágenes ya están etiquetadas.")
        _rebuild_manifest(images, OUTPUT_DIR)
        return

    print("\nCargando Grounding DINO SwinT-OGC...")
    model = load_gd_model(device)
    tokenizer = model.tokenizer
    print("Modelo listo.")

    # Warm-up con primera imagen pendiente para inicializar CUDA kernels
    print("Warm-up GPU...", end=" ", flush=True)
    try:
        t0 = __import__("time").time()
        img0, _, _ = preprocess_image(pending[0], device)
        run_inference(model, img0, CAPTION, tokenizer)
        print(f"{__import__('time').time()-t0:.1f}s (warm-up)")
    except Exception as e:
        print(f"(warm-up falló: {e})")

    stats = {"total": len(images), "with_labels": skipped, "empty": 0, "errors": 0}

    print("\nGenerando pseudo-labels...")
    for img_path in tqdm(pending, desc="Auto-labeling"):
        out_txt = OUTPUT_DIR / (img_path.stem + ".txt")

        try:
            img_tensor, orig_w, orig_h = preprocess_image(img_path, device)
            boxes, confs, phrases = run_inference(model, img_tensor, CAPTION, tokenizer)
            lines = boxes_to_yolo(boxes, confs, phrases)
        except Exception as e:
            tqdm.write(f"ERROR {img_path.name}: {e}")
            lines = []
            stats["errors"] += 1

        if lines:
            out_txt.write_text("\n".join(lines))
            stats["with_labels"] += 1
        else:
            out_txt.write_text("")
            stats["empty"] += 1

    _rebuild_manifest(images, OUTPUT_DIR, stats)

    print(f"\n{'='*50}")
    print(f"Auto-labeling completado:")
    print(f"  Con detecciones: {stats['with_labels']}")
    print(f"  Sin detecciones: {stats['empty']}")
    print(f"  Errores:         {stats['errors']}")
    print(f"  Labels en:       {OUTPUT_DIR}/")
    print(f"\nSiguiente paso:")
    print(f"  python scripts/upload_labels_roboflow.py")


def _rebuild_manifest(images: list, output_dir: Path, stats: dict | None = None):
    """Reconstruye el manifest leyendo el estado real de los txt."""
    manifest = []
    with_labels = 0
    empty = 0
    for img_path in images:
        out_txt = output_dir / (img_path.stem + ".txt")
        boxes = 0
        if out_txt.exists():
            content = out_txt.read_text().strip()
            boxes = len([l for l in content.splitlines() if l.strip() and not l.startswith("#")])
            if boxes > 0:
                with_labels += 1
            else:
                empty += 1
        manifest.append({"filename": img_path.name, "label_file": out_txt.name, "boxes": boxes})

    if stats is None:
        stats = {"total": len(images), "with_labels": with_labels, "empty": empty, "errors": 0}

    manifest_path = output_dir / "auto_label_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({"stats": stats, "images": manifest}, f, indent=2)
    print(f"Manifest actualizado: {manifest_path}")


if __name__ == "__main__":
    main()
