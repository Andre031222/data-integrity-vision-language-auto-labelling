"""Assemble a blind veterinary-labeling batch from the real (non-augmented) eye crops."""
import csv, shutil, random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "crops" / "eyes"
OUT = ROOT / "data" / "vet_labeling" / "batch_01"

# collect ONLY real originals (skip augmentations aug_*)
items = []
for cls in ("normal", "anomaly"):
    for p in sorted((SRC / cls).glob("*.jpg")):
        if not p.name.startswith("aug_"):
            items.append((p, cls))

random.seed(42)
random.shuffle(items)

review = OUT / "00_POR_REVISAR"
for d in (review, OUT / "normal", OUT / "anomalia", OUT / "no_evaluable"):
    d.mkdir(parents=True, exist_ok=True)

key_rows, sheet_rows = [], []
for i, (src, llm) in enumerate(items, 1):
    new = f"eye_{i:04d}.jpg"
    shutil.copy2(src, review / new)
    key_rows.append({"id": f"{i:04d}", "archivo": new,
                     "archivo_original": src.name, "etiqueta_llm_previa": llm})
    sheet_rows.append({"id": f"{i:04d}", "archivo": new,
                       "diagnostico": "", "categoria": "", "calidad": "", "notas": ""})

with open(OUT / "_clave_interna.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["id", "archivo", "archivo_original", "etiqueta_llm_previa"])
    w.writeheader(); w.writerows(key_rows)

with open(OUT / "planilla.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["id", "archivo", "diagnostico", "categoria", "calidad", "notas"])
    w.writeheader(); w.writerows(sheet_rows)

n_norm = sum(1 for _, c in items if c == "normal")
n_anom = sum(1 for _, c in items if c == "anomaly")
print(f"Batch listo: {len(items)} crops en {review}")
print(f"  (prior LLM: {n_norm} normal / {n_anom} anomaly -- OCULTO al veterinario)")
print(f"  planilla.csv y _clave_interna.csv escritos en {OUT}")
