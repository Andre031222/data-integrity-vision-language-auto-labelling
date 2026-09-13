"""AlpacaVision AI -- Resolucion de conflictos de etiqueta en crops de ojos."""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
load_dotenv(ROOT / ".env")

from label_crops_with_vision import analyze_crop, img_to_b64, EYE_PROMPT, LEG_PROMPT  # noqa: E402


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def find_conflicts(task: str):
    """Devuelve lista de (anomaly_path, normal_path) con MD5 identico (originales)."""
    A = ROOT / "data" / "crops" / task / "anomaly"
    N = ROOT / "data" / "crops" / task / "normal"
    def originals(d):
        return [p for p in d.iterdir()
                if p.suffix.lower() in (".jpg", ".jpeg", ".png") and not p.name.startswith("aug_")]
    a = {md5(p): p for p in originals(A)}
    n = {md5(p): p for p in originals(N)}
    return [(a[h], n[h]) for h in (set(a) & set(n))], A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["eyes", "legs"], default="eyes")
    ap.add_argument("--min-confidence", type=float, default=0.70)
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--sleep", type=float, default=4.5)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("ERROR: GROQ_API_KEY no configurada en .env")
        sys.exit(1)
    from groq import Groq
    client = Groq(api_key=api_key)
    prompt = EYE_PROMPT if args.task == "eyes" else LEG_PROMPT

    conflicts, anomaly_dir = find_conflicts(args.task)
    print(f"[{args.task}] crops en conflicto (anomaly==normal por MD5): {len(conflicts)}")
    if not conflicts:
        return

    manifest = {}
    to_anomaly = to_normal = 0

    for i, (apath, npath) in enumerate(sorted(conflicts, key=lambda t: t[0].name), 1):
        b64 = img_to_b64(apath)
        votes = []
        for _ in range(args.passes):
            r = analyze_crop(client, b64, prompt)
            votes.append(r)
            time.sleep(args.sleep)
        confs = [v.get("confidence", 0.0) for v in votes]
        anomaly_votes = [v.get("has_anomaly", False) for v in votes]
        mean_conf = sum(confs) / len(confs) if confs else 0.0
        is_anomaly = all(anomaly_votes) and mean_conf >= args.min_confidence
        decision = "anomaly" if is_anomaly else "normal"

        manifest[apath.name] = {
            "decision": decision,
            "mean_confidence": round(mean_conf, 3),
            "votes": [{"has_anomaly": v.get("has_anomaly"),
                       "anomaly_type": v.get("anomaly_type"),
                       "confidence": v.get("confidence")} for v in votes],
        }

        if decision == "anomaly":
            to_anomaly += 1
            if not args.dry_run and npath.exists():
                npath.unlink()  # quitar copia contradictoria de normal/
        else:
            to_normal += 1
            if not args.dry_run:
                if apath.exists():
                    apath.unlink()  # quitar copia erronea de anomaly/
                # borrar augmentadas derivadas de este original
                for aug in anomaly_dir.glob(f"aug_*_{apath.name}"):
                    aug.unlink()
        print(f"  [{i}/{len(conflicts)}] {decision.upper():7s} conf={mean_conf:.2f} "
              f"votes={[int(x) for x in anomaly_votes]} {apath.name[:48]}")

    out = ROOT / "data" / "crops" / f"conflict_resolution_manifest_{args.task}.json"
    if not args.dry_run:
        out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"  RESUELTO [{args.task}] {'(DRY-RUN, sin cambios)' if args.dry_run else ''}")
    print(f"  -> anomaly (consenso): {to_anomaly}")
    print(f"  -> normal  (revertido): {to_normal}")
    print(f"  Manifest: {out.name}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
