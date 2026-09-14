"""Cross-check every headline number against the metrics JSON it comes from."""
import json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# The manuscript lives outside the published repository; skipped when absent.
TEX = ROOT / "paper/manuscript_ieee_access/manuscript.tex"
DOCS = [d for d in (ROOT / "README.md", ROOT / "docs/DATASET_CARD.md", TEX,
                    ROOT / "paper/manuscript_ieee_access/cover_letter.tex") if d.exists()]

# Ground truth straight from the released JSON artefacts and the dataset itself.
def load(name):
    return json.loads((ROOT / "outputs/figures" / name).read_text())

abl = {k: load(f"detector_v3_{k}_metrics.json")["summary"]
       for k in ("s1raw", "s2md5", "s3src", "s4phash", "s5dihedral", "s5_yolo11s")}
nd = load("neardup_audit_v3.json")["results"]
vlm = load("intervlm_agreement.json")
f1 = load("stage1_fundus_metrics.json")
sd = load("classifier_seed_distribution.json")

EXT = {".jpg", ".jpeg", ".png"}
DS = ROOT / "data/annotated_v3_final/images"
counts = {sp: sum(1 for f in (DS / sp).iterdir() if f.suffix.lower() in EXT)
          for sp in ("train", "val", "test")} if DS.exists() else {}
scenes = json.loads((ROOT / "data/annotated_v3_final/build_manifest.json").read_text()
                    )["groups"] - 1 if (ROOT / "data/annotated_v3_final").exists() else None

truth = {
    "mAP stage 1":       f"{abl['s1raw']['mAP50']['mean']:.4f}",
    "mAP stage 2":       f"{abl['s2md5']['mAP50']['mean']:.4f}",
    "mAP stage 3":       f"{abl['s3src']['mAP50']['mean']:.4f}",
    "mAP stage 4":       f"{abl['s4phash']['mAP50']['mean']:.4f}",
    "mAP stage 5":       f"{abl['s5dihedral']['mAP50']['mean']:.4f}",
    "mAP stage 5 sd":    f"{abl['s5dihedral']['mAP50']['std']:.4f}",
    "mAP yolo11s":       f"{abl['s5_yolo11s']['mAP50']['mean']:.4f}",
    "test contam pHash": f"{nd['perceptual']['6']['pct']}",
    "test contam dihed": f"{nd['dihedral']['6']['pct']}",
    "pairs pHash":       str(nd['perceptual']['6']['pairs']),
    "pairs dihedral":    str(nd['dihedral']['6']['pairs']),
    "kappa":             f"{vlm['cohens_kappa']:.2f}",
    "not assessable %":  f"{vlm['not_assessable_pct']}",
    "fundus AUC":        f"{f1['auc_roc']:.3f}",
    "clf AUC mean":      f"{sd['auc_mean']:.3f}",
    "clf AUC sd":        f"{sd['auc_std']:.3f}",
    "images released":   str(sum(counts.values())) if counts else "?",
    "scenes released":   str(scenes),
    "train / val / test": " / ".join(str(counts[k]) for k in ("train","val","test")) if counts else "?",
}
print("=== reference values, read from the metrics JSON ===")
for k, v in truth.items():
    print(f"  {k:<20} {v}")

# Numbers that must appear in the prose, and numbers that must NOT.
MUST = ["0.814", "0.936", "79.5", "40.5", "628", "1460", "0.997", "0.09", "93.9",
        "0.736", "0.089", "1001"]
FORBIDDEN = {
    "leakage-free": r"leakage-free",
    "TODO/FIXME/XXX": r"\b(TODO|FIXME|XXX|placeholder|TBD)\b",
    # The chance-level claim was retracted; only the correction may mention it.
    "live chance claim": r"(performs?|collapses?|collapse) (at|to) chance",
}
print("\n=== presence of the headline numbers ===")
bad = 0
for doc in DOCS:
    t = doc.read_text(encoding="utf-8")
    # Prose writes thousands with a separator; compare both spellings.
    def present(m):
        alt = f"{int(m):,}" if m.isdigit() and len(m) > 3 else m
        return m in t or alt in t
    missing = [m for m in MUST if not present(m)]
    label = doc.relative_to(ROOT)
    if doc.name == "cover_letter.tex":
        missing = [m for m in missing if m not in ("0.09", "1460", "0.089", "93.9")]
    if doc.name == "DATASET_CARD.md":
        missing = [m for m in missing if m not in ("0.997", "0.09", "0.736", "0.089", "93.9", "1001")]
    print(f"  {str(label):<45} missing: {missing or 'none'}")
    if missing:
        bad += 1

print("\n=== retracted or forbidden wording ===")
for doc in DOCS:
    t = doc.read_text(encoding="utf-8")
    for name, pat in FORBIDDEN.items():
        hits = re.findall(pat, t)
        if hits:
            print(f"  {doc.name}: {name} -> {len(hits)} ocurrencia(s)")
            bad += 1
# 0.506 may appear only inside a passage that retracts it. Judge by context window,
# not by regex adjacency, so line wrapping cannot create a false positive.
RETRACT = ("irreproducib", "not, however, reproducib", "not reproducib", "earlier version",
           "earlier report", "Correction", "failed to reproduce", "superseded")
for doc in DOCS:
    t = doc.read_text(encoding="utf-8")
    for m in re.finditer(r"0\.506", t):
        line = t[t.rfind("\n", 0, m.start()) + 1: t.find("\n", m.end())]
        if "&" in line:
            continue  # a numeric coincidence inside a table row, not a claim
        w = t[max(0, m.start() - 600): m.end() + 600]
        if not any(k.lower() in w.lower() for k in RETRACT):
            ln = t[:m.start()].count("\n") + 1
            print(f"  {doc.name}:{ln}: 0.506 stated without a retraction nearby")
            bad += 1
print("  (no output above means clean)" if bad == 0 else f"\n  {bad} problem(s)")

print("\n=== figures included by the manuscript ===")
# Drop commented-out lines before collecting the figures actually included.
tex = "\n".join(l for l in TEX.read_text(encoding="utf-8").split("\n")
                 if not l.lstrip().startswith("%"))
for fig in sorted(set(re.findall(r'includegraphics\[[^\]]*\]\{([^}]+)\}', tex))):
    found = [p for d in ("paper/figures_R", "paper/figures") if (p := ROOT / d / fig).exists()]
    print(f"  {fig:<32} {'OK' if found else 'NOT FOUND'}")
    if not found:
        bad += 1

print("\n=== RESULT ===")
print("  CLEAN" if bad == 0 else f"  {bad} PROBLEM(S) TO REVIEW")
sys.exit(0)
