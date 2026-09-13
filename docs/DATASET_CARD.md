# AlpacaVision AI — Curated Alpaca Detection Dataset (v1.0)

**A curated, licence-audited, scene-deduplicated object-detection dataset of Peruvian
Altiplano alpacas (*Vicugna pacos*).**

- **Version:** 2.0
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Format:** YOLO (images + `.txt` bounding-box labels), single class `alpaca`
- **Total:** 1,460 images resolving to 742 distinct scenes · 1 class
- **Split (scene groups):** 519 / 111 / 112 → **images** 1,011 / 201 / 248

---

## Summary

This dataset consolidates alpaca imagery from public Roboflow Universe projects into a
deduplicated, licence-audited detection benchmark for the species. It was produced for the
AlpacaVision study (Universidad Nacional del Altiplano de Puno, Peru).

Version 2.0 supersedes v1.0 after an audit found that v1.0 was **not** free of leakage,
despite containing zero exact duplicates. Three corrections were applied:

1. **Licence filter.** One source project (591 images) declares `License: undefined` on
   Roboflow Universe. Absence of an express licence is not a grant of permission, so those
   images are removed. Every image in v2.0 comes from a source declaring CC BY 4.0.
2. **Perceptual deduplication.** Public projects re-encode images on export, so the same
   photograph appears under different byte content and survives cryptographic hashing.
3. **Dihedral invariance.** Three source projects ship three geometrically augmented copies
   (flips, 90-degree rotations) of every image. A plain perceptual hash maps these to
   unrelated codes, so the hash is minimised over the eight transforms of the dihedral group.

Under the v1.0 split, 43.5% of test images had a perceptual near-duplicate in training; under
the dihedral-invariant criterion the figure was **83.4%**. Users of v1.0 should treat any
metric computed on it as inflated.

## Contents & structure

```
annotated_v2/
├── images/
│   ├── train/   1,011 images  (519 scene groups)
│   ├── val/       201 images  (111 scene groups)
│   └── test/      248 images  (112 scene groups)
├── labels/                     one YOLO .txt per image
├── data.yaml
└── build_manifest.json         thresholds, seed, provenance, exclusions
```

- **Classes:** `0: alpaca` (single class).
- **Label format:** YOLO — one `.txt` per image; each line `class cx cy w h` (normalised).
- **Split:** whole scene groups, fixed seed = 42. The build script asserts that no
  near-duplicate pair crosses a split boundary.

## Provenance and licensing

| Source project | Images | Pre-augmented | Licence |
|---|:---:|:---:|---|
| `alpaca-ofxv9` | 681 | yes (3x) | CC BY 4.0 |
| `alpaca-5jmfl` | 312 | no | CC BY 4.0 |
| `alpaca-lls3s` | 294 | no | CC BY 4.0 |
| `alpaca-ibscl` | 64 | no | CC BY 4.0 |
| `alpaca-gkbmi` | 43 | yes (3x) | CC BY 4.0 |
| `alpaca-8baig` | 30 | no | CC BY 4.0 |
| `alpaca-nrzos` | 29 | no | CC BY 4.0 |
| `alpaca-epqna` | 7 | no | CC BY 4.0 |
| **Total retained** | **1,460** | | **CC BY 4.0** |
| `alpaca-xqfiw` | 591 | yes (3x) | **undefined — excluded** |
| `alpaca-zehtv` | — | yes (3x) | excluded, quality screening |

`alpaca-zehtv` was excluded before deduplication after quality screening (mean bounding-box
area ratio < 0.01). Supplementary imagery from iNaturalist (taxon 319688, *Vicugna pacos*,
Peruvian observations) was explored but never annotated and contributes **zero images**.

## Curation & preprocessing

1. **Cryptographic deduplication** — exact byte-level duplicates removed (1,037 of 3,088).
2. **Per-source licence filter** — 591 images removed (undefined licence).
3. **CLAHE** — applied to compensate for high ultraviolet irradiance and atmospheric haze
   typical of Andean altiplano photography.
4. **Dihedral-invariant perceptual grouping** — 1,460 images resolved into 742 scene groups
   at Hamming threshold τ = 6.
5. **Group-aware split** — whole groups assigned to a partition, seed = 42.

Reproduce with `scripts/build_dataset_v2.py` in the accompanying repository.

## Benchmark (reference)

A compact YOLOv11n detector (2.58M parameters) trained on this dataset reaches
**mAP@0.5 = 0.698 ± 0.024** over three seeds on the held-out test set; a larger YOLOv11s
reaches 0.704 ± 0.019, a difference smaller than the seed-to-seed spread.

> These figures are lower than those published with v1.0 (0.860). The difference is the
> removed contamination plus a smaller training set, not a regression in the model.

## Intended use & limitations

- **Intended use:** body-level alpaca detection/localisation; precision-livestock-farming
  research; transfer-learning source for camelid vision tasks.
- **Limitations:** single class (body only — no anatomical-part or anomaly labels); source
  imagery is globally sourced (not exclusively altiplano biome); not intended for clinical
  diagnosis.

## Citation

> Vilca-Solorzano, R.A.; Yana-Yucra, D.M.; Ccopa-Acero, C.D.; Aleman-Gonzales, L.
> *AlpacaVision AI: A Curated Dataset and Compact Detector for Altiplano Alpacas.*
> Universidad Nacional del Altiplano de Puno, 2026. Dataset, CC BY 4.0.

## Contact

John J. Hopfield Research Seedbed, Professional School of Statistical and Informatics
Engineering, Universidad Nacional del Altiplano de Puno (UNAP), Puno, Peru.
Corresponding: Leonid Aleman-Gonzales — laleman@unap.edu.pe
