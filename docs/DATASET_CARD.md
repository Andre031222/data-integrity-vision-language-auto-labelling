# AlpacaVision AI — Curated Alpaca Detection Dataset (v1.0)

**A curated, licence-audited, geometry-audited and scene-deduplicated object-detection
dataset of Peruvian Altiplano alpacas (*Vicugna pacos*).**

- **Version:** 3.0
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Format:** YOLO (images + `.txt` bounding-box labels), single class `alpaca`
- **Total:** 1,460 images resolving to 628 distinct scenes · 1 class
- **Split (scene groups):** 440 / 93 / 95 → **images** 1,022 / 214 / 224

---

## Summary

This dataset consolidates alpaca imagery from public Roboflow Universe projects into a
deduplicated, licence-audited detection benchmark. It was produced for the AlpacaVision
study (Universidad Nacional del Altiplano de Puno, Peru).

Version 3.0 supersedes 2.0 after an annotation audit found a **label-format fault**. Two
source projects export segmentation polygons rather than boxes, and the consolidation read
the first four values of every line as `(cx, cy, w, h)`. For a polygon those values are two
arbitrary vertices, so the resulting box is meaningless. The fault had two consequences:

1. One source (294 images, 523 boxes) entered v2.0 with boxes that do not contain the
   animal — 84.5% of them fail a geometry audit, against 0–2% for every box-format source.
2. The other polygon source had been **excluded by hand as "very low quality"**, because a
   screen that discarded projects with mean box area below 1% was in fact measuring the
   parser's garbage output. Its annotations are sound.

Parsing polygons as their enclosing box recovers **1,001 annotations** across the two
sources, none discarded.

Three filters then apply, each for a different reason:

| Filter | Removed | Reason |
|---|---:|---|
| Licence | 915 images | source declares `License: undefined` |
| Annotation target | 322 images | source annotates **heads** (median box area 0.071 vs 0.19–0.22) |
| Content | 3 images | one scene depicts **sheep**, found by human review |

## Contents & structure

```
annotated_v3_final/
├── images/
│   ├── train/   1,022 images  (440 scene groups)
│   ├── val/       214 images  ( 93 scene groups)
│   └── test/      224 images  ( 95 scene groups)
├── labels/                     one YOLO .txt per image
├── data.yaml
└── build_manifest.json         thresholds, seed, provenance, exclusions
```

- **Classes:** `0: alpaca` (single class).
- **Label format:** YOLO — one `.txt` per image; each line `class cx cy w h` (normalised).
- **Split:** whole scene groups, fixed seed = 42. The build script asserts that no
  near-duplicate pair crosses a split boundary.

## Provenance and licensing

| Source project | Images | Disposition |
|---|---:|---|
| `alpaca-ofxv9` | 684 | retained, CC BY 4.0 |
| `alpaca-5jmfl` | 313 | retained, CC BY 4.0 |
| `alpaca-zehtv` | 291 | retained, CC BY 4.0, **labels repaired from polygons** |
| `alpaca-ibscl` | 63 | retained, CC BY 4.0 |
| `alpaca-gkbmi` | 43 | retained, CC BY 4.0 |
| `alpaca-8baig` | 30 | retained, CC BY 4.0 |
| `alpaca-nrzos` | 29 | retained, CC BY 4.0 |
| `alpaca-epqna` | 7 | retained, CC BY 4.0 |
| **Total retained** | **1,460** | **CC BY 4.0** |
| `alpaca-xqfiw` | 915 | excluded: licence undefined |
| `alpaca-lls3s` | 322 | excluded: annotates heads, not whole animals |
| (one scene) | 3 | excluded: depicts sheep |

Supplementary imagery from iNaturalist (taxon 319688, *Vicugna pacos*, Peru) was explored but
never annotated and contributes **zero images**.

**Residual class noise:** human review of all 141 distinct scenes of `alpaca-zehtv` found one
depicting sheep (0.7%). It is excluded. We report the rate rather than implying the review
found nothing.

## Why scene groups, and not images

Splitting per image is not safe on this corpus. Audited at Hamming threshold 6 on the
deduplicated, source-filtered split (1,463 images, 220 test), which contains **zero** exact
duplicates:

| Notion of duplicate | Pairs | Test images with a training twin |
|---|---:|---:|
| Cryptographic (MD5) | — | 0 (0.0%) by construction |
| Perceptual (pHash) | 433 | 89 (**40.5%**) |
| **Dihedral-invariant pHash** | 1,137 | 175 (**79.5%**) |

Public projects re-encode images on export, which defeats cryptographic hashing, and three
of them ship three flipped or rotated copies of every image, which defeats a plain
perceptual hash. Only about one test image in five was genuinely unseen. Any metric computed
on a per-image split of this corpus should be treated as inflated.

## Curation & preprocessing

1. **Label-format normalisation** — polygons reduced to their enclosing box (1,001 recovered).
2. **Geometry audit** — coordinates in range, non-zero sides, box inside the image, aspect
   ratio below 10:1.
3. **Cryptographic deduplication** — 1,041 byte-identical files removed of 3,419.
4. **Source filters** — licence, annotation target, content.
5. **CLAHE** — compensates the high ultraviolet irradiance and haze of Andean photography.
6. **Dihedral-invariant perceptual grouping** — 1,460 images into 628 scene groups at τ = 6.
7. **Group-aware split** — whole groups assigned to a partition, seed = 42.

Reproduce with `scripts/consolidate_dataset.py` then `scripts/build_dataset_v2.py`.

## Benchmark (reference)

A compact YOLOv11n detector (2.58M parameters) trained on this dataset reaches
**mAP@0.5 = 0.814 ± 0.021** over three seeds on the held-out test set; a larger YOLOv11s
reaches 0.835 ± 0.009.

> Under a five-stage ablation the same detector scores 0.936 ± 0.002 on the raw corpus. The
> difference is removed contamination plus a smaller training set, not a regression.

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
