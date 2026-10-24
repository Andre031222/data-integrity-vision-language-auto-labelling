# AlpacaVision detection benchmark, v3.1

An openly licensed object-detection dataset of alpacas and other South American camelids,
assembled from public Roboflow Universe projects and audited for licence, label format,
exact duplicates, near-duplicates and content.

| | |
|---|---|
| Version | 3.1 (supersedes 3.0) |
| Licence | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| Format | YOLO detection: one image and one `.txt` label file per sample |
| Class | `0: alpaca` |
| Images | 1,461 (2,537 boxes) |
| Distinct photographs | 517 |
| Split | train / val / test = 361 / 77 / 79 photographs = 1,035 / 215 / 211 images |

## What the class covers

The source projects label alpacas, and also llamas and occasionally wild vicuñas and
guanacos, as `alpaca`. We kept these images because separating domestic camelids reliably
requires expert review of each animal. Treat the class as "South American camelid labelled
as alpaca by the source project". Fourteen photographs that do not show a live camelid
(camels, sheep, drawings, a decal and a sculpture) were removed.

## Contents

```
alpacavision-v3.1/
├── images/{train,val,test}/   1,035 / 215 / 211 images
├── labels/{train,val,test}/   YOLO labels: class cx cy w h (normalised)
├── data.yaml                  Ultralytics dataset file, relative paths
├── provenance.csv             per image: split, source project, source-photo id,
│                              photo group, number of boxes, SHA-256
├── exclusions.csv             every consolidated file not in the release, with the reason
├── build_manifest.json        build parameters (threshold, seed, exclusions)
├── CHANGELOG.md
└── LICENSE.txt
```

## How the dataset was built

1. **Consolidation.** 3,419 files from ten Roboflow Universe projects, single class.
2. **Label-format normalisation.** Two projects export segmentation polygons; each polygon
   is reduced to its enclosing box (1,001 annotations).
3. **Exact deduplication.** SHA-256 over file bytes; 1,041 byte-identical copies removed.
   Among copies, the one from a redistributable source is kept.
4. **Source filters.** `alpaca-xqfiw` (licence undefined, 556 images) and `alpaca-lls3s`
   (annotates heads, 322 images) are excluded.
5. **Content review.** 14 photographs that do not show a live camelid are excluded together
   with all their copies (39 files).
6. **Grouping.** Copies of one photograph are grouped by a 64-bit DCT perceptual hash made
   invariant to the eight flips and 90-degree rotations of the square (Hamming distance at
   most 6), and by the source-photo identifier that Roboflow keeps in exported file names,
   which also links cropped or rescaled copies. 1,461 images form 517 groups.
7. **Split.** Whole groups are assigned to train, validation and test (70/15/15, seed 42).
   No group and no source-photo identifier spans two partitions.

Images are distributed as exported by their source projects; no photometric preprocessing
was applied. The build is reproducible with `scripts/build_dataset_v2.py --hash dihedral
--source-id` in the code repository.

## Use the groups, not the images

Copies of the same photograph (flips, rotations, crops and re-encodings made by the source
platforms) are frequent. On a random per-image split of this corpus, 82.3% of test images
have a copy in training. Keep the released split, or split by the `photo_group` column of
`provenance.csv`, and compute confidence intervals over groups rather than images.

## Provenance

| Source project | Images | Disposition |
|---|---:|---|
| alpaca-ofxv9 | 666 | retained |
| alpaca-5jmfl | 336 | retained |
| alpaca-zehtv | 289 | retained, labels repaired from polygons |
| alpaca-ibscl | 65 | retained |
| alpaca-gkbmi | 40 | retained |
| alpaca-8baig | 31 | retained |
| alpaca-nrzos | 28 | retained |
| alpaca-epqna | 6 | retained |
| alpaca-xqfiw | 556 | excluded: licence undefined |
| alpaca-lls3s | 322 | excluded: annotates heads |
| 14 photographs | 39 | excluded: no live camelid |

Counts are after exact deduplication. Every retained project declares CC BY 4.0.

## Limitations

- One class only; no anatomical parts, individual identities or health labels.
- Global web imagery; Andean herds, breeds and seasons are unevenly represented.
- The class mixes alpacas with other South American camelids (see above).
- Fourteen non-camelid photographs were found in two review passes; a few may remain.
- Some byte-identical copies carried different boxes in different projects (56 groups); the
  release keeps the boxes of the retained copy.

## Citation

Please cite the dataset and the accompanying article:

D. M. Yana-Yucra, R. A. Vilca-Solorzano, L. Aleman-Gonzales, C. D. Ccopa-Acero and
V. Ibañez-Quispe, "AlpacaVision: a curated, licence-audited alpaca detection benchmark
(Peruvian Altiplano), v3.1," Zenodo, 2026.

Code: https://github.com/Andre031222/data-integrity-vision-language-auto-labelling

## Contact

John J. Hopfield Research Seedbed, Faculty of Statistical and Informatics Engineering
(FINESI), Universidad Nacional del Altiplano, Puno, Peru.
Corresponding author: Richar Andre Vilca-Solorzano, 75521963@est.unap.edu.pe
