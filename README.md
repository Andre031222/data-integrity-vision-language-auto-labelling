# AlpacaVision: near-duplicate leakage and auto-label reliability

Code, audit tools and results for an openly licensed detection benchmark of alpacas and other
South American camelids, assembled from public Roboflow Universe projects and audited for
licence, label format, exact duplicates, near-duplicates and content.

[![DOI](https://img.shields.io/badge/dataset_DOI-10.5281%2Fzenodo.23579349-1d4ed8?style=flat-square)](https://doi.org/10.5281/zenodo.23579349)
[![License: MIT](https://img.shields.io/badge/code-MIT-15803d?style=flat-square)](LICENSE)
[![Dataset: CC BY 4.0](https://img.shields.io/badge/dataset-CC_BY_4.0-1d4ed8?style=flat-square)](#data)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776ab?style=flat-square)](https://www.python.org/)
[![Live demo](https://img.shields.io/badge/demo-paqocha.ginit.dev-0f2444?style=flat-square)](https://paqocha.ginit.dev)

Companion code for the manuscript *Near-Duplicate Leakage and Auto-Label Reliability in Image
Datasets Assembled From Public Sources: An Audited Benchmark for Alpaca Detection* (under review
at *IEEE Access*). The manuscript itself is not mirrored here.

## Summary

Datasets built from public image repositories, and labelled by machines, are cheap to make but
hard to evaluate. We audited our own pipeline and found that each standard check certified a
split that the next one showed to be contaminated:

| Notion of identity used to split | Test images with a copy in training |
|---|:---:|
| Bytes (SHA-256) | 0% by construction |
| Perceptual hash (DCT pHash) | 36.8% |
| pHash invariant to flips and 90-degree rotations | 75.9% |
| + source-photo identifier (cropped copies) | **82.3%** |

Shares measured on a random image-level split of the deduplicated, source-filtered corpus
(1461 images, 220 test images).

<p align="center"><img src="assets/fig_leakage.png" width="94%" alt="Detector under six constructions of the dataset"></p>

A YOLOv11n detector trained on six progressively stricter versions of the dataset falls from
mAP@0.5 = 0.941 to 0.852. Controls for training-set size, split variance and test composition
show that about half of the fall is the smaller training set and half is leakage; within each
leaky stage, test images with a training copy score 0.04 to 0.17 higher than images without one.

| Stage | Train | Test | mAP@0.5 | mAP@0.5:0.95 |
|---|:---:|:---:|:---:|:---:|
| 1. Raw | 2393 | 513 | 0.941 ± 0.003 | 0.838 |
| 2. + exact deduplication | 1664 | 358 | 0.915 ± 0.004 | 0.761 |
| 3. + source and content filters | 1022 | 220 | 0.898 ± 0.002 | 0.801 |
| 4. + perceptual hash | 1022 | 224 | 0.877 ± 0.003 | 0.711 |
| 5. + dihedral-invariant hash | 1023 | 228 | 0.822 ± 0.017 | 0.603 |
| **6. + source-photo identifier (released)** | 1035 | 211 | **0.852 ± 0.016** | **0.636** |
| 1. Raw, training cut to 1035 | 1035 | 513 | 0.895 ± 0.004 | 0.751 |
| 2. Dedup., training cut to 1035 | 1035 | 358 | 0.883 ± 0.002 | 0.718 |

Three training seeds per row. Over five random splits, stages 3, 4 and 5 keep their order in
every split, and stages 5 and 6 cannot be told apart (0.808 and 0.814). The released model scores
0.853 with a 95% interval of 0.80 to 0.91 computed over the 79 test photographs.

<p align="center"><img src="assets/fig_detections.jpg" width="94%" alt="Detections of the released model"></p>

## Ocular case study: an above-chance score that measures the labeller

Eye crops were labelled *normal* or *anomaly* by a vision–language model
(`llama-4-scout-17b-16e-instruct`) without expert review, and an EfficientNet-B2 was trained on
them after pre-training on public human fundus photographs (cataract versus normal, from a Kaggle compilation of IDRiD, ODIR and HRF images).

<p align="center"><img src="assets/fig_transfer.png" width="94%" alt="Two-stage classifier and label check"></p>

The classifier recovers the auto-labels above chance (AUC-ROC 0.654 ± 0.054 over ten seeds on
one split; 0.741 ± 0.049 over ten splits). Three authors then rated all 462 crops blind to the
labels: a majority found 92.4% of them not assessable (Fleiss' κ = 0.39), and of the 91 crops
labelled anomalous only 2 show a visible sign. No crop below 32 px was assessable, and the
largest crops do not frame an eye at all. The score measures agreement with the labeller, not
ocular pathology. Per-crop labels and ratings are in `outputs/figures/v31/ocular/`.

<p align="center"><img src="assets/fig_classifier.png" width="94%" alt="Classifier score and blind rating"></p>

## Data

<p align="center"><img src="assets/fig_dataset.png" width="94%" alt="Dataset composition"></p>

| Field | Value |
|---|---|
| DOI | [10.5281/zenodo.23579349](https://doi.org/10.5281/zenodo.23579349) (v3.1) |
| Images | 1461 (3419 consolidated − 1041 byte-identical copies − 917 excluded) |
| Boxes | 2537 |
| Distinct photographs | 517 |
| Split | 361 / 77 / 79 photographs = 1035 / 215 / 211 images |
| Excluded | 556 images without a licence, 322 annotating heads, 39 copies of 14 photographs that do not show a live camelid |
| Class | `alpaca`; the sources also label llamas and some wild camelids as alpaca |
| Licence | CC BY 4.0 |

Version 3.1 corrects version 3.0 (DOI 10.5281/zenodo.21134001); see
[`docs/release/CHANGELOG.md`](docs/release/CHANGELOG.md).

## Reproducing the study

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124
uv pip install --python .venv/bin/python -r requirements.txt imagehash open_clip_torch

.venv/bin/python scripts/build_dataset_v2.py --hash dihedral --source-id --out data/v3.1/s6photoid
scripts/run_revision.sh
.venv/bin/python scripts/summarise_revision.py
.venv/bin/python scripts/make_revision_figures.py
```

| Script | Purpose |
|---|---|
| `build_dataset_v2.py` | exact deduplication, source and content filters, grouping, split; asserts that no near-duplicate or source-photo identifier crosses a split |
| `audit_near_duplicates.py` | share of test images with a training copy under each notion of identity |
| `run_revision.sh` | every detector, classifier and benchmark run behind the manuscript |
| `analyse_contamination.py` | contaminated versus clean test images, group bootstrap intervals |
| `analyse_assessability.py` | agreement of the blind ratings of the eye crops |
| `compare_dedup_methods.py` | pHash versus CLIP: cost and recovery of known copies |
| `benchmark_latency.py` | CPU, GPU and ONNX latency |
| `package_release.py` | Zenodo archive with provenance and exclusions |

All reported numbers are collected in `outputs/figures/v31/summary.json`.

## Web application

A Flask dashboard runs the detector with an ONNX runtime. The ocular classifier is not used.

```bash
python run_webapp.py
```

## Authors

John J. Hopfield Research Seedbed, Faculty of Statistical and Informatics Engineering (FINESI),
Universidad Nacional del Altiplano, Puno, Peru. In the order of the manuscript: Dina Maribel
Yana-Yucra, Richar Andre Vilca-Solorzano (corresponding), Leonid Aleman-Gonzales, Cristian Daniel
Ccopa-Acero and Vladimiro Ibañez-Quispe.

## Citation

```bibtex
@dataset{yanayucra2026alpacavision_v31,
  title     = {AlpacaVision: a curated, licence-audited alpaca detection benchmark
               (Peruvian Altiplano), v3.1},
  author    = {Yana-Yucra, Dina Maribel and Vilca-Solorzano, Richar Andre and
               Aleman-Gonzales, Leonid and Ccopa-Acero, Cristian Daniel and
               Iba{\~n}ez-Quispe, Vladimiro},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.23579349}
}
```

## Licence

Code and documentation: MIT ([LICENSE](LICENSE)). Dataset: CC BY 4.0, via Zenodo.
