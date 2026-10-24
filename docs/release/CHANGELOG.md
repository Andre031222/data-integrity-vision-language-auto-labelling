# Changelog

## v3.1 (2026)

Corrections found while revising the accompanying article.

- **Exact deduplication** now uses SHA-256 and a fixed rule for byte-identical copies: the
  copy from a redistributable source is kept. Under the earlier file-order rule, 37 files
  that a licensed project also distributes were kept as their unlicensed copy and then
  dropped by the licence filter. They are now retained.
- **Content review** of every distinct photograph removed 14 photographs that do not show a
  live camelid (six of camels, three of sheep, two drawings, a printed illustration, a
  vehicle decal and a sculpture), with all their copies (39 files). v3.0 had excluded one
  sheep photograph within one project only, so two of its copies and the other thirteen
  photographs remained in v3.0.
- **Grouping by source-photo identifier.** Roboflow keeps the name of the uploaded file in
  every export. Copies that a platform cropped or rescaled share that name but escape the
  perceptual hash. In the v3.0 split, such copies linked test images to training images.
  v3.1 groups them, and the build asserts that no identifier spans two partitions.
- **Documentation.** The v3.0 README stated that CLAHE had been applied to the images; it had
  not. Images are distributed as exported by their sources. `data.yaml` now uses relative
  paths. `provenance.csv` and `exclusions.csv` are new.

Result: 1,461 images in 517 distinct photographs (v3.0: 1,460 images in 628 scene groups).

## v3.0 (2026)

Polygon labels of two projects repaired to boxes; licence and annotation-target filters;
dihedral-invariant perceptual grouping. 1,460 images, 628 scene groups.
