# Ocular crop ratings: criteria and columns

`ocular_crop_ratings.csv` holds one row per eye crop (462 crops, anonymised as eye_0001 to
eye_0462 in random order).

## Columns

| Column | Meaning |
|---|---|
| smaller_side_px | Smaller side of the crop in pixels |
| source | Source of the photograph (iNaturalist or Roboflow project) |
| vlm_label_llama4scout | Zero-shot label of `meta-llama/llama-4-scout-17b-16e-instruct`; prompt in `scripts/label_crops_with_vision.py` |
| second_screen_claude | Second machine screen. Crops with a smaller side below 32 px were set to `not_assessable` by a fixed rule; the rest were screened by Claude (Anthropic) with the criteria below |
| second_screen_quality | Image quality recorded in the second screen (poor, limited, sharp) |
| rater_X_assessable, rater_X_sign | Blind human rating by three authors (A, B, C) |

## Human rating criteria

Each rater worked alone, without the auto-labels and without discussing with the others, in a
browser tool that showed the crops enlarged and at native size.

- **Assessable**: an eye is recognisable, open, and the cornea and pupil are sharp enough to
  show opacity, discharge or a lesion if present.
- **Not assessable**: blurred, closed, covered by fibre or glare, no camelid eye in the crop
  (another animal, background, a person), or too small to see the surface of the eye. In
  doubt, not assessable.
- **Sign** (assessable crops only): `none`, `visible` (opacity, discharge, lesion or
  inflammation clearly visible) or `doubtful`.
