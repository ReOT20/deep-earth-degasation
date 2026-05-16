# Config-Driven `run-mvp` Requirements

## Problem

The config model validates many sections, but some fields do not yet fully affect runtime behavior. This creates false confidence: a user can change configuration values and believe the pipeline changed when it did not.

## Principle

Every active config field must satisfy one of these conditions:

1. It changes runtime behavior.
2. It is explicitly documented as planned/deferred.
3. It is removed from the active config schema.

No active field should be silently ignored.

## Fields that must affect behavior

### Scoring

```text
scoring.priority_thresholds
scoring.cropland weights
scoring.penalties
```

Required behavior:

```text
changing scoring.cropland weights changes object_score/component scores
changing priority_thresholds changes priority_class
changing scoring.penalties changes false_positive_penalty/object_score
```

### False-positive filters

```text
false_positive_filters.use_penalties
```

Required behavior:

```text
use_penalties=true:
  false-positive flags subtract configured penalties

use_penalties=false:
  false-positive flags are still exported, but score penalty is zero
```

### Outputs

```text
outputs.candidates_top_n
outputs.export_geojson
outputs.export_csv
outputs.generate_passports
outputs.generate_quicklooks
outputs.export_time_series
outputs.export_validation_summary
```

Required behavior:

```text
export_geojson=false → do not write candidates.geojson
export_csv=false → do not write candidate_scores.csv unless required by another explicit output
candidates_top_n → limits review/passport/labeling exports where configured
generate_passports=false → do not write passport markdown
generate_quicklooks=false → do not write quicklook PNGs
export_time_series=false → do not write per-candidate time-series CSVs
export_validation_summary=false → do not write validation_summary.json
```

### Dynamic detector

```text
dynamic_detector.min_repeated_seasons
dynamic_detector.min_valid_observations_per_season
dynamic_detector.merge_across_dates
```

Required behavior:

```text
min_repeated_seasons controls persistence eligibility or priority caps
min_valid_observations_per_season controls missing-data flags and candidate eligibility
merge_across_dates controls whether repeated detections merge into stable objects
```

### Time windows

```text
time.vegetation_months
time.bare_soil_months
```

Required behavior:

```text
vegetation_months restrict vegetation-stress components
bare_soil_months restrict bare-soil brightness/BSI components
features outside their configured window are excluded or flagged as out-of-window
```

## Required tests

Add or update tests so each active config group has a behavioral assertion.

Minimum tests:

```text
test_scoring_weights_change_scores
test_priority_thresholds_change_classes
test_use_penalties_false_keeps_flags_without_penalty
test_output_toggles_control_artifacts
test_candidates_top_n_limits_review_exports
test_merge_across_dates_toggle_changes_candidate_count_or_ids
test_time_windows_exclude_out_of_window_components
```

## Acceptance criteria

```text
No active config key is silently ignored.
Unknown config keys still fail loudly.
Deferred keys are visibly marked as planned/deferred in docs or schema metadata.
All behavior-changing config tests pass.
```
