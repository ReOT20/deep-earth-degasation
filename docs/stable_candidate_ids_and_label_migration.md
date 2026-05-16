# Stable Candidate IDs and Review-Label Migration

## Problem

Dynamic candidates currently risk receiving order-dependent IDs such as:

```text
dynamic-object-000001
```

These IDs are fragile. A change in feature order, thresholding, merge behavior, or input dates can shift IDs. If QGIS review labels are preserved only by `candidate_id`, old labels can silently attach to different objects.

## Goal

Implement stable candidate IDs and a safe label-migration workflow.

## Stable ID requirements

Candidate IDs should be deterministic and based on durable object properties.

Recommended inputs:

```text
field_id
rounded centroid, e.g. nearest 10 m
rounded/normalized geometry hash
dominant evidence class or branch
dominant season/date bucket when useful
```

Recommended format:

```text
dyn-{field_id}-{centroid_10m_hash}-{geometry_hash8}
```

If `field_id` is unavailable:

```text
dyn-unknownfield-{centroid_10m_hash}-{geometry_hash8}
```

## ID stability expectations

The same candidate should keep the same ID when:

```text
input object rows are reordered
candidate sorting changes
export order changes
non-geometric metadata changes
```

The ID may change when:

```text
geometry changes materially
field assignment changes
candidate split/merge behavior changes the object footprint
```

When ID changes because geometry changed, label migration should prevent silent label loss or misattachment.

## Label migration helper

Implement migration using multiple matching signals, not ID alone.

Recommended matching hierarchy:

```text
1. exact candidate_id match and compatible geometry
2. same field_id + high IoU
3. same field_id + centroid distance below threshold + area ratio compatible
4. unmatched old labels exported as retired/review_needed
5. unmatched new candidates exported as new/unreviewed
```

Suggested thresholds for first implementation:

```text
high_iou_threshold: 0.60
centroid_distance_threshold_m: 30
area_ratio_min: 0.50
area_ratio_max: 2.00
```

## Outputs

Migration should produce:

```text
updated_labeling_table.csv
label_migration_report.csv
retired_labels.csv
new_candidates.csv
```

Public CLI helper:

```bash
degasation migrate-labels \
  --old-labels path/to/old_labeling_table.csv \
  --old-candidates path/to/old_candidates.geojson \
  --new-labels path/to/new_labeling_table.csv \
  --new-candidates path/to/new_candidates.geojson \
  --output-dir path/to/label_migration
```

`label_migration_report.csv` should include:

```text
old_candidate_id
new_candidate_id
match_method
iou
centroid_distance_m
area_ratio
old_label
migration_status
review_required
```

## Tests

Minimum tests:

```text
test_stable_ids_are_order_independent
test_stable_ids_change_for_material_geometry_change
test_label_migration_exact_id_match
test_label_migration_geometry_match_without_same_id
test_label_migration_retires_unmatched_old_label
test_label_migration_marks_split_or_merge_review_required
```

## Acceptance criteria

```text
Deterministic reruns preserve IDs for unchanged candidates.
QGIS labels are not silently attached to changed geometries.
Retired and ambiguous labels are exported for manual review.
```
