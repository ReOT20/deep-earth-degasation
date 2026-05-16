# Composite-Based Segmentation Requirements

## Problem

The dynamic MVP computes field-normalized anomaly components and a composite anomaly map, but candidate extraction may still behave like independent per-feature connected-component extraction followed by merging.

This can overproduce:

```text
small disconnected objects
partial rings
merged nearby objects
broad high-scoring field patches
feature-specific noise blobs
```

## Principle

Candidate generation should be driven primarily by the weighted composite anomaly map.

Per-feature layers should support explanation, zonal statistics, scoring, and diagnostics. They should not independently flood the candidate inventory unless an explicit diagnostic mode is selected.

## Target pipeline

```text
prepared raster stack
→ feature layers
→ field-normalized anomaly components
→ weighted composite anomaly map
→ field-local thresholding
→ connected components
→ morphological cleanup
→ split/merge logic
→ polygonization
→ zonal statistics from all component layers
→ object score
→ false-positive filters
→ ranking
```

## Field-local thresholding

Thresholding should happen within field boundaries, not globally across the AOI.

Supported modes:

```text
percentile: top N% anomaly within each field
absolute_z: anomaly score above configured z threshold
hybrid: percentile with absolute minimum anomaly
```

## Object cleanup

Implement or strengthen:

```text
minimum support-pixel count
minimum area
maximum area / broad-patch cap
hole/ring preservation where possible
small-object removal
linear stripe suppression
nearby object separation where evidence supports separation
```

## Ring and annulus support

Ringness should not depend only on polygon holes.

Add annulus-style evidence when raster support exists:

```text
annulus anomaly > center anomaly and local background
```

Store ring evidence separately from geometry-only circularity:

```text
circularity
elongation
polygon_has_hole
annulus_contrast
ringness_score
```

## Broad-patch suppression

Large high-scoring field patches should not automatically rank high unless there is independent morphology or persistence evidence.

Recommended rule:

```text
if area is high and circularity/ringness is weak and anomaly is diffuse:
    apply broad_patch penalty or cap priority at C
```

## Tests

Minimum synthetic tests:

```text
test_composite_blob_becomes_one_candidate
test_single_feature_noise_does_not_flood_inventory
test_linear_stripe_is_flagged_or_removed
test_nearby_blobs_can_remain_separate
test_partial_ring_has_annulus_evidence
test_broad_field_patch_is_capped_or_penalized
test_min_support_pixels_caps_tiny_candidate
```

## Acceptance criteria

```text
Candidate extraction uses composite anomaly maps by default.
Diagnostic per-feature extraction is optional and clearly labeled.
Small-object rate decreases in private pilot comparison.
Partial rings are less fragmented.
Nearby objects are not over-merged.
Broad field patches without morphology are downgraded.
```
