# TP09 Cropland Refinement Specification

## Purpose

TP09 turns the operational cropland-first prepared-data MVP into a more defensible pilot system.

The current dynamic MVP is operational, but still alpha. TP09 focuses on validation, segmentation, scoring calibration, context completeness, and stable review workflows.

## Correct status wording

```text
Engineering status:
  Cropland-first dynamic prepared-data MVP is operational.

Scientific/product status:
  Alpha pilot candidate-ranking system; not validated enough for field-priority claims.
```

## Output language guardrail

The system must not claim direct H2 detection from satellite imagery.

Use:

```text
ranked candidate surface anomalies compatible with possible deep degassing indicators,
requiring expert and field validation
```

Avoid:

```text
detected H2
detected active degassing
confirmed degassing
proved degassing
```

## Main TP09 engineering risks

1. Config values validate but do not always affect runtime behavior.
2. Candidates are still likely over-produced by per-feature connected components.
3. Sequential candidate IDs are unstable for QGIS review preservation.
4. Small, disconnected, huge, elongated, and broad field-patch objects can rank too high.
5. Land-cover assignment is not fully integrated into the end-to-end pipeline.
6. Known-site validation is not meaningful without a private seed layer.
7. Quicklook overlays can be wrong unless raster transforms are honored.
8. Post-rain/weather evidence exists in schema/helpers but is not fully end-to-end operational.

## Required TP09 public-repo changes

### 1. README correction

Update README to reflect implemented dynamic MVP functionality.

The README should explicitly say:

```text
Cropland-first prepared-data dynamic MVP is implemented.
Private pilot review is preliminary.
Forest/mixed branches remain deferred or missing-data flagged.
Outputs are candidate rankings, not direct H2 detections.
```

### 2. Config-driven runtime behavior

See:

```text
docs/config_driven_run_mvp_requirements.md
```

### 3. Stable candidate IDs and review migration

See:

```text
docs/stable_candidate_ids_and_label_migration.md
```

### 4. Composite-based segmentation

See:

```text
docs/composite_based_segmentation_requirements.md
```

## Acceptance criteria

TP09 public-repo work is complete when:

```text
pytest passes
README status is accurate
active config changes alter behavior or fail loudly if unsupported
candidate IDs are stable under deterministic reruns and row reordering
candidate generation is driven primarily by composite anomaly maps
small/huge/linear/broad-patch flags affect scoring or priority caps
land-cover branch assignment is integrated into run-mvp
quicklooks are transform-aware or disabled by default
validation summary checks CRS when known sites are supplied
```

## Recommended implementation order

```text
1. README/status correction
2. Config-driven scoring/output behavior
3. Stable candidate ID function
4. Label migration helper
5. Land-cover assignment integration
6. Tiny/huge/linear/broad-patch scoring caps
7. Composite-based candidate extraction
8. Quicklook transform handling or safe disable
9. Validation CRS checks and known-site metrics improvements
```
