# Dynamic MVP pipeline specification

## Scope

This specification defines the next MVP after the current geometry-only static artifact pipeline.

The MVP is a prepared-data pipeline. It does not need to download satellite data automatically. It consumes prepared local raster/time-series inputs and vector context layers, then produces ranked candidate objects and review artifacts.

## Guardrails

- Do not claim direct H2 detection.
- Do not treat a dry patch, ring shape or bright soil patch as proof of degassing.
- Treat outputs as ranked candidates for expert and field review.
- Treat random background as unlabeled, not as a reliable negative class.
- Preserve source context separately from detector-derived metrics.

## Inputs

### Required

```text
config.yaml
prepared raster stack manifest
AOI vector
field or cropland-mask vector
```

### Optional but recommended

```text
roads vector
water vector
built-up vector
settlement vector
quarry/excluded-zone vector
DEM derivatives
fault/lineament/geology context
weather/rainfall context
known-site seed objects
```

## Prepared raster stack contract

Each raster layer in the manifest should declare:

```text
id
sensor
feature_name
date
path
crs
resolution_m
nodata
quality_mask_path optional
role
```

The MVP implementation may support GeoTIFF first. Zarr/NetCDF can be added later.

## Pipeline stages

```text
1. Load and validate config.
2. Load AOI, field polygons and context vectors.
3. Load prepared raster stack from manifest.
4. Validate CRS, transform, resolution, nodata and masks.
5. Clip vectors and rasters to AOI.
6. Compute or normalize feature layers.
7. Assign pixels/objects to fields and land-cover branches.
8. Compute field-level robust z-score anomaly components.
9. Combine anomaly components into a configured anomaly score map.
10. Threshold anomaly maps inside fields.
11. Extract connected components.
12. Polygonize components.
13. Compute object morphology and zonal feature statistics.
14. Merge repeated detections across dates/seasons.
15. Apply false-positive flags and penalties.
16. Score and rank candidates.
17. Export artifacts, passports, quicklooks and validation summary.
18. Export reviewed/labelable object table for weak/PU learning.
```

## Primary cropland anomaly components

```text
moisture_anomaly
vegetation_stress_anomaly
soil_brightness_anomaly
thermal_anomaly
sar_anomaly
post_rain_drying
persistence
morphology
geology_context
false_positive_penalty
```

## Object score fields

Minimum score table fields:

```text
run_id
candidate_id
rank
priority_class
object_score
static_score
dynamic_score
landcover_branch
dominant_landcover_branch
area_m2
diameter_m
circularity
elongation
ringness
field_id
distance_to_field_edge_m
moisture_anomaly
vegetation_stress
soil_brightness_bsi
thermal_anomaly
sar_anomaly
persistence
post_rain_drying
geology_context
false_positive_penalty
dominant_evidence
false_positive_flags
missing_data_flags
passport_path
```

## Priority classes

```text
A: high-priority candidate; strong multisensor evidence and no obvious false-positive cause.
B: medium-priority candidate; several signals but incomplete or ambiguous evidence.
C: weak candidate; limited evidence or low persistence.
D: probable false positive; retained for audit and hard-negative learning.
U: unknown; insufficient data or unresolved context.
```

## Non-goals for this MVP

- automatic global satellite ingestion;
- direct gas detection;
- final scientific confirmation;
- neural network segmentation;
- production user interface;
- replacing expert review.
