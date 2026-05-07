# Prepared data contract for dynamic MVP

## Purpose

The dynamic MVP is intentionally scoped around prepared local data. This avoids making satellite download, cloud preprocessing and external APIs the first blocker.

## Manifest shape

Use `data/manifests/prepared_stack_manifest.template.yaml` as the starter format.

Minimum manifest sections:

```text
run
crs
resolution_m
aoi
vectors
raster_layers
quality
notes
```

## Raster layer requirements

Each raster layer must have:

| Field | Required | Meaning |
|---|---:|---|
| `id` | yes | Stable layer ID unique within the manifest. |
| `sensor` | yes | `sentinel2`, `sentinel1`, `landsat`, `dem`, `weather`, or `derived`. |
| `feature_name` | yes | Feature consumed by the pipeline, for example `NDMI`, `VV`, `LST`. |
| `date` | yes for dynamic layers | ISO date or date range. |
| `path` | yes | Local path relative to manifest or repository root. |
| `crs` | yes | CRS identifier, expected to match configured metric CRS unless resampling/reprojection is explicit. |
| `resolution_m` | yes | Pixel size in metres. |
| `nodata` | yes | Nodata value or null. |
| `quality_mask_path` | optional | Mask where invalid pixels are excluded. |
| `role` | yes | `input_feature`, `quality_mask`, `context`, or `derived`. |

## Vector layer requirements

Each vector layer should declare:

```text
path
crs
role
required
id_field optional
```

Recommended roles:

```text
aoi
fields
landcover
roads
water
built_up
settlements
excluded_zones
geology
lineaments
known_sites
```

## CRS policy

Metric operations require a projected CRS in metres. Lon/lat inputs must be rejected unless an explicit reprojection target and reprojection setting are provided.

## Config policy

Public YAML configs are strict. Unknown keys fail validation unless they are inside an explicit `metadata` or `provenance` map.

The current geometry-only static baseline keeps `object_constraints.max_diameter_m: 1500` because the first private static pilot and existing static examples were reviewed with that broader limit. The dynamic prepared-data template uses `max_diameter_m: 1000` as a stricter first-pass review default for field-normalized anomaly objects. This is a phase-specific threshold decision, not a conflicting unit convention.

## Data privacy policy

Private AOIs, field boundaries, QGIS seed geometries and generated private pilot outputs should remain in the control-plane/private repository or ignored output directories. Public examples must use synthetic coordinates or non-sensitive public samples only.
