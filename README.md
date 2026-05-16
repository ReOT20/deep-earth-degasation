# deep-earth-degasation

Public implementation repository for the Deep Earth Degasation MVP.

## Current status

Engineering status: cropland-first prepared-data dynamic MVP is implemented and
tested on synthetic fixtures.

Scientific/product status: alpha pilot candidate-ranking system. Private pilot
review is preliminary, forest/mixed branches remain deferred or missing-data
flagged, and outputs are candidate rankings for expert review rather than field
priority claims.

The system does **not** directly detect molecular hydrogen. It produces ranked
candidate surface anomalies compatible with possible deep degassing indicators
that require expert and/or field validation.

## MVP architecture

    Static SCD/Ring Detector
    + Dynamic Surface Anomaly Detector
    + Land-cover-specific Scoring
    + Weak/PU Learning Dataset Preparation

## Detection branches

- **Cropland:** field-normalized moisture, vegetation, brightness, thermal and SAR anomalies.
- **Forest:** ring morphology, canopy anomaly, SAR and DEM support.
- **Mixed landscapes:** candidates that remain coherent across land-cover boundaries.

## Geometry-only static artifacts

Run the public synthetic static candidate artifact path with:

    degasation static-candidates \
      --input examples/synthetic_candidates.geojson \
      --output-dir artifacts/static_demo \
      --config configs/lipetsk_voronezh_mvp.yaml

This command reads only supplied GeoJSON polygon or multipolygon geometries,
runs the static morphology scaffold, and writes:

- `candidates.geojson`
- `candidate_scores.csv`
- `passports/*.md`
- `labeling_table.csv`

Input coordinates must already be projected in metres; lon/lat GeoJSON such as
EPSG:4326, CRS84 or WGS84 is rejected because automatic reprojection is not
implemented. This static command is limited to supplied geometries; use
`degasation run-mvp` for prepared raster analysis. Neither command directly
detects H2, proves active degassing, or performs field validation.

## Prepared-data dynamic MVP

Run the public prepared-data dynamic path with a project config and prepared
stack manifest:

    degasation run-mvp \
      --config configs/lipetsk_voronezh_dynamic_mvp.template.yaml \
      --data-manifest data/manifests/prepared_stack_manifest.template.yaml \
      --output-dir artifacts/dynamic_demo

The dynamic path expects already-prepared analytical raster layers and projected
vector context. It can export anomaly maps, candidate objects, score tables,
passports, labeling tables, validation summaries and weak/PU learning tables.
It does not download raw satellite products, perform field validation, directly
detect H2 or prove active degassing.

TP09 guidance for refining this alpha path is documented under `docs/`.

## Development checks

When changing code, configs or tests, run from this repository root:

    ruff check .
    ruff format --check .
    basedpyright
    pytest

Shortcut:

    make check

`make check` is the default verification path for agent work touching this public repository.
