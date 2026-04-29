# deep-earth-degasation

Public implementation repository for the Deep Earth Degasation MVP.

## Current status

Initial public skeleton/scaffold. Core helpers, scoring primitives, a geometry-only static candidate extraction scaffold, public configuration, candidate artifact serialization, a static-candidates CLI path and tests are present. Raster ingestion, dynamic anomaly extraction, passport generation and training dataset preparation are not yet implemented.

The system does **not** directly detect molecular hydrogen. It produces candidate objects that require expert and/or field validation.

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
runs the static morphology scaffold, and writes `candidates.geojson` and
`candidate_scores.csv`. Input coordinates must already be projected in metres;
lon/lat GeoJSON such as EPSG:4326, CRS84 or WGS84 is rejected because automatic
reprojection is not implemented. The command does not ingest raster data, run
dynamic anomaly detection, directly detect H2, prove active degassing, or
perform field validation.

## Development checks

When changing code, configs or tests, run from this repository root:

    ruff check .
    ruff format --check .
    basedpyright
    pytest

Shortcut:

    make check

`make check` is the default verification path for agent work touching this public repository.
