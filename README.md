# deep-earth-degasation

Public implementation repository for the Deep Earth Degasation MVP.

## Current status

Initial public skeleton/scaffold. Core helpers, scoring primitives, a geometry-only static candidate extraction scaffold, public configuration and tests are present. Raster ingestion, full candidate extraction, passport generation and training dataset preparation are not yet implemented.

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

## Development checks

When changing code, configs or tests, run from this repository root:

    ruff check .
    ruff format --check .
    basedpyright
    pytest

Shortcut:

    make check

`make check` is the default verification path for agent work touching this public repository.
