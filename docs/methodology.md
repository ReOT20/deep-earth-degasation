# Methodology overview

## Scientific positioning

This project detects and ranks candidate surface anomalies compatible with possible deep degassing. It does not directly detect H₂ and does not prove active degassing without validation.

## Current MVP approach

```text
land-cover-stratified candidate detection
```

The MVP combines static morphology of sub-circular/ring structures, dynamic anomalies on cropland, canopy/SAR/DEM anomalies in forested areas, object-level scoring and expert labeling preparation for weak/PU learning.

## Why land-cover stratification

Surface manifestations may differ by land cover:

- cropland: soil brightness, humus/soil exposure, crop water stress;
- forest: canopy structure, phenology, SAR backscatter, DEM depression;
- mixed landscapes: coherent ring objects spanning different surface types.

## Unit of analysis

The unit of analysis is a candidate object/polygon, not a pixel.

## Learning strategy

The first release uses rule-based and anomaly-detection baselines. Later training should use object-level weak supervision and positive-unlabeled learning. Random background should be treated as unlabeled, not as a reliable negative class.
