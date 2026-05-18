from __future__ import annotations

import geopandas as gpd
from shapely.geometry import box

from deep_earth_degasation.scoring import (
    CandidateEvidence,
    priority_class,
    score_candidate,
    score_candidate_objects,
    score_cropland,
    score_forest,
    score_mixed,
)


def test_cropland_score_is_clamped_to_unit_interval() -> None:
    evidence = CandidateEvidence(
        landcover_branch="cropland",
        moisture_anomaly=1.0,
        vegetation_stress=1.0,
        soil_brightness_bsi=1.0,
        thermal_anomaly=1.0,
        sar_anomaly=1.0,
        morphology=1.0,
        persistence=1.0,
        post_rain_drying=1.0,
        geology_context=1.0,
    )

    score = score_cropland(evidence)

    assert 0.0 <= score <= 1.0


def test_false_positive_penalty_lowers_score() -> None:
    base = CandidateEvidence(
        landcover_branch="cropland",
        moisture_anomaly=1.0,
        vegetation_stress=1.0,
        soil_brightness_bsi=1.0,
        morphology=1.0,
        persistence=1.0,
    )
    penalized = CandidateEvidence(
        landcover_branch="cropland",
        moisture_anomaly=1.0,
        vegetation_stress=1.0,
        soil_brightness_bsi=1.0,
        morphology=1.0,
        persistence=1.0,
        false_positive_penalty=0.5,
    )

    assert score_cropland(penalized) < score_cropland(base)


def test_forest_score_uses_forest_evidence() -> None:
    evidence = CandidateEvidence(
        landcover_branch="forest",
        morphology=1.0,
        canopy_structure=1.0,
        canopy_moisture=1.0,
        dem_support=1.0,
        sar_anomaly=1.0,
        persistence=1.0,
    )

    score = score_forest(evidence)

    assert 0.0 <= score <= 1.0
    assert score > 0.5


def test_mixed_score_uses_mixed_evidence() -> None:
    evidence = CandidateEvidence(
        landcover_branch="mixed",
        object_integrity=1.0,
        morphology=1.0,
        cropland_signal=0.8,
        forest_signal=0.7,
        sar_anomaly=0.5,
        dem_support=0.5,
    )

    score = score_mixed(evidence)

    assert 0.0 <= score <= 1.0
    assert score > 0.5


def test_score_candidate_dispatches_by_branch() -> None:
    evidence = CandidateEvidence(landcover_branch="forest", morphology=1.0, dem_support=1.0)

    assert score_candidate(evidence) == score_forest(evidence)


def test_priority_class_thresholds() -> None:
    assert priority_class(0.75) == "A"
    assert priority_class(0.55) == "B"
    assert priority_class(0.35) == "C"
    assert priority_class(0.34) == "D"


def test_tiny_dynamic_objects_are_capped_without_strong_support() -> None:
    candidates = _dynamic_candidates(
        support_pixel_count=9,
        dynamic_object_flags=("too_small",),
        source_feature_names=("NDMI", "NDVI"),
        repeated_seasons=1,
    )

    scored = score_candidate_objects(candidates)

    assert scored.iloc[0]["object_score"] < 0.35
    assert scored.iloc[0]["priority_class"] == "D"


def test_tiny_dynamic_objects_can_survive_with_strong_multisensor_support() -> None:
    candidates = _dynamic_candidates(
        support_pixel_count=9,
        dynamic_object_flags=("too_small",),
        source_feature_names=("NDMI", "NDVI", "BSI"),
        repeated_seasons=2,
    )

    scored = score_candidate_objects(candidates)

    assert scored.iloc[0]["object_score"] >= 0.35


def test_large_broad_and_elongated_dynamic_objects_are_capped() -> None:
    scored = score_candidate_objects(
        _dynamic_candidates(
            support_pixel_count=80,
            dynamic_object_flags=("too_large", "broad_patch", "elongated"),
            source_feature_names=("NDMI", "NDVI", "BSI"),
            repeated_seasons=2,
        )
    )

    assert scored.iloc[0]["object_score"] < 0.55
    assert scored.iloc[0]["priority_class"] in {"C", "D"}


def _dynamic_candidates(
    *,
    support_pixel_count: int,
    dynamic_object_flags: tuple[str, ...],
    source_feature_names: tuple[str, ...],
    repeated_seasons: int,
) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "candidate_id": ["candidate-a"],
            "evidence_class": ["dynamic_only"],
            "landcover_branch": ["cropland"],
            "per_feature_max": [{"NDMI": 3.0, "NDVI": 3.0, "BSI": 3.0}],
            "source_feature_names": [source_feature_names],
            "repeated_seasons": [repeated_seasons],
            "dynamic_object_flags": [dynamic_object_flags],
            "support_pixel_count": [support_pixel_count],
            "false_positive_flags": [()],
            "missing_data_flags": [()],
            "geometry": [box(0, 0, 30, 30)],
        },
        geometry="geometry",
        crs="EPSG:32637",
    )
