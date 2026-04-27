from __future__ import annotations

from deep_earth_degasation.scoring import (
    CandidateEvidence,
    priority_class,
    score_candidate,
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
