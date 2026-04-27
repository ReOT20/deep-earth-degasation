from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CandidateEvidence:
    landcover_branch: str
    moisture_anomaly: float = 0.0
    vegetation_stress: float = 0.0
    soil_brightness_bsi: float = 0.0
    thermal_anomaly: float = 0.0
    sar_anomaly: float = 0.0
    morphology: float = 0.0
    persistence: float = 0.0
    post_rain_drying: float = 0.0
    geology_context: float = 0.0
    canopy_structure: float = 0.0
    canopy_moisture: float = 0.0
    dem_support: float = 0.0
    cluster_adjacency: float = 0.0
    object_integrity: float = 0.0
    cropland_signal: float = 0.0
    forest_signal: float = 0.0
    false_positive_penalty: float = 0.0
    flags: list[str] = field(default_factory=list)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def score_cropland(e: CandidateEvidence) -> float:
    score = (
        0.20 * e.moisture_anomaly
        + 0.15 * e.vegetation_stress
        + 0.15 * e.soil_brightness_bsi
        + 0.10 * e.thermal_anomaly
        + 0.10 * e.sar_anomaly
        + 0.10 * e.morphology
        + 0.10 * e.persistence
        + 0.05 * e.post_rain_drying
        + 0.05 * e.geology_context
        - e.false_positive_penalty
    )
    return _clamp01(score)


def score_forest(e: CandidateEvidence) -> float:
    score = (
        0.25 * e.morphology
        + 0.15 * e.canopy_structure
        + 0.15 * e.canopy_moisture
        + 0.15 * e.dem_support
        + 0.10 * e.sar_anomaly
        + 0.10 * e.persistence
        + 0.05 * e.geology_context
        + 0.05 * e.cluster_adjacency
        - e.false_positive_penalty
    )
    return _clamp01(score)


def score_mixed(e: CandidateEvidence) -> float:
    score = (
        0.25 * e.object_integrity
        + 0.20 * e.morphology
        + 0.15 * e.cropland_signal
        + 0.15 * e.forest_signal
        + 0.10 * e.sar_anomaly
        + 0.10 * e.dem_support
        + 0.05 * e.geology_context
        - e.false_positive_penalty
    )
    return _clamp01(score)


def score_candidate(e: CandidateEvidence) -> float:
    branch = e.landcover_branch.lower()
    if branch == "cropland":
        return score_cropland(e)
    if branch == "forest":
        return score_forest(e)
    if branch == "mixed":
        return score_mixed(e)
    return _clamp01(
        0.25 * e.morphology
        + 0.25 * e.persistence
        + 0.25 * e.sar_anomaly
        + 0.25 * e.dem_support
        - e.false_positive_penalty
    )


def priority_class(score: float) -> str:
    if score >= 0.75:
        return "A"
    if score >= 0.55:
        return "B"
    if score >= 0.35:
        return "C"
    return "D"
