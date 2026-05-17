from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import geopandas as gpd
from shapely.geometry.base import BaseGeometry

from deep_earth_degasation.config import PriorityThresholds, ScoreWeights, ScoringConfig
from deep_earth_degasation.morphology.shape import compute_shape_metrics

_STRONG_ANOMALY_ZSCORE = 3.0
_STATIC_ONLY_SCORE_CAP = 0.70
_FOREST_BRANCH_MISSING_FLAG = "missing_forest_branch_scoring"
_MIXED_BRANCH_MISSING_FLAG = "missing_mixed_branch_scoring"
_DEFAULT_CROPLAND_WEIGHTS = {
    "moisture_anomaly": 0.20,
    "vegetation_stress": 0.15,
    "soil_brightness_bsi": 0.15,
    "thermal_anomaly": 0.10,
    "sar_anomaly": 0.10,
    "morphology": 0.10,
    "persistence": 0.10,
    "post_rain_drying": 0.05,
    "geology_context": 0.05,
}
_DEFAULT_PRIORITY_THRESHOLDS = PriorityThresholds(A=0.75, B=0.55, C=0.35, D=0.0)


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


def score_cropland(e: CandidateEvidence, weights: ScoreWeights | None = None) -> float:
    score = (
        sum(
            weight * getattr(e, field_name)
            for field_name, weight in _cropland_weight_values(weights).items()
        )
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


def priority_class(score: float, thresholds: PriorityThresholds | None = None) -> str:
    active_thresholds = thresholds or _DEFAULT_PRIORITY_THRESHOLDS
    if score >= active_thresholds.A:
        return "A"
    if score >= active_thresholds.B:
        return "B"
    if score >= active_thresholds.C:
        return "C"
    return "D"


def score_candidate_objects(
    candidates: gpd.GeoDataFrame,
    scoring_config: ScoringConfig | None = None,
    *,
    min_repeated_seasons: int = 2,
) -> gpd.GeoDataFrame:
    """Add object-level review scores to static, dynamic or fused candidate objects."""
    output = candidates.copy()
    static_scores: list[float] = []
    object_scores: list[float] = []
    dynamic_scores: list[float] = []
    priority_classes: list[str] = []
    dominant_evidence_values: list[str] = []
    missing_data_flags_values: list[tuple[str, ...]] = []

    for _, row in output.iterrows():
        evidence_class = _text(_row_value(row, "evidence_class")) or "dynamic_only"
        landcover_branch = _landcover_branch(row)
        static_score = _score_value(_row_value(row, "static_score"))
        morphology_score = _morphology_score(row, static_score=static_score)
        dynamic_score = _dynamic_score(
            row,
            landcover_branch=landcover_branch,
            morphology_score=morphology_score,
            scoring_config=scoring_config,
            min_repeated_seasons=min_repeated_seasons,
        )
        penalty = _score_value(_row_value(row, "false_positive_penalty"))
        missing_data_flags = list(_tuple_value(_row_value(row, "missing_data_flags")))

        if evidence_class == "static_only":
            dynamic_score = 0.0
            missing_data_flags.append("missing_dynamic_evidence")

        if landcover_branch == "forest":
            missing_data_flags.append(_FOREST_BRANCH_MISSING_FLAG)
        elif landcover_branch == "mixed":
            missing_data_flags.append(_MIXED_BRANCH_MISSING_FLAG)

        object_score = _object_score(
            evidence_class=evidence_class,
            static_score=static_score,
            dynamic_score=dynamic_score,
            morphology_score=morphology_score,
            penalty=penalty,
        )
        object_score = _apply_dynamic_flag_caps(object_score, row, scoring_config)

        static_scores.append(static_score)
        object_scores.append(object_score)
        dynamic_scores.append(dynamic_score)
        priority_classes.append(
            priority_class(
                object_score,
                None if scoring_config is None else scoring_config.priority_thresholds,
            )
        )
        dominant_evidence_values.append(
            _dominant_evidence(
                evidence_class=evidence_class,
                static_score=static_score,
                dynamic_score=dynamic_score,
                landcover_branch=landcover_branch,
            )
        )
        missing_data_flags_values.append(tuple(sorted({str(flag) for flag in missing_data_flags})))

    output["static_score"] = static_scores
    output["dynamic_score"] = dynamic_scores
    output["object_score"] = object_scores
    output["priority_class"] = priority_classes
    output["dominant_evidence"] = dominant_evidence_values
    output["missing_data_flags"] = missing_data_flags_values
    return output


def _object_score(
    *,
    evidence_class: str,
    static_score: float,
    dynamic_score: float,
    morphology_score: float,
    penalty: float,
) -> float:
    if evidence_class == "static_only":
        base_score = min(static_score, _STATIC_ONLY_SCORE_CAP)
    elif evidence_class == "static_dynamic":
        base_score = 0.55 * dynamic_score + 0.45 * static_score
    elif evidence_class == "dynamic_only":
        base_score = 0.85 * dynamic_score + 0.15 * morphology_score
    else:
        base_score = 0.50 * dynamic_score + 0.50 * static_score
    return _clamp01(base_score - penalty)


def _apply_dynamic_flag_caps(
    object_score: float, row: Any, scoring_config: ScoringConfig | None
) -> float:
    flags = set(_tuple_value(_row_value(row, "dynamic_object_flags")))
    if "broad_patch" not in flags:
        return object_score
    thresholds = (
        _DEFAULT_PRIORITY_THRESHOLDS
        if scoring_config is None or scoring_config.priority_thresholds is None
        else scoring_config.priority_thresholds
    )
    return min(object_score, max(0.0, thresholds.B - 1.0e-6))


def _dynamic_score(
    row: Any,
    *,
    landcover_branch: str,
    morphology_score: float,
    scoring_config: ScoringConfig | None,
    min_repeated_seasons: int,
) -> float:
    component_values = _component_values(row)
    if landcover_branch != "cropland":
        return _generic_dynamic_score(row, component_values)

    if component_values:
        evidence = CandidateEvidence(
            landcover_branch=landcover_branch,
            moisture_anomaly=component_values.get("moisture", 0.0),
            vegetation_stress=component_values.get("vegetation", 0.0),
            soil_brightness_bsi=component_values.get("brightness", 0.0),
            thermal_anomaly=component_values.get("thermal", 0.0),
            sar_anomaly=component_values.get("sar", 0.0),
            morphology=morphology_score,
            persistence=_persistence_score(row, min_repeated_seasons=min_repeated_seasons),
            post_rain_drying=component_values.get("post_rain", 0.0),
            geology_context=component_values.get("geology", 0.0),
        )
        return score_cropland(
            evidence,
            None if scoring_config is None else scoring_config.cropland,
        )
    return _generic_dynamic_score(row, component_values)


def _cropland_weight_values(weights: ScoreWeights | None) -> dict[str, float]:
    if weights is None:
        return _DEFAULT_CROPLAND_WEIGHTS

    configured = {
        "moisture_anomaly": weights.moisture_weight,
        "vegetation_stress": weights.vegetation_weight,
        "soil_brightness_bsi": weights.brightness_weight,
        "thermal_anomaly": weights.thermal_weight,
        "sar_anomaly": weights.sar_weight,
        "morphology": weights.morphology_weight,
        "persistence": weights.persistence_weight,
        "post_rain_drying": weights.post_rain_weight,
        "geology_context": weights.geology_weight,
    }
    active = {
        field_name: float(weight) for field_name, weight in configured.items() if weight is not None
    }
    return active or _DEFAULT_CROPLAND_WEIGHTS


def _generic_dynamic_score(row: Any, component_values: dict[str, float]) -> float:
    if component_values:
        return max(component_values.values())
    return _normalized_anomaly(
        _row_value(row, "dynamic_max_anomaly") or _row_value(row, "max_anomaly")
    )


def _component_values(row: Any) -> dict[str, float]:
    per_feature_max = _row_value(row, "per_feature_max")
    if not isinstance(per_feature_max, dict):
        return {}

    components: dict[str, float] = {}
    for feature_name, value in per_feature_max.items():
        component = _feature_component(str(feature_name))
        if component is None:
            continue
        components[component] = max(components.get(component, 0.0), _normalized_anomaly(value))
    return components


def _morphology_score(row: Any, *, static_score: float) -> float:
    if static_score > 0:
        return static_score

    circularity = _score_value(_row_value(row, "circularity"))
    elongation = _score_value(_row_value(row, "elongation"))
    if circularity > 0 or elongation > 0:
        return _shape_support(circularity=circularity, elongation=elongation)

    geometry = _row_value(row, "geometry")
    if not isinstance(geometry, BaseGeometry):
        return 0.0
    metrics = compute_shape_metrics(geometry)
    return _shape_support(circularity=metrics.circularity, elongation=metrics.elongation)


def _shape_support(*, circularity: float, elongation: float) -> float:
    elongation_support = _clamp01((4.0 - max(elongation, 1.0)) / 3.0) if elongation > 0 else 0.0
    return _clamp01(0.70 * _clamp01(circularity) + 0.30 * elongation_support)


def _feature_component(feature_name: str) -> str | None:
    normalized = feature_name.lower()
    if any(token in normalized for token in ("ndmi", "ndwi", "msi", "moisture")):
        return "moisture"
    if any(token in normalized for token in ("ndvi", "red_edge", "red-edge", "vegetation")):
        return "vegetation"
    if any(token in normalized for token in ("bsi", "brightness", "bare_soil", "bare-soil")):
        return "brightness"
    if any(token in normalized for token in ("lst", "thermal", "temperature")):
        return "thermal"
    if any(token in normalized for token in ("sar", "vv", "vh")):
        return "sar"
    if "post_rain" in normalized or "drying" in normalized:
        return "post_rain"
    if "geology" in normalized:
        return "geology"
    return None


def _persistence_score(row: Any, *, min_repeated_seasons: int) -> float:
    target = max(1, int(min_repeated_seasons))
    repeated_seasons = _score_value(_row_value(row, "repeated_seasons"))
    if repeated_seasons > 0:
        return _clamp01(repeated_seasons / target)

    dates = _tuple_value(_row_value(row, "anomalous_dates"))
    seasons = {str(date)[:4] for date in dates if str(date)}
    if seasons:
        return _clamp01(len(seasons) / target)
    return 0.0


def _landcover_branch(row: Any) -> str:
    branch = _text(_row_value(row, "landcover_branch")).lower()
    if branch:
        return branch
    return "cropland"


def _dominant_evidence(
    *,
    evidence_class: str,
    static_score: float,
    dynamic_score: float,
    landcover_branch: str,
) -> str:
    if evidence_class == "static_dynamic":
        return f"{landcover_branch} dynamic anomaly with static morphology"
    if evidence_class == "static_only":
        return "static morphology"
    if dynamic_score > static_score:
        return f"{landcover_branch} dynamic anomaly"
    return evidence_class.replace("_", " ")


def _row_value(row: Any, key: str) -> object | None:
    if hasattr(row, "index") and key in row.index:
        return row[key]
    if isinstance(row, dict):
        return row.get(key)
    return None


def _tuple_value(value: object | None) -> tuple[object, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list | tuple | set):
        return tuple(value)
    return (value,)


def _score_value(value: object | None) -> float:
    if isinstance(value, int | float) and not isinstance(value, bool):
        number = float(value)
        return number if math.isfinite(number) else 0.0
    return 0.0


def _normalized_anomaly(value: object | None) -> float:
    return _clamp01(_score_value(value) / _STRONG_ANOMALY_ZSCORE)


def _text(value: object | None) -> str:
    if value is None:
        return ""
    return str(value)
