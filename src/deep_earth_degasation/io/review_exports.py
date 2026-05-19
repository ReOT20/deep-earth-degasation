from __future__ import annotations

import csv
import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path

from deep_earth_degasation.io.candidates import FALSE_POSITIVE_PROFILE_FIELDS

OBJECT_FEATURE_FIELDNAMES = [
    "rank",
    "candidate_id",
    "priority_class",
    "object_score",
    "static_score",
    "dynamic_score",
    "evidence_class",
    "morphology_type",
    "landcover_branch",
    "dominant_landcover_branch",
    "field_id",
    "area_m2",
    "diameter_m",
    "circularity",
    "elongation",
    "annulus_contrast",
    "ringness_score",
    "support_pixel_count",
    "field_residual_max",
    "peer_residual_max",
    "temporal_residual_max",
    "residual_types_present",
    "moisture_anomaly",
    "vegetation_stress",
    "soil_brightness_bsi",
    "thermal_anomaly",
    "sar_anomaly",
    "post_rain_drying",
    "persistence",
    "false_positive_penalty",
    *FALSE_POSITIVE_PROFILE_FIELDS,
    "false_positive_flags",
    "false_positive_profile",
    "missing_data_flags",
    "anomalous_dates",
    "source_feature_names",
    "source_layer_ids",
]

RANK_EXPLANATION_FIELDNAMES = [
    "rank",
    "candidate_id",
    "priority_class",
    "object_score",
    "dominant_evidence",
    "positive_evidence",
    "false_positive_evidence",
    "false_positive_distances",
    "missing_context",
    "score_components",
    "source_feature_names",
    "source_layer_ids",
    "explanation",
]

COMPONENT_FIELDS = [
    "object_score",
    "static_score",
    "dynamic_score",
    "moisture_anomaly",
    "vegetation_stress",
    "soil_brightness_bsi",
    "thermal_anomaly",
    "sar_anomaly",
    "post_rain_drying",
    "persistence",
    "annulus_contrast",
    "ringness_score",
    "support_pixel_count",
    "field_residual_max",
    "peer_residual_max",
    "temporal_residual_max",
    "false_positive_penalty",
    *FALSE_POSITIVE_PROFILE_FIELDS,
]

POSITIVE_EVIDENCE_FIELDS = [
    "dominant_evidence",
    "dynamic_score",
    "static_score",
    "moisture_anomaly",
    "vegetation_stress",
    "soil_brightness_bsi",
    "thermal_anomaly",
    "sar_anomaly",
    "post_rain_drying",
    "persistence",
    "annulus_contrast",
    "ringness_score",
    "support_pixel_count",
    "field_residual_max",
    "peer_residual_max",
    "temporal_residual_max",
]


def write_object_feature_table(score_rows: Iterable[Mapping[str, object]], path: Path) -> None:
    """Write a stable per-object feature snapshot for run comparison."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=OBJECT_FEATURE_FIELDNAMES)
        writer.writeheader()
        for score_row in score_rows:
            writer.writerow(object_feature_row(score_row))


def object_feature_row(score_row: Mapping[str, object]) -> dict[str, str]:
    """Return the feature-table row for one ranked candidate score row."""
    return {
        field_name: _text(score_row.get(field_name)) for field_name in OBJECT_FEATURE_FIELDNAMES
    }


def write_rank_explanations(score_rows: Iterable[Mapping[str, object]], path: Path) -> None:
    """Write reviewer-readable explanations for ranked candidate scores."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=RANK_EXPLANATION_FIELDNAMES)
        writer.writeheader()
        for score_row in score_rows:
            writer.writerow(rank_explanation_row(score_row))


def rank_explanation_row(score_row: Mapping[str, object]) -> dict[str, str]:
    """Return the rank-explanation row for one ranked candidate score row."""
    positive_evidence = _positive_evidence(score_row)
    false_positive_evidence = _false_positive_evidence(score_row)
    false_positive_distances = _false_positive_distances(score_row)
    missing_context = _missing_context(score_row)
    return {
        "rank": _text(score_row.get("rank")),
        "candidate_id": _text(score_row.get("candidate_id")),
        "priority_class": _text(score_row.get("priority_class")),
        "object_score": _text(score_row.get("object_score")),
        "dominant_evidence": _text(score_row.get("dominant_evidence")),
        "positive_evidence": positive_evidence,
        "false_positive_evidence": false_positive_evidence,
        "false_positive_distances": false_positive_distances,
        "missing_context": missing_context,
        "score_components": _score_components(score_row),
        "source_feature_names": _text(score_row.get("source_feature_names")),
        "source_layer_ids": _text(score_row.get("source_layer_ids")),
        "explanation": _explanation(
            score_row,
            positive_evidence,
            false_positive_evidence,
            false_positive_distances,
            missing_context,
        ),
    }


def _positive_evidence(score_row: Mapping[str, object]) -> str:
    evidence_parts: list[str] = []
    dominant = _text(score_row.get("dominant_evidence"))
    if dominant:
        evidence_parts.append(dominant)
    for field_name in POSITIVE_EVIDENCE_FIELDS:
        if field_name == "dominant_evidence":
            continue
        value = _finite_float(score_row.get(field_name))
        if value is not None and value > 0:
            evidence_parts.append(f"{field_name}={_format_number(value)}")
    return "; ".join(evidence_parts)


def _false_positive_evidence(score_row: Mapping[str, object]) -> str:
    parts: list[str] = []
    penalty = _finite_float(score_row.get("false_positive_penalty"))
    if penalty is not None and penalty > 0:
        parts.append(f"false_positive_penalty={_format_number(penalty)}")
    flags = _json_list(score_row.get("false_positive_flags"))
    if flags:
        parts.append("flags=" + "|".join(flags))
    profile = _text(score_row.get("false_positive_profile"))
    if profile and profile != "{}":
        parts.append("profile=" + profile)
    distances = _false_positive_distances(score_row)
    if distances:
        parts.append("distances=" + distances)
    return "; ".join(parts)


def _false_positive_distances(score_row: Mapping[str, object]) -> str:
    parts: list[str] = []
    for field_name in FALSE_POSITIVE_PROFILE_FIELDS:
        value = _finite_float(score_row.get(field_name))
        if value is not None:
            parts.append(f"{field_name}={_format_number(value)}")
    return "; ".join(parts)


def _missing_context(score_row: Mapping[str, object]) -> str:
    flags = [
        flag
        for flag in _json_list(score_row.get("missing_data_flags"))
        if flag.startswith("missing_") or flag.startswith("below_")
    ]
    return "|".join(flags)


def _score_components(score_row: Mapping[str, object]) -> str:
    components = {
        field_name: _text(score_row.get(field_name))
        for field_name in COMPONENT_FIELDS
        if _text(score_row.get(field_name))
    }
    return json.dumps(components, sort_keys=True)


def _explanation(
    score_row: Mapping[str, object],
    positive_evidence: str,
    false_positive_evidence: str,
    false_positive_distances: str,
    missing_context: str,
) -> str:
    parts = [
        "Rank "
        + _text(score_row.get("rank"))
        + " scored "
        + _text(score_row.get("object_score"))
        + " as priority "
        + _text(score_row.get("priority_class"))
        + "."
    ]
    if positive_evidence:
        parts.append("Main supporting evidence: " + positive_evidence + ".")
    if false_positive_evidence:
        parts.append("False-positive pressure: " + false_positive_evidence + ".")
    if false_positive_distances:
        parts.append("False-positive distances: " + false_positive_distances + ".")
    if missing_context:
        parts.append("Missing or limited context: " + missing_context + ".")
    parts.append("Candidate requires expert and field validation; this is not direct H2 detection.")
    return " ".join(parts)


def _json_list(value: object | None) -> list[str]:
    text = _text(value)
    if not text:
        return []
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        return [text]
    if isinstance(loaded, list):
        return [str(item) for item in loaded]
    return [str(loaded)]


def _finite_float(value: object | None) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool):
        number = float(value)
        return number if math.isfinite(number) else None
    text = _text(value)
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _format_number(value: float) -> str:
    return f"{value:.6g}"


def _text(value: object | None) -> str:
    return "" if value is None else str(value)
