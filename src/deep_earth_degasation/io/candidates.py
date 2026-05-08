from __future__ import annotations

import csv
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shapely.geometry import mapping

from deep_earth_degasation.morphology.static_detector import StaticCandidate
from deep_earth_degasation.scoring import priority_class

SourceProperties = list[dict[str, object]]


@dataclass(frozen=True)
class CandidateArtifactPaths:
    candidates_geojson: Path
    candidate_scores_csv: Path


SOURCE_CONTEXT_FIELD_MAP = {
    "landcover_context": "source_landcover_context",
    "morphology_type": "source_morphology_type",
    "false_positive_risk": "source_false_positive_risk",
    "notes": "source_notes",
}

SCORE_FIELDNAMES = [
    "rank",
    "priority_class",
    "candidate_id",
    "morphology_type",
    "source_landcover_context",
    "source_morphology_type",
    "source_false_positive_risk",
    "source_notes",
    "static_score",
    "dynamic_score",
    "area_m2",
    "diameter_m",
    "circularity",
    "elongation",
    "evidence",
    "flags",
    "dominant_evidence",
    "false_positive_flags",
]


def static_candidate_to_feature(
    candidate: StaticCandidate, source_properties: dict[str, object] | None = None
) -> dict[str, Any]:
    metrics = candidate.shape_metrics
    return {
        "type": "Feature",
        "properties": {
            "candidate_id": candidate.candidate_id,
            "morphology_type": candidate.morphology_type,
            **_source_context(source_properties),
            "static_score": candidate.static_score,
            "area_m2": metrics.area,
            "diameter_m": metrics.equivalent_diameter,
            "circularity": metrics.circularity,
            "elongation": metrics.elongation,
            "evidence": _sorted_evidence(candidate),
            "flags": sorted(candidate.flags),
            "false_positive_flags": _false_positive_flags(sorted(candidate.flags)),
        },
        "geometry": mapping(candidate.geometry),
    }


def static_candidate_to_score_row(
    candidate: StaticCandidate,
    rank: int,
    source_properties: dict[str, object] | None = None,
) -> dict[str, str | int | float]:
    metrics = candidate.shape_metrics
    sorted_flags = sorted(candidate.flags)
    return {
        "rank": rank,
        "priority_class": priority_class(candidate.static_score),
        "candidate_id": candidate.candidate_id,
        "morphology_type": candidate.morphology_type,
        **_source_context(source_properties),
        "static_score": candidate.static_score,
        "dynamic_score": "",
        "area_m2": metrics.area,
        "diameter_m": metrics.equivalent_diameter,
        "circularity": metrics.circularity,
        "elongation": metrics.elongation,
        "evidence": _json_string(_sorted_evidence(candidate)),
        "flags": _json_string(sorted_flags),
        "dominant_evidence": _dominant_evidence(candidate),
        "false_positive_flags": _json_string(_false_positive_flags(sorted_flags)),
    }


def candidate_object_to_feature(candidate: Any) -> dict[str, Any]:
    """Serialize a dynamic or fused object candidate row to GeoJSON."""
    geometry = _object_geometry(candidate)
    flags = _object_flags(candidate)
    false_positive_flags = _object_list(candidate, "false_positive_flags")
    return {
        "type": "Feature",
        "properties": {
            "candidate_id": _object_candidate_id(candidate),
            "evidence_class": _text(_object_value(candidate, "evidence_class")),
            "morphology_type": _object_morphology_type(candidate),
            "static_score": _object_score_value(candidate, "static_score"),
            "dynamic_score": _object_dynamic_score(candidate),
            "area_m2": _object_score_value(candidate, "area_m2", fallback=geometry.area),
            "diameter_m": _object_score_value(candidate, "diameter_m"),
            "circularity": _object_score_value(candidate, "circularity"),
            "elongation": _object_score_value(candidate, "elongation"),
            "flags": flags,
            "dynamic_object_flags": _object_list(candidate, "dynamic_object_flags"),
            "false_positive_flags": false_positive_flags,
            "false_positive_penalty": _object_score_value(candidate, "false_positive_penalty"),
            "missing_data_flags": _object_list(candidate, "missing_data_flags"),
        },
        "geometry": mapping(geometry),
    }


def candidate_object_to_score_row(
    candidate: Any,
    rank: int,
) -> dict[str, str | int | float]:
    """Serialize a dynamic or fused object candidate row to the score CSV schema."""
    static_score = _object_score_value(candidate, "static_score")
    return {
        "rank": rank,
        "priority_class": priority_class(static_score) if isinstance(static_score, float) else "",
        "candidate_id": _object_candidate_id(candidate),
        "morphology_type": _object_morphology_type(candidate),
        "source_landcover_context": "",
        "source_morphology_type": "",
        "source_false_positive_risk": "",
        "source_notes": "",
        "static_score": static_score,
        "dynamic_score": _object_dynamic_score(candidate),
        "area_m2": _object_score_value(
            candidate, "area_m2", fallback=_object_geometry(candidate).area
        ),
        "diameter_m": _object_score_value(candidate, "diameter_m"),
        "circularity": _object_score_value(candidate, "circularity"),
        "elongation": _object_score_value(candidate, "elongation"),
        "evidence": _json_string(
            {"evidence_class": _text(_object_value(candidate, "evidence_class"))}
        ),
        "flags": _json_string(_object_flags(candidate)),
        "dominant_evidence": _object_dominant_evidence(candidate),
        "false_positive_flags": _json_string(_object_list(candidate, "false_positive_flags")),
    }


def write_candidates_geojson(
    candidates: list[StaticCandidate],
    path: Path,
    source_properties: SourceProperties | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    feature_collection = {
        "type": "FeatureCollection",
        "features": [
            static_candidate_to_feature(candidate, properties)
            for candidate, properties in _candidate_source_pairs(candidates, source_properties)
        ],
    }
    path.write_text(
        json.dumps(feature_collection, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_candidate_scores_csv(
    candidates: list[StaticCandidate],
    path: Path,
    source_properties: SourceProperties | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SCORE_FIELDNAMES)
        writer.writeheader()
        for rank, (candidate, properties) in enumerate(
            _ranked_candidate_source_pairs(candidates, source_properties), start=1
        ):
            writer.writerow(static_candidate_to_score_row(candidate, rank, properties))


def write_candidate_objects_geojson(candidates: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    feature_collection = {
        "type": "FeatureCollection",
        "features": [
            candidate_object_to_feature(candidate)
            for candidate in _candidate_object_rows(candidates)
        ],
    }
    path.write_text(
        json.dumps(feature_collection, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_candidate_object_scores_csv(candidates: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SCORE_FIELDNAMES)
        writer.writeheader()
        for rank, candidate in enumerate(_candidate_object_rows(candidates), start=1):
            writer.writerow(candidate_object_to_score_row(candidate, rank))


def write_candidate_artifacts(
    candidates: list[StaticCandidate],
    output_dir: Path,
    source_properties: SourceProperties | None = None,
) -> CandidateArtifactPaths:
    paths = CandidateArtifactPaths(
        candidates_geojson=output_dir / "candidates.geojson",
        candidate_scores_csv=output_dir / "candidate_scores.csv",
    )
    write_candidates_geojson(candidates, paths.candidates_geojson, source_properties)
    write_candidate_scores_csv(candidates, paths.candidate_scores_csv, source_properties)
    return paths


def _source_context(source_properties: dict[str, object] | None) -> dict[str, str]:
    properties = source_properties or {}
    return {
        output_field: _text(properties.get(input_field))
        for input_field, output_field in SOURCE_CONTEXT_FIELD_MAP.items()
    }


def _text(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def _sorted_evidence(candidate: StaticCandidate) -> dict[str, float]:
    return {key: candidate.evidence[key] for key in sorted(candidate.evidence)}


def _ranked_candidates(candidates: list[StaticCandidate]) -> list[StaticCandidate]:
    return sorted(
        candidates, key=lambda candidate: (-candidate.static_score, candidate.candidate_id)
    )


def _candidate_source_pairs(
    candidates: list[StaticCandidate], source_properties: SourceProperties | None
) -> list[tuple[StaticCandidate, dict[str, object]]]:
    properties = source_properties or []
    return [
        (candidate, properties[index] if index < len(properties) else {})
        for index, candidate in enumerate(candidates)
    ]


def _ranked_candidate_source_pairs(
    candidates: list[StaticCandidate], source_properties: SourceProperties | None
) -> list[tuple[StaticCandidate, dict[str, object]]]:
    return sorted(
        _candidate_source_pairs(candidates, source_properties),
        key=lambda pair: (-pair[0].static_score, pair[0].candidate_id),
    )


def _json_string(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _candidate_object_rows(candidates: Any) -> list[Any]:
    if hasattr(candidates, "iterrows"):
        return [row for _, row in candidates.iterrows()]
    return list(candidates)


def _object_geometry(candidate: Any) -> Any:
    if hasattr(candidate, "geometry"):
        return candidate.geometry
    return _object_value(candidate, "geometry")


def _object_candidate_id(candidate: Any) -> str:
    for key in ("candidate_id", "object_id"):
        value = _text(_object_value(candidate, key))
        if value:
            return value
    return ""


def _object_value(candidate: Any, key: str) -> object | None:
    if hasattr(candidate, "index") and key in candidate.index:
        return candidate[key]
    if isinstance(candidate, dict):
        return candidate.get(key)
    return None


def _object_list(candidate: Any, key: str) -> list[str]:
    value = _object_value(candidate, key)
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable):
        return sorted(str(item) for item in value)
    return [str(value)]


def _object_flags(candidate: Any) -> list[str]:
    return sorted(
        set(_object_list(candidate, "static_flags"))
        | set(_object_list(candidate, "dynamic_object_flags"))
        | set(_object_list(candidate, "false_positive_flags"))
    )


def _object_score_value(candidate: Any, key: str, *, fallback: object | None = None) -> str | float:
    value = _object_value(candidate, key)
    if value is None:
        value = fallback
    if isinstance(value, int | float) and not isinstance(value, bool):
        number = float(value)
        return number if math.isfinite(number) else ""
    return ""


def _object_dynamic_score(candidate: Any) -> str | float:
    for key in ("dynamic_score", "dynamic_max_anomaly", "max_anomaly"):
        value = _object_score_value(candidate, key)
        if value != "":
            return value
    return ""


def _object_morphology_type(candidate: Any) -> str:
    for key in ("morphology_type", "static_morphology_type"):
        value = _text(_object_value(candidate, key))
        if value:
            return value
    return ""


def _object_dominant_evidence(candidate: Any) -> str:
    evidence_class = _text(_object_value(candidate, "evidence_class"))
    if evidence_class:
        return evidence_class.replace("_", " ")
    morphology_type = _object_morphology_type(candidate)
    if morphology_type:
        return f"{morphology_type} morphology"
    return ""


def _false_positive_flags(flags: list[str]) -> list[str]:
    return [flag for flag in flags if flag.endswith("_risk")]


def _dominant_evidence(candidate: StaticCandidate) -> str:
    if candidate.morphology_type == "ring":
        return "ring morphology"
    if candidate.morphology_type == "chain":
        return "chain-like geometry"
    return f"{candidate.morphology_type} morphology"
