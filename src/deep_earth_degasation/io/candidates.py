from __future__ import annotations

import csv
import json
import math
import re
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
    "object_score",
    "morphology_type",
    "evidence_class",
    "landcover_branch",
    "dominant_landcover_branch",
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
    "annulus_contrast",
    "ringness_score",
    "support_pixel_count",
    "field_residual_max",
    "peer_residual_max",
    "temporal_residual_max",
    "residual_types_present",
    "field_id",
    "distance_to_field_edge_m",
    "moisture_anomaly",
    "vegetation_stress",
    "soil_brightness_bsi",
    "thermal_anomaly",
    "sar_anomaly",
    "persistence",
    "post_rain_drying",
    "geology_context",
    "false_positive_penalty",
    "evidence",
    "flags",
    "dominant_evidence",
    "false_positive_flags",
    "missing_data_flags",
    "anomalous_dates",
    "source_feature_names",
    "source_layer_ids",
    "passport_path",
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


def candidate_object_to_feature(
    candidate: Any,
    passport_path: Path | str | None = None,
) -> dict[str, Any]:
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
            "object_score": _object_score_value(candidate, "object_score"),
            "static_score": _object_score_value(candidate, "static_score"),
            "dynamic_score": _object_dynamic_score(candidate),
            "area_m2": _object_score_value(candidate, "area_m2", fallback=geometry.area),
            "diameter_m": _object_diameter(candidate),
            "circularity": _object_score_value(candidate, "circularity"),
            "elongation": _object_score_value(candidate, "elongation"),
            "annulus_contrast": _object_score_value(candidate, "annulus_contrast"),
            "ringness_score": _object_score_value(candidate, "ringness_score"),
            "support_pixel_count": _object_int_value(candidate, "support_pixel_count"),
            "field_residual_max": _object_score_value(candidate, "field_residual_max"),
            "peer_residual_max": _object_score_value(candidate, "peer_residual_max"),
            "temporal_residual_max": _object_score_value(candidate, "temporal_residual_max"),
            "residual_types_present": _object_list(candidate, "residual_types_present"),
            "landcover_branch": _text(_object_value(candidate, "landcover_branch")),
            "dominant_landcover_branch": _text(
                _object_value(candidate, "dominant_landcover_branch")
            ),
            "field_id": _text(_object_value(candidate, "field_id")),
            "distance_to_field_edge_m": _object_score_value(candidate, "distance_to_field_edge_m"),
            "dominant_evidence": _object_dominant_evidence(candidate),
            "dynamic_evidence": _dynamic_evidence(candidate),
            "flags": flags,
            "dynamic_object_flags": _object_list(candidate, "dynamic_object_flags"),
            "false_positive_flags": false_positive_flags,
            "false_positive_penalty": _object_score_value(candidate, "false_positive_penalty"),
            "missing_data_flags": _object_list(candidate, "missing_data_flags"),
            "anomalous_dates": _object_list(candidate, "anomalous_dates"),
            "source_feature_names": _object_list(candidate, "source_feature_names"),
            "source_layer_ids": _object_list(candidate, "source_layer_ids"),
            "passport_path": str(passport_path or _object_value(candidate, "passport_path") or ""),
        },
        "geometry": mapping(geometry),
    }


def candidate_object_to_score_row(
    candidate: Any,
    rank: int,
    passport_path: Path | str | None = None,
) -> dict[str, str | int | float]:
    """Serialize a dynamic or fused object candidate row to the score CSV schema."""
    static_score = _object_score_value(candidate, "static_score")
    object_score = _object_score_value(candidate, "object_score")
    dynamic_evidence = _dynamic_evidence(candidate)
    return {
        "rank": rank,
        "priority_class": _object_priority_class(candidate, object_score),
        "candidate_id": _object_candidate_id(candidate),
        "object_score": object_score,
        "morphology_type": _object_morphology_type(candidate),
        "evidence_class": _text(_object_value(candidate, "evidence_class")),
        "landcover_branch": _text(_object_value(candidate, "landcover_branch")),
        "dominant_landcover_branch": _text(_object_value(candidate, "dominant_landcover_branch")),
        "source_landcover_context": "",
        "source_morphology_type": "",
        "source_false_positive_risk": "",
        "source_notes": "",
        "static_score": static_score,
        "dynamic_score": _object_dynamic_score(candidate),
        "area_m2": _object_score_value(
            candidate, "area_m2", fallback=_object_geometry(candidate).area
        ),
        "diameter_m": _object_diameter(candidate),
        "circularity": _object_score_value(candidate, "circularity"),
        "elongation": _object_score_value(candidate, "elongation"),
        "annulus_contrast": _object_score_value(candidate, "annulus_contrast"),
        "ringness_score": _object_score_value(candidate, "ringness_score"),
        "support_pixel_count": _object_int_value(candidate, "support_pixel_count"),
        "field_residual_max": _object_score_value(candidate, "field_residual_max"),
        "peer_residual_max": _object_score_value(candidate, "peer_residual_max"),
        "temporal_residual_max": _object_score_value(candidate, "temporal_residual_max"),
        "residual_types_present": _json_string(_object_list(candidate, "residual_types_present")),
        "field_id": _text(_object_value(candidate, "field_id")),
        "distance_to_field_edge_m": _object_score_value(candidate, "distance_to_field_edge_m"),
        "moisture_anomaly": dynamic_evidence["moisture_anomaly"],
        "vegetation_stress": dynamic_evidence["vegetation_stress"],
        "soil_brightness_bsi": dynamic_evidence["soil_brightness_bsi"],
        "thermal_anomaly": dynamic_evidence["thermal_anomaly"],
        "sar_anomaly": dynamic_evidence["sar_anomaly"],
        "persistence": _object_score_value(candidate, "repeated_seasons"),
        "post_rain_drying": dynamic_evidence["post_rain_drying"],
        "geology_context": dynamic_evidence["geology_context"],
        "false_positive_penalty": _object_score_value(candidate, "false_positive_penalty"),
        "evidence": _json_string(
            {"evidence_class": _text(_object_value(candidate, "evidence_class"))}
        ),
        "flags": _json_string(_object_flags(candidate)),
        "dominant_evidence": _object_dominant_evidence(candidate),
        "false_positive_flags": _json_string(_object_list(candidate, "false_positive_flags")),
        "missing_data_flags": _json_string(_object_list(candidate, "missing_data_flags")),
        "anomalous_dates": _json_string(_object_list(candidate, "anomalous_dates")),
        "source_feature_names": _json_string(_object_list(candidate, "source_feature_names")),
        "source_layer_ids": _json_string(_object_list(candidate, "source_layer_ids")),
        "passport_path": str(passport_path or _object_value(candidate, "passport_path") or ""),
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


def write_candidate_objects_geojson(
    candidates: Any,
    path: Path,
    *,
    passports_dir: Path | None = None,
    limit: int | None = None,
    passport_path_limit: int | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    feature_collection = {
        "type": "FeatureCollection",
        "features": [
            candidate_object_to_feature(
                candidate,
                _object_passport_path(
                    candidate,
                    rank,
                    passports_dir,
                    passport_path_limit=passport_path_limit,
                ),
            )
            for rank, candidate in enumerate(
                _ranked_candidate_object_rows(candidates, limit=limit), start=1
            )
        ],
    }
    crs = _candidate_objects_crs(candidates)
    if crs:
        feature_collection["crs"] = {"type": "name", "properties": {"name": crs}}
    path.write_text(
        json.dumps(feature_collection, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_candidate_object_scores_csv(
    candidates: Any,
    path: Path,
    *,
    passports_dir: Path | None = None,
    limit: int | None = None,
    passport_path_limit: int | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SCORE_FIELDNAMES)
        writer.writeheader()
        writer.writerows(
            candidate_object_score_rows(
                candidates,
                passports_dir=passports_dir,
                limit=limit,
                passport_path_limit=passport_path_limit,
            )
        )


def candidate_object_score_rows(
    candidates: Any,
    *,
    passports_dir: Path | None = None,
    limit: int | None = None,
    passport_path_limit: int | None = None,
) -> list[dict[str, str | int | float]]:
    """Return ranked score rows for dynamic or fused object candidates."""
    return [
        candidate_object_to_score_row(
            candidate,
            rank,
            _object_passport_path(
                candidate,
                rank,
                passports_dir,
                passport_path_limit=passport_path_limit,
            ),
        )
        for rank, candidate in enumerate(
            _ranked_candidate_object_rows(candidates, limit=limit), start=1
        )
    ]


def write_candidate_object_time_series(
    candidates: Any, output_dir: Path, *, limit: int | None = None
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for rank, candidate in enumerate(
        _ranked_candidate_object_rows(candidates, limit=limit), start=1
    ):
        rows = _object_time_series_rows(candidate)
        if not rows:
            continue
        path = output_dir / f"{_safe_filename_stem(_object_candidate_id(candidate), rank)}.csv"
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=["candidate_id", "date", "source_layer_ids", "source_feature_names"],
            )
            writer.writeheader()
            writer.writerows(rows)
        paths.append(path)
    return paths


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


def _ranked_candidate_object_rows(candidates: Any, *, limit: int | None = None) -> list[Any]:
    rows = sorted(
        _candidate_object_rows(candidates),
        key=lambda candidate: (
            -_sortable_score(_object_value(candidate, "object_score")),
            _object_candidate_id(candidate),
        ),
    )
    return rows if limit is None else rows[: max(0, limit)]


def _candidate_objects_crs(candidates: Any) -> str:
    crs = getattr(candidates, "crs", None)
    if crs is None:
        return ""
    to_string = getattr(crs, "to_string", None)
    if callable(to_string):
        return str(to_string())
    return str(crs)


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
    if value is None or _is_nan(value):
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable):
        return sorted(str(item) for item in value)
    return [str(value)]


def _is_nan(value: object) -> bool:
    return isinstance(value, float) and math.isnan(value)


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


def _object_diameter(candidate: Any) -> str | float:
    for key in ("diameter_m", "equivalent_diameter_m"):
        value = _object_score_value(candidate, key)
        if value != "":
            return value
    return ""


def _object_int_value(candidate: Any, key: str) -> str | int:
    value = _object_value(candidate, key)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return int(value)
    return ""


def _object_dynamic_score(candidate: Any) -> str | float:
    for key in ("dynamic_score", "dynamic_max_anomaly", "max_anomaly"):
        value = _object_score_value(candidate, key)
        if value != "":
            return value
    return ""


def _object_priority_class(candidate: Any, object_score: str | float) -> str:
    existing = _text(_object_value(candidate, "priority_class"))
    if existing:
        return existing
    return priority_class(object_score) if isinstance(object_score, float) else ""


def _sortable_score(value: object | None) -> float:
    if isinstance(value, int | float) and not isinstance(value, bool):
        number = float(value)
        return number if math.isfinite(number) else 0.0
    return 0.0


def _dynamic_evidence(candidate: Any) -> dict[str, str | float]:
    per_feature_max = _object_value(candidate, "per_feature_max")
    evidence: dict[str, str | float] = {
        "moisture_anomaly": "",
        "vegetation_stress": "",
        "soil_brightness_bsi": "",
        "thermal_anomaly": "",
        "sar_anomaly": "",
        "post_rain_drying": "",
        "geology_context": "",
    }
    if not isinstance(per_feature_max, dict):
        return evidence
    for feature_name, value in per_feature_max.items():
        output_field = _dynamic_evidence_field(str(feature_name))
        if output_field is None:
            continue
        current = evidence[output_field]
        score_value = _score_value(value)
        if not isinstance(score_value, float):
            continue
        evidence[output_field] = (
            max(current, score_value) if isinstance(current, float) else score_value
        )
    return evidence


def _dynamic_evidence_field(feature_name: str) -> str | None:
    normalized = feature_name.lower()
    if any(token in normalized for token in ("ndmi", "ndwi", "msi", "moisture")):
        return "moisture_anomaly"
    if any(token in normalized for token in ("ndvi", "red_edge", "red-edge", "vegetation")):
        return "vegetation_stress"
    if any(token in normalized for token in ("bsi", "brightness", "bare_soil", "bare-soil")):
        return "soil_brightness_bsi"
    if any(token in normalized for token in ("lst", "thermal", "temperature")):
        return "thermal_anomaly"
    if any(token in normalized for token in ("sar", "vv", "vh")):
        return "sar_anomaly"
    if "post_rain" in normalized or "drying" in normalized:
        return "post_rain_drying"
    if "geology" in normalized:
        return "geology_context"
    return None


def _score_value(value: object) -> str | float:
    if isinstance(value, int | float) and not isinstance(value, bool):
        number = float(value)
        return number if math.isfinite(number) else ""
    return ""


def _object_passport_path(
    candidate: Any,
    rank: int,
    passports_dir: Path | None,
    *,
    passport_path_limit: int | None = None,
) -> Path | str:
    if passports_dir is None:
        value = _object_value(candidate, "passport_path")
        return "" if value is None else str(value)
    if passport_path_limit is not None and rank > passport_path_limit:
        return ""
    return passports_dir / f"{_safe_filename_stem(_object_candidate_id(candidate), rank)}.md"


def _object_time_series_rows(candidate: Any) -> list[dict[str, str]]:
    dates = _object_list(candidate, "anomalous_dates")
    if not dates:
        return []
    candidate_id = _object_candidate_id(candidate)
    layer_ids = _json_string(_object_list(candidate, "source_layer_ids"))
    feature_names = _json_string(_object_list(candidate, "source_feature_names"))
    return [
        {
            "candidate_id": candidate_id,
            "date": date,
            "source_layer_ids": layer_ids,
            "source_feature_names": feature_names,
        }
        for date in dates
    ]


def _safe_filename_stem(candidate_id: str, rank: int) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", candidate_id).strip("._-")
    return stem or f"candidate-{rank}"


def _object_morphology_type(candidate: Any) -> str:
    for key in ("morphology_type", "static_morphology_type"):
        value = _text(_object_value(candidate, key))
        if value:
            return value
    return ""


def _object_dominant_evidence(candidate: Any) -> str:
    existing = _text(_object_value(candidate, "dominant_evidence"))
    if existing:
        return existing
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
