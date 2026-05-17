from __future__ import annotations

import math
from collections.abc import Hashable
from dataclasses import dataclass
from typing import Any

import geopandas as gpd
from shapely.geometry.base import BaseGeometry

from deep_earth_degasation.morphology.static_detector import StaticCandidate


@dataclass(frozen=True)
class CandidateFusionConfig:
    iou_min: float = 0.10
    centroid_distance_max_m: float = 50.0


def fuse_static_dynamic_candidates(
    static_candidates: list[StaticCandidate],
    dynamic_objects: gpd.GeoDataFrame,
    *,
    config: CandidateFusionConfig | None = None,
    crs: str | None = None,
) -> gpd.GeoDataFrame:
    """Fuse static morphology candidates and dynamic anomaly objects for review."""
    fusion_config = config or CandidateFusionConfig()
    rows: list[dict[str, Any]] = []
    used_dynamic_indices: set[Hashable] = set()

    for static_candidate in sorted(static_candidates, key=lambda candidate: candidate.candidate_id):
        match = _best_dynamic_match(
            static_candidate.geometry,
            dynamic_objects,
            used_dynamic_indices=used_dynamic_indices,
            config=fusion_config,
        )
        if match is None:
            rows.append(_static_only_row(static_candidate))
            continue

        dynamic_index, dynamic_row = match
        used_dynamic_indices.add(dynamic_index)
        rows.append(_static_dynamic_row(static_candidate, dynamic_row))

    for dynamic_index, dynamic_row in dynamic_objects.iterrows():
        if dynamic_index not in used_dynamic_indices:
            rows.append(_dynamic_only_row(dynamic_row))

    if not rows:
        output_crs = crs if crs is not None else _crs_text(dynamic_objects.crs)
        return _empty_fused_candidates(output_crs)
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=crs or dynamic_objects.crs)


def _best_dynamic_match(
    static_geometry: BaseGeometry,
    dynamic_objects: gpd.GeoDataFrame,
    *,
    used_dynamic_indices: set[Hashable],
    config: CandidateFusionConfig,
) -> tuple[Hashable, Any] | None:
    matches: list[tuple[float, float, str, Hashable, Any]] = []
    for dynamic_index, dynamic_row in dynamic_objects.iterrows():
        if dynamic_index in used_dynamic_indices:
            continue
        dynamic_geometry = dynamic_row.geometry
        iou = _intersection_over_union(static_geometry, dynamic_geometry)
        centroid_distance = float(static_geometry.centroid.distance(dynamic_geometry.centroid))
        if iou >= config.iou_min or centroid_distance <= config.centroid_distance_max_m:
            matches.append((iou, centroid_distance, str(dynamic_index), dynamic_index, dynamic_row))

    if not matches:
        return None
    _, _, _, index, row = sorted(matches, key=lambda item: (-item[0], item[1], item[2]))[0]
    return index, row


def _static_dynamic_row(static_candidate: StaticCandidate, dynamic_row: Any) -> dict[str, Any]:
    dynamic_object_id = _dynamic_object_id(dynamic_row)
    geometry = static_candidate.geometry.union(dynamic_row.geometry)
    false_positive_flags = sorted(
        set(_static_false_positive_flags(static_candidate))
        | set(_list_value(dynamic_row, "false_positive_flags"))
    )
    return {
        **_static_fields(static_candidate),
        **_dynamic_fields(dynamic_row),
        "candidate_id": f"fused-{static_candidate.candidate_id}-{dynamic_object_id}",
        "evidence_class": "static_dynamic",
        "source_static_candidate_id": static_candidate.candidate_id,
        "source_dynamic_object_id": dynamic_object_id,
        "source_static_candidate_ids": (static_candidate.candidate_id,),
        "source_dynamic_object_ids": (dynamic_object_id,),
        "false_positive_flags": tuple(false_positive_flags),
        "false_positive_penalty": _float_value(dynamic_row, "false_positive_penalty", default=0.0),
        "missing_data_flags": tuple(_list_value(dynamic_row, "missing_data_flags")),
        "geometry": geometry,
    }


def _static_only_row(static_candidate: StaticCandidate) -> dict[str, Any]:
    return {
        **_static_fields(static_candidate),
        **_empty_dynamic_fields(),
        "candidate_id": f"static-{static_candidate.candidate_id}",
        "evidence_class": "static_only",
        "source_static_candidate_id": static_candidate.candidate_id,
        "source_dynamic_object_id": "",
        "source_static_candidate_ids": (static_candidate.candidate_id,),
        "source_dynamic_object_ids": (),
        "false_positive_flags": tuple(_static_false_positive_flags(static_candidate)),
        "false_positive_penalty": 0.0,
        "missing_data_flags": (),
        "geometry": static_candidate.geometry,
    }


def _dynamic_only_row(dynamic_row: Any) -> dict[str, Any]:
    dynamic_object_id = _dynamic_object_id(dynamic_row)
    return {
        **_empty_static_fields(),
        **_dynamic_fields(dynamic_row),
        "candidate_id": f"dynamic-{dynamic_object_id}",
        "evidence_class": "dynamic_only",
        "source_static_candidate_id": "",
        "source_dynamic_object_id": dynamic_object_id,
        "source_static_candidate_ids": (),
        "source_dynamic_object_ids": (dynamic_object_id,),
        "geometry": dynamic_row.geometry,
    }


def _static_fields(candidate: StaticCandidate) -> dict[str, Any]:
    return {
        "static_score": candidate.static_score,
        "static_morphology_type": candidate.morphology_type,
        "static_flags": tuple(sorted(candidate.flags)),
    }


def _empty_static_fields() -> dict[str, Any]:
    return {
        "static_score": None,
        "static_morphology_type": "",
        "static_flags": (),
    }


def _dynamic_fields(dynamic_row: Any) -> dict[str, Any]:
    return {
        "area_m2": _row_value(dynamic_row, "area_m2"),
        "perimeter_m": _row_value(dynamic_row, "perimeter_m"),
        "equivalent_diameter_m": _row_value(dynamic_row, "equivalent_diameter_m"),
        "circularity": _row_value(dynamic_row, "circularity"),
        "elongation": _row_value(dynamic_row, "elongation"),
        "dynamic_mean_anomaly": _row_value(dynamic_row, "mean_anomaly"),
        "dynamic_max_anomaly": _row_value(dynamic_row, "max_anomaly"),
        "per_feature_mean": _dict_value(dynamic_row, "per_feature_mean"),
        "per_feature_max": _dict_value(dynamic_row, "per_feature_max"),
        "source_feature_names": tuple(_list_value(dynamic_row, "source_feature_names")),
        "dynamic_object_flags": tuple(_row_value(dynamic_row, "dynamic_object_flags") or ()),
        "anomalous_dates": tuple(_row_value(dynamic_row, "anomalous_dates") or ()),
        "source_layer_ids": tuple(_row_value(dynamic_row, "source_layer_ids") or ()),
        "source_detection_count": _row_value(dynamic_row, "source_detection_count"),
        "repeated_seasons": _row_value(dynamic_row, "repeated_seasons"),
        "false_positive_flags": tuple(_list_value(dynamic_row, "false_positive_flags")),
        "false_positive_penalty": _float_value(dynamic_row, "false_positive_penalty", default=0.0),
        "missing_data_flags": tuple(_list_value(dynamic_row, "missing_data_flags")),
    }


def _empty_dynamic_fields() -> dict[str, Any]:
    return {
        "area_m2": None,
        "perimeter_m": None,
        "equivalent_diameter_m": None,
        "circularity": None,
        "elongation": None,
        "dynamic_mean_anomaly": None,
        "dynamic_max_anomaly": None,
        "per_feature_mean": {},
        "per_feature_max": {},
        "source_feature_names": (),
        "dynamic_object_flags": (),
        "anomalous_dates": (),
        "source_layer_ids": (),
        "source_detection_count": None,
        "repeated_seasons": None,
    }


def _dynamic_object_id(dynamic_row: Any) -> str:
    return str(_row_value(dynamic_row, "object_id"))


def _row_value(row: Any, field_name: str) -> Any:
    return row[field_name] if field_name in row else None


def _list_value(row: Any, field_name: str) -> tuple[Any, ...]:
    value = _row_value(row, field_name)
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(value)


def _dict_value(row: Any, field_name: str) -> dict[Any, Any]:
    value = _row_value(row, field_name)
    return value if isinstance(value, dict) else {}


def _float_value(row: Any, field_name: str, *, default: float) -> float:
    value = _row_value(row, field_name)
    if isinstance(value, int | float):
        number = float(value)
        return number if math.isfinite(number) else default
    return default


def _static_false_positive_flags(candidate: StaticCandidate) -> list[str]:
    return [flag for flag in sorted(candidate.flags) if flag.endswith("_risk")]


def _intersection_over_union(left: BaseGeometry, right: BaseGeometry) -> float:
    union_area = float(left.union(right).area)
    if union_area <= 0:
        return 0.0
    return float(left.intersection(right).area) / union_area


def _empty_fused_candidates(crs: str | None) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "candidate_id": [],
            "evidence_class": [],
            "source_static_candidate_id": [],
            "source_dynamic_object_id": [],
            "source_static_candidate_ids": [],
            "source_dynamic_object_ids": [],
            "static_score": [],
            "static_morphology_type": [],
            "static_flags": [],
            "area_m2": [],
            "perimeter_m": [],
            "equivalent_diameter_m": [],
            "circularity": [],
            "elongation": [],
            "dynamic_mean_anomaly": [],
            "dynamic_max_anomaly": [],
            "per_feature_mean": [],
            "per_feature_max": [],
            "source_feature_names": [],
            "dynamic_object_flags": [],
            "anomalous_dates": [],
            "source_layer_ids": [],
            "source_detection_count": [],
            "repeated_seasons": [],
            "false_positive_flags": [],
            "false_positive_penalty": [],
            "missing_data_flags": [],
            "geometry": [],
        },
        geometry="geometry",
        crs=crs,
    )


def _crs_text(crs: object | None) -> str | None:
    if crs is None:
        return None
    return str(crs)
