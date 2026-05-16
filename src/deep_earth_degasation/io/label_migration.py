from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import geopandas as gpd
from shapely.geometry.base import BaseGeometry

from deep_earth_degasation.geo.crs import is_projected_metric_crs, parse_crs
from deep_earth_degasation.io.labeling import LABELING_FIELDNAMES

REVIEW_FIELDS = ("expert_label", "expert_confidence", "false_positive_reason", "reviewer_notes")
REPORT_FIELDNAMES = [
    "old_candidate_id",
    "new_candidate_id",
    "match_method",
    "iou",
    "centroid_distance_m",
    "area_ratio",
    "old_label",
    "migration_status",
    "review_required",
]


class LabelMigrationError(ValueError):
    """Raised when labels cannot be migrated safely."""


@dataclass(frozen=True)
class LabelMigrationThresholds:
    high_iou_threshold: float = 0.60
    centroid_distance_threshold_m: float = 30.0
    area_ratio_min: float = 0.50
    area_ratio_max: float = 2.00


@dataclass(frozen=True)
class LabelMigrationResult:
    updated_label_rows: list[dict[str, str]]
    migration_report_rows: list[dict[str, str]]
    retired_label_rows: list[dict[str, str]]
    new_candidate_rows: list[dict[str, str]]


def migrate_review_labels(
    *,
    old_label_rows: Iterable[Mapping[str, object]],
    old_candidates: gpd.GeoDataFrame,
    new_label_rows: Iterable[Mapping[str, object]],
    new_candidates: gpd.GeoDataFrame,
    thresholds: LabelMigrationThresholds | None = None,
) -> LabelMigrationResult:
    """Migrate reviewed labels from an old run to a new candidate inventory."""
    active_thresholds = thresholds or LabelMigrationThresholds()
    _require_compatible_metric_crs(old_candidates, new_candidates)
    old_rows = [_string_row(row) for row in old_label_rows]
    updated_rows = [_string_row(row) for row in new_label_rows]
    old_geometries = _candidate_geometries(old_candidates)
    new_geometries = _candidate_geometries(new_candidates)
    new_properties = _candidate_properties(new_candidates)
    used_new_ids: set[str] = set()
    report_rows: list[dict[str, str]] = []
    retired_rows: list[dict[str, str]] = []

    updated_by_id = {row.get("candidate_id", ""): row for row in updated_rows}
    for old_row in old_rows:
        if not _has_review(old_row):
            continue
        old_id = old_row.get("candidate_id", "")
        match = _match_label(
            old_id,
            old_geometries,
            new_geometries,
            used_new_ids,
            active_thresholds,
        )
        if match.status == "migrated":
            new_row = updated_by_id.get(match.new_candidate_id)
            if new_row is not None:
                _copy_review_fields(old_row, new_row)
                used_new_ids.add(match.new_candidate_id)
                report_rows.append(_report_row(old_row, match))
                continue
            appended_row = dict(new_properties.get(match.new_candidate_id, {}))
            appended_row["candidate_id"] = match.new_candidate_id
            _copy_review_fields(old_row, appended_row)
            appended_row["migration_status"] = "migrated_missing_target_row"
            updated_rows.append(appended_row)
            updated_by_id[match.new_candidate_id] = appended_row
            used_new_ids.add(match.new_candidate_id)
            report_rows.append(
                _report_row(
                    old_row,
                    replace(
                        match,
                        status="migrated_missing_target_row",
                        review_required=True,
                    ),
                )
            )
            continue
        report_rows.append(_report_row(old_row, match))
        retired_rows.append({**old_row, "migration_status": match.status})

    new_rows = [row for row in updated_rows if row.get("candidate_id", "") not in used_new_ids]
    return LabelMigrationResult(
        updated_label_rows=updated_rows,
        migration_report_rows=report_rows,
        retired_label_rows=retired_rows,
        new_candidate_rows=new_rows,
    )


def write_label_migration_outputs(
    result: LabelMigrationResult,
    output_dir: Path,
    *,
    updated_fieldnames: list[str] | None = None,
    retired_fieldnames: list[str] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    updated_fields = updated_fieldnames or LABELING_FIELDNAMES
    retired_fields = sorted(set(retired_fieldnames or LABELING_FIELDNAMES) | {"migration_status"})
    _write_csv(output_dir / "updated_labeling_table.csv", result.updated_label_rows, updated_fields)
    _write_csv(
        output_dir / "label_migration_report.csv", result.migration_report_rows, REPORT_FIELDNAMES
    )
    _write_csv(output_dir / "retired_labels.csv", result.retired_label_rows, retired_fields)
    _write_csv(output_dir / "new_candidates.csv", result.new_candidate_rows, updated_fields)


@dataclass(frozen=True)
class _CandidateGeometry:
    candidate_id: str
    field_id: str
    geometry: BaseGeometry


@dataclass(frozen=True)
class _Match:
    old_candidate_id: str
    new_candidate_id: str = ""
    method: str = ""
    iou: float = 0.0
    centroid_distance_m: float = 0.0
    area_ratio: float = 0.0
    status: str = "retired"
    review_required: bool = True


def _match_label(
    old_id: str,
    old_geometries: dict[str, list[_CandidateGeometry]],
    new_geometries: dict[str, list[_CandidateGeometry]],
    used_new_ids: set[str],
    thresholds: LabelMigrationThresholds,
) -> _Match:
    old_items = old_geometries.get(old_id, [])
    if len(old_items) != 1:
        return _Match(old_candidate_id=old_id, status="missing_or_duplicate_old_geometry")
    old_item = old_items[0]

    exact_items = [
        item for item in new_geometries.get(old_id, []) if item.candidate_id not in used_new_ids
    ]
    if len(exact_items) == 1:
        exact = _scored_match(old_item, exact_items[0], "exact_id")
        if _compatible(exact, thresholds):
            return exact
        return _Match(
            old_candidate_id=old_id,
            new_candidate_id=exact.new_candidate_id,
            method="exact_id",
            iou=exact.iou,
            centroid_distance_m=exact.centroid_distance_m,
            area_ratio=exact.area_ratio,
            status="geometry_changed",
        )
    if len(exact_items) > 1:
        return _Match(old_candidate_id=old_id, method="exact_id", status="ambiguous")

    candidates = [
        item
        for items in new_geometries.values()
        for item in items
        if item.candidate_id not in used_new_ids and item.field_id == old_item.field_id
    ]
    high_iou = [
        _scored_match(old_item, candidate, "field_iou")
        for candidate in candidates
        if _iou(old_item.geometry, candidate.geometry) >= thresholds.high_iou_threshold
    ]
    if len(high_iou) == 1:
        return high_iou[0]
    if len(high_iou) > 1:
        return _Match(old_candidate_id=old_id, method="field_iou", status="ambiguous")

    centroid_matches = [
        match
        for match in (
            _scored_match(old_item, candidate, "field_centroid_area") for candidate in candidates
        )
        if match.centroid_distance_m <= thresholds.centroid_distance_threshold_m
        and thresholds.area_ratio_min <= match.area_ratio <= thresholds.area_ratio_max
    ]
    if len(centroid_matches) == 1:
        return centroid_matches[0]
    if len(centroid_matches) > 1:
        return _Match(old_candidate_id=old_id, method="field_centroid_area", status="ambiguous")
    return _Match(old_candidate_id=old_id, status="retired")


def _scored_match(
    old_item: _CandidateGeometry,
    new_item: _CandidateGeometry,
    method: str,
) -> _Match:
    return _Match(
        old_candidate_id=old_item.candidate_id,
        new_candidate_id=new_item.candidate_id,
        method=method,
        iou=_iou(old_item.geometry, new_item.geometry),
        centroid_distance_m=old_item.geometry.centroid.distance(new_item.geometry.centroid),
        area_ratio=_area_ratio(old_item.geometry.area, new_item.geometry.area),
        status="migrated",
        review_required=False,
    )


def _compatible(match: _Match, thresholds: LabelMigrationThresholds) -> bool:
    return match.iou >= thresholds.high_iou_threshold or (
        match.centroid_distance_m <= thresholds.centroid_distance_threshold_m
        and thresholds.area_ratio_min <= match.area_ratio <= thresholds.area_ratio_max
    )


def _candidate_geometries(candidates: gpd.GeoDataFrame) -> dict[str, list[_CandidateGeometry]]:
    geometries: dict[str, list[_CandidateGeometry]] = {}
    for _, row in candidates.iterrows():
        candidate_id = _text(row.get("candidate_id"))
        if not candidate_id:
            continue
        geometries.setdefault(candidate_id, []).append(
            _CandidateGeometry(
                candidate_id=candidate_id,
                field_id=_text(row.get("field_id")),
                geometry=row.geometry,
            )
        )
    return geometries


def _candidate_properties(candidates: gpd.GeoDataFrame) -> dict[str, dict[str, str]]:
    properties: dict[str, dict[str, str]] = {}
    for _, row in candidates.iterrows():
        candidate_id = _text(row.get("candidate_id"))
        if not candidate_id or candidate_id in properties:
            continue
        properties[candidate_id] = {
            str(key): _text(value) for key, value in row.items() if key != candidates.geometry.name
        }
    return properties


def _require_compatible_metric_crs(
    old_candidates: gpd.GeoDataFrame,
    new_candidates: gpd.GeoDataFrame,
) -> None:
    if old_candidates.crs is None:
        raise LabelMigrationError("old_candidates has no CRS metadata.")
    if new_candidates.crs is None:
        raise LabelMigrationError("new_candidates has no CRS metadata.")
    old_crs = parse_crs(old_candidates.crs)
    new_crs = parse_crs(new_candidates.crs)
    if not is_projected_metric_crs(old_crs):
        raise LabelMigrationError("old_candidates must use a projected metre CRS.")
    if not is_projected_metric_crs(new_crs):
        raise LabelMigrationError("new_candidates must use a projected metre CRS.")
    if not old_crs.equals(new_crs):
        raise LabelMigrationError(
            "old_candidates and new_candidates must use the same CRS before label migration."
        )


def _copy_review_fields(source: Mapping[str, str], target: dict[str, str]) -> None:
    for field in REVIEW_FIELDS:
        target[field] = source.get(field, "")


def _report_row(old_row: Mapping[str, str], match: _Match) -> dict[str, str]:
    return {
        "old_candidate_id": match.old_candidate_id,
        "new_candidate_id": match.new_candidate_id,
        "match_method": match.method,
        "iou": _number(match.iou),
        "centroid_distance_m": _number(match.centroid_distance_m),
        "area_ratio": _number(match.area_ratio),
        "old_label": old_row.get("expert_label", ""),
        "migration_status": match.status,
        "review_required": "yes" if match.review_required else "no",
    }


def _has_review(row: Mapping[str, str]) -> bool:
    return any(row.get(field, "") for field in REVIEW_FIELDS)


def _iou(left: BaseGeometry, right: BaseGeometry) -> float:
    union_area = left.union(right).area
    if union_area <= 0:
        return 0.0
    return float(left.intersection(right).area / union_area)


def _area_ratio(old_area: float, new_area: float) -> float:
    if old_area <= 0 or new_area <= 0:
        return 0.0
    return float(new_area / old_area)


def _string_row(row: Mapping[str, object]) -> dict[str, str]:
    return {str(key): _text(value) for key, value in row.items()}


def _text(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def _number(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _write_csv(path: Path, rows: Sequence[Mapping[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
