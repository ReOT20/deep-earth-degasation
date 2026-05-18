from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import geopandas as gpd
from shapely.geometry.base import BaseGeometry

DEFAULT_GUARDRAIL = (
    "Validation summaries describe ranked candidate surface anomalies for expert and "
    "field review; they are not direct H2 detections or proof of active degassing."
)

POSITIVE_LABELS = {"positive", "weak_positive"}
NEGATIVE_LABELS = {"hard_negative"}
UNKNOWN_LABELS = {"", "unlabeled", "unknown", "uncertain"}


def build_validation_summary(
    *,
    score_rows: Sequence[Mapping[str, object]],
    candidates: gpd.GeoDataFrame | None = None,
    known_sites: gpd.GeoDataFrame | None = None,
    label_rows: Sequence[Mapping[str, object]] | None = None,
    top_n: int = 20,
    known_site_recall_top_n: Sequence[int] = (),
    expert_precision_top_n: int | None = None,
    guardrail: str = DEFAULT_GUARDRAIL,
) -> dict[str, Any]:
    """Build a JSON-serializable validation summary for ranked candidate artifacts."""
    ranked_rows = _ranked_rows(score_rows)
    run_flags: set[str] = set()
    recall_at_n = _known_site_recall_at_n(
        ranked_rows=ranked_rows,
        candidates=candidates,
        known_sites=known_sites,
        n_values=known_site_recall_top_n or (top_n,),
        run_flags=run_flags,
    )
    expert_precision = _expert_precision(
        ranked_rows=ranked_rows,
        label_rows=label_rows if label_rows is not None else score_rows,
        top_n=expert_precision_top_n or top_n,
        run_flags=run_flags,
    )
    false_positive_counts = _flag_counts(ranked_rows, "false_positive_flags")
    missing_data = _missing_data_summary(ranked_rows)
    if missing_data["affected_candidate_count"] > 0:
        run_flags.add("candidate_missing_data_flags_present")

    return {
        "schema_version": "validation_summary.v1",
        "guardrail": guardrail,
        "candidate_count": len(ranked_rows),
        "top_n": top_n,
        "known_site_recall_at_n": recall_at_n,
        "known_site_label_counts": _known_site_label_counts(known_sites),
        "known_site_confidence_counts": _known_site_confidence_counts(known_sites),
        "expert_precision_top_n": expert_precision,
        "false_positive_counts": false_positive_counts,
        "multi_sensor_agreement": _multi_sensor_agreement(ranked_rows),
        "persistence": _persistence_summary(ranked_rows),
        "missing_data": missing_data,
        "missing_data_flags": sorted(run_flags),
        "unlabeled_background_treated_as_negative": False,
    }


def write_validation_summary(summary: Mapping[str, object], path: Path) -> None:
    """Write a validation summary JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _ranked_rows(rows: Sequence[Mapping[str, object]]) -> list[Mapping[str, object]]:
    return sorted(
        rows,
        key=lambda row: (_rank_value(row), _text(row.get("candidate_id"))),
    )


def _rank_value(row: Mapping[str, object]) -> int:
    rank = _int_value(row.get("rank"))
    if rank is not None:
        return rank
    return 1_000_000


def _known_site_recall_at_n(
    *,
    ranked_rows: Sequence[Mapping[str, object]],
    candidates: gpd.GeoDataFrame | None,
    known_sites: gpd.GeoDataFrame | None,
    n_values: Sequence[int],
    run_flags: set[str],
) -> dict[str, float | None]:
    values = tuple(sorted({int(n) for n in n_values if int(n) > 0}))
    if not values:
        return {}

    if known_sites is None or known_sites.empty:
        run_flags.add("missing_known_sites")
        return {f"top_{n}": None for n in values}
    if candidates is None or candidates.empty:
        return {f"top_{n}": 0.0 for n in values}

    positive_known_sites = _positive_known_sites(known_sites)
    if positive_known_sites.empty:
        run_flags.add("missing_positive_known_sites")
        return {f"top_{n}": None for n in values}
    _require_matching_crs(candidates, positive_known_sites)
    candidate_geometries = _candidate_geometries_by_id(candidates)
    known_geometries = _valid_geometries(positive_known_sites)
    if not known_geometries:
        run_flags.add("missing_known_sites")
        return {f"top_{n}": None for n in values}

    recall: dict[str, float | None] = {}
    for n in values:
        top_ids = [_text(row.get("candidate_id")) for row in ranked_rows[:n]]
        top_geometries = [
            candidate_geometries[candidate_id]
            for candidate_id in top_ids
            if candidate_id in candidate_geometries
        ]
        matched_count = sum(
            _intersects_any(known_geometry, top_geometries) for known_geometry in known_geometries
        )
        recall[f"top_{n}"] = matched_count / len(known_geometries)
    return recall


def _positive_known_sites(known_sites: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    label_column = _first_existing_column(known_sites, ("expert_label", "status", "label"))
    if label_column is None:
        return known_sites
    labels = known_sites[label_column].map(_normalized_label)
    return known_sites[labels.isin(POSITIVE_LABELS)].copy()


def _require_matching_crs(candidates: gpd.GeoDataFrame, known_sites: gpd.GeoDataFrame) -> None:
    if candidates.crs is None or known_sites.crs is None:
        raise ValueError("Known-site recall requires candidates and known_sites to declare CRS.")
    if candidates.crs != known_sites.crs:
        raise ValueError(
            "Known-site recall requires candidates and known_sites to use the same CRS; "
            f"got candidates={candidates.crs} known_sites={known_sites.crs}."
        )


def _candidate_geometries_by_id(candidates: gpd.GeoDataFrame) -> dict[str, BaseGeometry]:
    geometries: dict[str, BaseGeometry] = {}
    for _, row in candidates.iterrows():
        candidate_id = _text(_row_value(row, "candidate_id") or _row_value(row, "object_id"))
        geometry = _row_value(row, "geometry")
        if candidate_id and isinstance(geometry, BaseGeometry) and not geometry.is_empty:
            geometries[candidate_id] = geometry
    return geometries


def _known_site_label_counts(known_sites: gpd.GeoDataFrame | None) -> dict[str, int]:
    if known_sites is None or known_sites.empty:
        return {}
    label_column = _first_existing_column(known_sites, ("expert_label", "status", "label"))
    if label_column is None:
        return {"positive_assumed_unlabeled_known_sites": len(known_sites)}
    counter: Counter[str] = Counter()
    for value in known_sites[label_column]:
        label = _label_count_value(value)
        counter[label] += 1
    return dict(sorted(counter.items()))


def _known_site_confidence_counts(known_sites: gpd.GeoDataFrame | None) -> dict[str, int]:
    if known_sites is None or known_sites.empty:
        return {}
    confidence_column = _first_existing_column(
        known_sites,
        ("expert_confidence", "confidence", "review_confidence"),
    )
    if confidence_column is None:
        return {}
    counter: Counter[str] = Counter()
    for value in known_sites[confidence_column]:
        text = _text(value).strip()
        counter[text if text else "unknown"] += 1
    return dict(sorted(counter.items()))


def _first_existing_column(data: gpd.GeoDataFrame, columns: Sequence[str]) -> str | None:
    for column in columns:
        if column in data.columns:
            return column
    return None


def _label_count_value(value: object) -> str:
    label = _text(value).strip().lower().replace(" ", "_").replace("-", "_")
    return label or "unknown"


def _valid_geometries(data: gpd.GeoDataFrame) -> list[BaseGeometry]:
    return [
        geometry
        for geometry in data.geometry
        if isinstance(geometry, BaseGeometry) and not geometry.is_empty
    ]


def _intersects_any(geometry: BaseGeometry, candidates: Sequence[BaseGeometry]) -> bool:
    return any(geometry.intersects(candidate_geometry) for candidate_geometry in candidates)


def _expert_precision(
    *,
    ranked_rows: Sequence[Mapping[str, object]],
    label_rows: Sequence[Mapping[str, object]],
    top_n: int,
    run_flags: set[str],
) -> dict[str, int | float | None]:
    labels_by_id = {
        _text(row.get("candidate_id")): _normalized_label(
            row.get("expert_label") or row.get("status")
        )
        for row in label_rows
        if _text(row.get("candidate_id"))
    }
    considered_rows = ranked_rows[:top_n]
    positive_count = 0
    negative_count = 0
    for row in considered_rows:
        label = labels_by_id.get(_text(row.get("candidate_id")), "")
        if label in POSITIVE_LABELS:
            positive_count += 1
        elif label in NEGATIVE_LABELS:
            negative_count += 1

    reviewed_count = positive_count + negative_count
    if reviewed_count == 0:
        run_flags.add("missing_expert_labels")
        precision: float | None = None
    else:
        precision = positive_count / reviewed_count
        if reviewed_count < len(considered_rows):
            run_flags.add("partial_expert_labels")

    return {
        "top_n": top_n,
        "precision": precision,
        "reviewed_count": reviewed_count,
        "positive_count": positive_count,
        "negative_count": negative_count,
    }


def _normalized_label(value: object) -> str:
    label = _text(value).strip().lower().replace(" ", "_").replace("-", "_")
    return "" if label in UNKNOWN_LABELS else label


def _flag_counts(rows: Sequence[Mapping[str, object]], field_name: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        counter.update(_list_value(row.get(field_name)))
    return dict(sorted(counter.items()))


def _multi_sensor_agreement(rows: Sequence[Mapping[str, object]]) -> dict[str, int | float]:
    counts = [len(set(_list_value(row.get("source_feature_names")))) for row in rows]
    if not counts:
        return {
            "mean_source_feature_count": 0.0,
            "max_source_feature_count": 0,
            "candidate_count_with_2plus_source_features": 0,
            "share_with_2plus_source_features": 0.0,
        }
    multi_count = sum(count >= 2 for count in counts)
    return {
        "mean_source_feature_count": sum(counts) / len(counts),
        "max_source_feature_count": max(counts),
        "candidate_count_with_2plus_source_features": multi_count,
        "share_with_2plus_source_features": multi_count / len(counts),
    }


def _persistence_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, int | float]:
    values = [_float_value(row.get("persistence")) for row in rows]
    persistence = [value for value in values if value is not None]
    if not persistence:
        return {
            "mean_repeated_seasons": 0.0,
            "max_repeated_seasons": 0.0,
            "persistent_candidate_count": 0,
        }
    return {
        "mean_repeated_seasons": sum(persistence) / len(persistence),
        "max_repeated_seasons": max(persistence),
        "persistent_candidate_count": sum(value >= 2.0 for value in persistence),
    }


def _missing_data_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, Any]:
    flag_counts = _flag_counts(rows, "missing_data_flags")
    affected_count = sum(1 for row in rows if _list_value(row.get("missing_data_flags")))
    return {
        "affected_candidate_count": affected_count,
        "flag_counts": flag_counts,
    }


def _list_value(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return [text]
        return _list_value(parsed)
    if isinstance(value, Mapping):
        return [str(key) for key in value]
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [str(item) for item in value if str(item)]
    return [str(value)]


def _int_value(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = int(str(value))
    except ValueError:
        return None
    return number


def _float_value(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(str(value))
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _text(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def _row_value(row: Any, key: str) -> object | None:
    if hasattr(row, "index") and key in row.index:
        return row[key]
    if isinstance(row, Mapping):
        return row.get(key)
    return None
