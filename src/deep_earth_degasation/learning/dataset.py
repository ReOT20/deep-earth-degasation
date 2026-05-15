from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping
from pathlib import Path

CANONICAL_LABELS = {"positive", "weak_positive", "hard_negative", "unlabeled", "uncertain"}

LEARNING_FIELDNAMES = [
    "object_id",
    "candidate_id",
    "source_type",
    "geometry_ref",
    "aoi_name",
    "run_id",
    "feature_snapshot_id",
    "rank",
    "priority_class",
    "field_id",
    "landcover_branch",
    "dominant_landcover_branch",
    "object_score",
    "static_score",
    "dynamic_score",
    "moisture_anomaly",
    "vegetation_stress",
    "soil_brightness_bsi",
    "thermal_anomaly",
    "sar_anomaly",
    "persistence",
    "support_pixel_count",
    "dominant_evidence",
    "feature_flags",
    "missing_data_flags",
    "model_label",
    "legacy_label",
    "label_source",
    "label_confidence",
    "reviewer_id",
    "review_date",
    "reviewer_notes",
    "false_positive_reason",
    "false_positive_flags",
    "pu_role",
    "split_group",
    "train_split",
    "leakage_notes",
    "use_for_training",
]


class LearningDatasetError(ValueError):
    """Raised when label rows cannot be converted to weak/PU learning rows."""


def build_learning_dataset_rows(
    score_rows: Iterable[Mapping[str, object]],
    label_rows: Iterable[Mapping[str, object]] | None = None,
    *,
    run_id: str = "",
    feature_snapshot_id: str = "",
    aoi_name: str = "",
    geometry_ref: str = "",
) -> list[dict[str, str]]:
    """Build modeling-ready weak/PU rows from ranked candidate score rows."""
    labels_by_candidate = _labels_by_candidate(label_rows or ())
    return [
        _learning_row(
            score_row,
            labels_by_candidate.get(_text(score_row.get("candidate_id"))),
            run_id=run_id,
            feature_snapshot_id=feature_snapshot_id,
            aoi_name=aoi_name,
            geometry_ref=geometry_ref,
        )
        for score_row in score_rows
    ]


def write_learning_dataset(
    score_rows: Iterable[Mapping[str, object]],
    path: Path,
    label_rows: Iterable[Mapping[str, object]] | None = None,
    *,
    run_id: str = "",
    feature_snapshot_id: str = "",
    aoi_name: str = "",
    geometry_ref: str = "",
) -> None:
    """Write a weak/PU learning dataset CSV."""
    rows = build_learning_dataset_rows(
        score_rows,
        label_rows,
        run_id=run_id,
        feature_snapshot_id=feature_snapshot_id,
        aoi_name=aoi_name,
        geometry_ref=geometry_ref,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=LEARNING_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _labels_by_candidate(
    label_rows: Iterable[Mapping[str, object]],
) -> dict[str, Mapping[str, object]]:
    labels: dict[str, Mapping[str, object]] = {}
    for row in label_rows:
        candidate_id = _text(row.get("candidate_id"))
        if candidate_id:
            labels[candidate_id] = row
    return labels


def _learning_row(
    score_row: Mapping[str, object],
    label_row: Mapping[str, object] | None,
    *,
    run_id: str,
    feature_snapshot_id: str,
    aoi_name: str,
    geometry_ref: str,
) -> dict[str, str]:
    candidate_id = _text(score_row.get("candidate_id"))
    label = _label_text(label_row, "expert_label")
    model_label, legacy_label = _model_label(label, label_row)
    false_positive_reason = _label_text(label_row, "false_positive_reason")
    if model_label == "hard_negative" and not false_positive_reason:
        raise LearningDatasetError(
            f"hard_negative candidate {candidate_id or '<missing>'} requires false_positive_reason"
        )

    return {
        "object_id": candidate_id,
        "candidate_id": candidate_id,
        "source_type": _text(score_row.get("source_type")) or "detector_export",
        "geometry_ref": _text(score_row.get("geometry_ref")) or geometry_ref,
        "aoi_name": aoi_name,
        "run_id": run_id,
        "feature_snapshot_id": feature_snapshot_id,
        "rank": _text(score_row.get("rank")),
        "priority_class": _text(score_row.get("priority_class")),
        "field_id": _text(score_row.get("field_id")),
        "landcover_branch": _text(score_row.get("landcover_branch")),
        "dominant_landcover_branch": _text(score_row.get("dominant_landcover_branch")),
        "object_score": _text(score_row.get("object_score")),
        "static_score": _text(score_row.get("static_score")),
        "dynamic_score": _text(score_row.get("dynamic_score")),
        "moisture_anomaly": _text(score_row.get("moisture_anomaly")),
        "vegetation_stress": _text(score_row.get("vegetation_stress")),
        "soil_brightness_bsi": _text(score_row.get("soil_brightness_bsi")),
        "thermal_anomaly": _text(score_row.get("thermal_anomaly")),
        "sar_anomaly": _text(score_row.get("sar_anomaly")),
        "persistence": _text(score_row.get("persistence")),
        "support_pixel_count": _text(score_row.get("support_pixel_count")),
        "dominant_evidence": _text(score_row.get("dominant_evidence")),
        "feature_flags": _feature_flags(score_row),
        "missing_data_flags": _text(score_row.get("missing_data_flags")),
        "model_label": model_label,
        "legacy_label": legacy_label,
        "label_source": _label_source(label_row),
        "label_confidence": _label_text(label_row, "expert_confidence"),
        "reviewer_id": _label_text(label_row, "reviewer_id"),
        "review_date": _label_text(label_row, "review_date")
        or _label_text(label_row, "reviewed_at"),
        "reviewer_notes": _label_text(label_row, "reviewer_notes"),
        "false_positive_reason": false_positive_reason,
        "false_positive_flags": _text(score_row.get("false_positive_flags")),
        "pu_role": _pu_role(model_label),
        "split_group": _split_group(score_row, candidate_id),
        "train_split": "",
        "leakage_notes": _leakage_notes(score_row),
        "use_for_training": "no" if model_label == "uncertain" else "yes",
    }


def _model_label(label: str, label_row: Mapping[str, object] | None) -> tuple[str, str]:
    normalized = _normalize_label(label)
    if normalized == "":
        return "unlabeled", ""
    if normalized in CANONICAL_LABELS:
        return normalized, ""
    raise LearningDatasetError(f"unsupported expert_label {label!r}")


def _normalize_label(value: str) -> str:
    return value.strip().lower().replace("-", "_")


def _pu_role(model_label: str) -> str:
    return {
        "positive": "positive_seed",
        "weak_positive": "weak_positive_seed",
        "hard_negative": "hard_negative_control",
        "unlabeled": "unlabeled_pool",
        "uncertain": "exclude",
    }[model_label]


def _split_group(score_row: Mapping[str, object], candidate_id: str) -> str:
    field_id = _text(score_row.get("field_id"))
    if field_id:
        return f"field:{field_id}"
    return f"object:{candidate_id}"


def _feature_flags(score_row: Mapping[str, object]) -> str:
    values = [
        _text(score_row.get("evidence")),
        _text(score_row.get("flags")),
        _text(score_row.get("dynamic_object_flags")),
    ]
    return "; ".join(value for value in values if value)


def _leakage_notes(score_row: Mapping[str, object]) -> str:
    if _text(score_row.get("field_id")):
        return "grouped by field_id"
    return "grouped by candidate_id"


def _label_source(label_row: Mapping[str, object] | None) -> str:
    if label_row is None:
        return "detector_export"
    if _label_text(label_row, "label_source"):
        return _label_text(label_row, "label_source")
    if _label_text(label_row, "expert_label"):
        return "expert"
    return "detector_export"


def _label_text(label_row: Mapping[str, object] | None, key: str) -> str:
    if label_row is None:
        return ""
    return _text(label_row.get(key))


def _text(value: object) -> str:
    if value is None:
        return ""
    return str(value)
