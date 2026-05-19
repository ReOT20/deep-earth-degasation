from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping
from pathlib import Path

LABELING_FIELDNAMES = [
    "candidate_id",
    "rank",
    "priority_class",
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
    "field_id",
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
    "persistence",
    "post_rain_drying",
    "false_positive_penalty",
    "road_distance_m",
    "built_up_distance_m",
    "water_distance_m",
    "quarry_distance_m",
    "woody_patch_distance_m",
    "irrigation_distance_m",
    "cloud_shadow_distance_m",
    "harvest_pattern_distance_m",
    "excluded_zone_distance_m",
    "false_positive_profile",
    "dominant_evidence",
    "false_positive_flags",
    "missing_data_flags",
    "anomalous_dates",
    "source_feature_names",
    "source_layer_ids",
    "passport_path",
    "expert_label",
    "expert_confidence",
    "false_positive_reason",
    "reviewer_notes",
]


def score_row_to_labeling_row(score_row: Mapping[str, object]) -> dict[str, str]:
    row = {field: _text(score_row, field) for field in LABELING_FIELDNAMES}
    row["expert_label"] = ""
    row["expert_confidence"] = ""
    row["false_positive_reason"] = ""
    row["reviewer_notes"] = ""
    return row


def write_labeling_table(score_rows: Iterable[Mapping[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=LABELING_FIELDNAMES)
        writer.writeheader()
        for score_row in score_rows:
            writer.writerow(score_row_to_labeling_row(score_row))


def _text(row: Mapping[str, object], key: str) -> str:
    value = row.get(key, "")
    if value is None:
        return ""
    return str(value)
