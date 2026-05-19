from __future__ import annotations

import csv
import json
from pathlib import Path

from deep_earth_degasation.io.review_exports import (
    object_feature_row,
    rank_explanation_row,
    write_object_feature_table,
    write_rank_explanations,
)


def test_object_feature_table_preserves_ranked_feature_snapshot(tmp_path: Path) -> None:
    output_path = tmp_path / "object_feature_table.csv"

    write_object_feature_table([_score_row()], output_path)

    with output_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert rows[0] == object_feature_row(_score_row())
    assert rows[0]["candidate_id"] == "candidate-a"
    assert rows[0]["field_residual_max"] == "2.4"
    assert rows[0]["source_feature_names"] == '["NDMI", "BSI"]'
    assert rows[0]["road_distance_m"] == "12.5"
    assert rows[0]["water_distance_m"] == "100"
    assert rows[0]["false_positive_profile"] == '{"flags": ["road_risk"]}'


def test_rank_explanation_table_summarizes_evidence_penalties_and_missing_context(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "rank_explanations.csv"

    write_rank_explanations([_score_row()], output_path)

    with output_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert rows[0] == rank_explanation_row(_score_row())
    assert "moisture_anomaly=2.1" in rows[0]["positive_evidence"]
    assert "flags=road_risk" in rows[0]["false_positive_evidence"]
    assert rows[0]["false_positive_distances"] == "road_distance_m=12.5; water_distance_m=100"
    assert (
        "distances=road_distance_m=12.5; water_distance_m=100" in rows[0]["false_positive_evidence"]
    )
    assert rows[0]["missing_context"] == "missing_context_water"
    assert json.loads(rows[0]["score_components"])["object_score"] == "0.82"
    assert json.loads(rows[0]["score_components"])["road_distance_m"] == "12.5"
    assert (
        "False-positive distances: road_distance_m=12.5; water_distance_m=100"
        in rows[0]["explanation"]
    )
    assert "Candidate requires expert and field validation" in rows[0]["explanation"]
    assert "not direct H2 detection" in rows[0]["explanation"]


def _score_row() -> dict[str, object]:
    return {
        "rank": "1",
        "candidate_id": "candidate-a",
        "priority_class": "A",
        "object_score": "0.82",
        "static_score": "0.4",
        "dynamic_score": "0.9",
        "evidence_class": "dynamic_only",
        "morphology_type": "irregular",
        "landcover_branch": "cropland",
        "dominant_landcover_branch": "cropland",
        "field_id": "field-1",
        "area_m2": "700",
        "diameter_m": "30",
        "circularity": "0.7",
        "elongation": "1.3",
        "annulus_contrast": "0.2",
        "ringness_score": "0.35",
        "support_pixel_count": "12",
        "field_residual_max": "2.4",
        "peer_residual_max": "1.8",
        "temporal_residual_max": "0.6",
        "residual_types_present": '["field", "peer"]',
        "moisture_anomaly": "2.1",
        "vegetation_stress": "0.8",
        "soil_brightness_bsi": "1.5",
        "thermal_anomaly": "",
        "sar_anomaly": "",
        "post_rain_drying": "0.4",
        "persistence": "2",
        "false_positive_penalty": "0.2",
        "road_distance_m": "12.5",
        "built_up_distance_m": "",
        "water_distance_m": "100",
        "quarry_distance_m": "",
        "woody_patch_distance_m": "",
        "irrigation_distance_m": "",
        "cloud_shadow_distance_m": "",
        "harvest_pattern_distance_m": "",
        "excluded_zone_distance_m": "",
        "false_positive_flags": '["road_risk"]',
        "false_positive_profile": '{"flags": ["road_risk"]}',
        "missing_data_flags": '["missing_context_water"]',
        "anomalous_dates": '["2025-05-15"]',
        "source_feature_names": '["NDMI", "BSI"]',
        "source_layer_ids": '["ndmi-2025-05", "bsi-2025-05"]',
        "dominant_evidence": "cropland dynamic anomaly",
    }
