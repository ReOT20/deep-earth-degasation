from __future__ import annotations

import csv
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import box

from deep_earth_degasation.io.candidates import (
    candidate_object_to_score_row,
    write_candidate_object_scores_csv,
    write_candidate_object_time_series,
    write_candidate_objects_geojson,
)
from deep_earth_degasation.io.labeling import score_row_to_labeling_row
from deep_earth_degasation.reports.passport import render_candidate_passport
from deep_earth_degasation.reports.quicklook import write_quicklook_png

CRS = "EPSG:32637"


def test_dynamic_candidate_scores_include_review_evidence_and_passport_paths(
    tmp_path: Path,
) -> None:
    candidates = _candidate_objects()
    output_path = tmp_path / "candidate_scores.csv"

    write_candidate_object_scores_csv(
        candidates,
        output_path,
        passports_dir=tmp_path / "passports",
    )

    with output_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert rows[0]["candidate_id"] == "candidate-high"
    assert rows[0]["object_score"] == "0.91"
    assert rows[0]["dynamic_score"] == "0.82"
    assert rows[0]["landcover_branch"] == "cropland"
    assert rows[0]["field_id"] == "field-7"
    assert rows[0]["support_pixel_count"] == "7"
    assert rows[0]["false_positive_flags"] == '["road_risk"]'
    assert rows[0]["missing_data_flags"] == '["missing_context_water"]'
    assert rows[0]["passport_path"].endswith("passports/candidate-high.md")
    assert rows[0]["moisture_anomaly"] == "3.0"
    assert rows[0]["vegetation_stress"] == "2.5"
    assert rows[0]["soil_brightness_bsi"] == "2.25"
    assert rows[0]["sar_anomaly"] == "2.0"


def test_dynamic_candidate_geojson_preserves_scores_flags_and_source_evidence(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "candidates.geojson"

    write_candidate_objects_geojson(
        _candidate_objects(),
        output_path,
        passports_dir=tmp_path / "passports",
    )

    data = json.loads(output_path.read_text(encoding="utf-8"))
    properties = data["features"][0]["properties"]
    reloaded = gpd.read_file(output_path)

    assert str(reloaded.crs) == CRS
    assert properties["candidate_id"] == "candidate-high"
    assert properties["object_score"] == 0.91
    assert properties["dynamic_score"] == 0.82
    assert properties["dominant_evidence"] == "cropland dynamic anomaly with static morphology"
    assert properties["false_positive_flags"] == ["road_risk"]
    assert properties["support_pixel_count"] == 7
    assert properties["missing_data_flags"] == ["missing_context_water"]
    assert properties["anomalous_dates"] == ["2023-05-01", "2024-05-01"]
    assert properties["dynamic_evidence"]["moisture_anomaly"] == 3.0
    assert properties["passport_path"].endswith("passports/candidate-high.md")


def test_time_series_export_is_created_only_when_anomaly_dates_exist(tmp_path: Path) -> None:
    paths = write_candidate_object_time_series(_candidate_objects(), tmp_path / "time_series")

    assert len(paths) == 1
    with paths[0].open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert [row["date"] for row in rows] == ["2023-05-01", "2024-05-01"]
    assert rows[0]["candidate_id"] == "candidate-high"
    assert rows[0]["source_layer_ids"] == '["s1","s2"]'
    assert rows[0]["source_feature_names"] == '["BSI","NDMI","NDVI","VV_VH_ratio"]'


def test_dynamic_score_rows_feed_labeling_and_enriched_passports() -> None:
    score_row = candidate_object_to_score_row(
        _candidate_objects().iloc[0],
        rank=1,
        passport_path="passports/candidate-high.md",
    )

    labeling_row = score_row_to_labeling_row(score_row)
    passport = render_candidate_passport(score_row)

    assert labeling_row["object_score"] == "0.91"
    assert labeling_row["missing_data_flags"] == '["missing_context_water"]'
    assert labeling_row["anomalous_dates"] == '["2023-05-01","2024-05-01"]'
    assert labeling_row["source_feature_names"] == '["BSI","NDMI","NDVI","VV_VH_ratio"]'
    assert labeling_row["source_layer_ids"] == '["s1","s2"]'
    assert "`object_score`: 0.91" in passport
    assert "`evidence_class`: static_dynamic" in passport
    assert "`anomalous_dates`: 2023-05-01, 2024-05-01" in passport
    assert "`source_feature_names`: BSI, NDMI, NDVI, VV_VH_ratio" in passport
    assert "`field_id`: field-7" in passport
    assert "`missing_data_flags`: missing_context_water" in passport
    assert "not direct H2 detection" in passport


def test_quicklook_png_generation_works_on_synthetic_raster(tmp_path: Path) -> None:
    output_path = tmp_path / "quicklook.png"

    write_quicklook_png(
        np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64),
        output_path,
        candidate_geometry=box(0, 0, 1, 1),
        title="candidate-high",
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0
    assert output_path.read_bytes().startswith(b"\x89PNG")


def _candidate_objects() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        [
            {
                "candidate_id": "candidate-high",
                "object_score": 0.91,
                "priority_class": "A",
                "evidence_class": "static_dynamic",
                "static_score": 0.88,
                "dynamic_score": 0.82,
                "landcover_branch": "cropland",
                "dominant_landcover_branch": "cropland",
                "field_id": "field-7",
                "distance_to_field_edge_m": 35.0,
                "area_m2": 1200.0,
                "support_pixel_count": 7,
                "equivalent_diameter_m": 39.1,
                "circularity": 0.82,
                "elongation": 1.4,
                "per_feature_max": {
                    "NDMI": 3.0,
                    "NDVI": 2.5,
                    "BSI": 2.25,
                    "VV_VH_ratio": 2.0,
                },
                "source_feature_names": ("NDMI", "NDVI", "BSI", "VV_VH_ratio"),
                "source_layer_ids": ("s1", "s2"),
                "anomalous_dates": ("2023-05-01", "2024-05-01"),
                "repeated_seasons": 2,
                "dominant_evidence": "cropland dynamic anomaly with static morphology",
                "false_positive_flags": ("road_risk",),
                "false_positive_penalty": 0.3,
                "missing_data_flags": ("missing_context_water",),
                "geometry": box(0, 0, 10, 10),
            },
            {
                "candidate_id": "candidate-low",
                "object_score": 0.2,
                "priority_class": "D",
                "evidence_class": "dynamic_only",
                "static_score": 0.0,
                "dynamic_score": 0.2,
                "false_positive_flags": (),
                "missing_data_flags": (),
                "geometry": box(20, 20, 25, 25),
            },
        ],
        geometry="geometry",
        crs=CRS,
    )
