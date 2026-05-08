from __future__ import annotations

import json

import geopandas as gpd
import pytest
from shapely.geometry import Point, box

from deep_earth_degasation.validation.summary import build_validation_summary

CRS = "EPSG:32637"


def test_validation_summary_reports_known_site_recall_and_partial_expert_precision() -> None:
    score_rows = [
        _score_row(
            "candidate-a",
            rank=1,
            flags=["road_intersection"],
            missing_flags=["missing_sar_features"],
            features=["NDMI", "NDVI"],
            persistence=2,
        ),
        _score_row(
            "candidate-b",
            rank=2,
            flags=["water_intersection"],
            features=["NDMI"],
            persistence=1,
        ),
        _score_row("candidate-c", rank=3, features=["NDMI", "NDVI", "BSI"], persistence=3),
    ]
    candidates = gpd.GeoDataFrame(
        {
            "object_id": ["candidate-a", "candidate-b", "candidate-c"],
            "geometry": [
                box(0, 0, 10, 10),
                box(20, 0, 30, 10),
                box(40, 0, 50, 10),
            ],
        },
        crs=CRS,
    )
    known_sites = gpd.GeoDataFrame(
        {"site_id": ["known-1", "known-2"], "geometry": [Point(5, 5), Point(45, 5)]},
        crs=CRS,
    )
    label_rows = [
        {"candidate_id": "candidate-a", "expert_label": "A"},
        {"candidate_id": "candidate-b", "expert_label": "N"},
        {"candidate_id": "candidate-c", "expert_label": "U"},
    ]

    summary = build_validation_summary(
        score_rows=score_rows,
        candidates=candidates,
        known_sites=known_sites,
        label_rows=label_rows,
        top_n=3,
        known_site_recall_top_n=(1, 2, 3),
        expert_precision_top_n=3,
    )

    assert summary["known_site_recall_at_n"] == {
        "top_1": 0.5,
        "top_2": 0.5,
        "top_3": 1.0,
    }
    assert summary["expert_precision_top_n"] == {
        "top_n": 3,
        "precision": 0.5,
        "reviewed_count": 2,
        "positive_count": 1,
        "negative_count": 1,
    }
    assert summary["false_positive_counts"] == {
        "road_intersection": 1,
        "water_intersection": 1,
    }
    assert summary["multi_sensor_agreement"] == {
        "mean_source_feature_count": 2.0,
        "max_source_feature_count": 3,
        "candidate_count_with_2plus_source_features": 2,
        "share_with_2plus_source_features": pytest.approx(2 / 3),
    }
    assert summary["persistence"] == {
        "mean_repeated_seasons": 2.0,
        "max_repeated_seasons": 3.0,
        "persistent_candidate_count": 2,
    }
    assert summary["missing_data"] == {
        "affected_candidate_count": 1,
        "flag_counts": {"missing_sar_features": 1},
    }
    assert "partial_expert_labels" in summary["missing_data_flags"]
    assert summary["unlabeled_background_treated_as_negative"] is False


def test_validation_summary_handles_missing_known_sites_and_expert_labels() -> None:
    summary = build_validation_summary(
        score_rows=[_score_row("candidate-a", rank=1)],
        candidates=gpd.GeoDataFrame(
            {"object_id": ["candidate-a"], "geometry": [box(0, 0, 10, 10)]},
            crs=CRS,
        ),
        known_sites=None,
        top_n=5,
        known_site_recall_top_n=(5,),
        expert_precision_top_n=5,
    )

    assert summary["known_site_recall_at_n"] == {"top_5": None}
    assert summary["expert_precision_top_n"]["precision"] is None
    assert summary["expert_precision_top_n"]["reviewed_count"] == 0
    assert "missing_known_sites" in summary["missing_data_flags"]
    assert "missing_expert_labels" in summary["missing_data_flags"]
    assert summary["unlabeled_background_treated_as_negative"] is False


def _score_row(
    candidate_id: str,
    *,
    rank: int,
    flags: list[str] | None = None,
    missing_flags: list[str] | None = None,
    features: list[str] | None = None,
    persistence: int = 1,
) -> dict[str, object]:
    return {
        "rank": str(rank),
        "candidate_id": candidate_id,
        "false_positive_flags": json.dumps(flags or []),
        "missing_data_flags": json.dumps(missing_flags or []),
        "source_feature_names": json.dumps(features or ["NDMI"]),
        "persistence": str(persistence),
    }
