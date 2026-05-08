from __future__ import annotations

import geopandas as gpd
from shapely.geometry import Point, box

from deep_earth_degasation.candidates.fusion import (
    CandidateFusionConfig,
    fuse_static_dynamic_candidates,
)
from deep_earth_degasation.morphology.static_detector import extract_static_candidates

CRS = "EPSG:32637"


def test_fusion_preserves_static_dynamic_and_dynamic_only_evidence_classes() -> None:
    static_candidates = extract_static_candidates(
        [Point(0, 0).buffer(30), Point(300, 0).buffer(30)],
        candidate_ids=["static-match", "static-only"],
    )
    dynamic_objects = _dynamic_objects(
        [
            ("dynamic-match", Point(20, 0).buffer(25), 3.0),
            ("dynamic-only", Point(600, 0).buffer(25), 2.0),
        ]
    )

    fused = fuse_static_dynamic_candidates(static_candidates, dynamic_objects, crs=CRS)

    assert list(fused["evidence_class"]) == ["static_dynamic", "static_only", "dynamic_only"]
    matched = fused[fused["evidence_class"] == "static_dynamic"].iloc[0]
    assert matched["source_static_candidate_id"] == "static-match"
    assert matched["source_dynamic_object_id"] == "dynamic-match"
    assert matched["source_static_candidate_ids"] == ("static-match",)
    assert matched["source_dynamic_object_ids"] == ("dynamic-match",)
    assert matched.geometry.area > static_candidates[0].geometry.area


def test_centroid_distance_match_works_when_iou_is_below_threshold() -> None:
    static_candidates = extract_static_candidates(
        [box(0, 0, 10, 10)],
        candidate_ids=["static-near"],
    )
    dynamic_objects = _dynamic_objects([("dynamic-near", box(12, 0, 22, 10), 2.0)])

    fused = fuse_static_dynamic_candidates(
        static_candidates,
        dynamic_objects,
        config=CandidateFusionConfig(iou_min=0.5, centroid_distance_max_m=20.0),
        crs=CRS,
    )

    assert fused["evidence_class"].iloc[0] == "static_dynamic"
    assert fused["source_dynamic_object_id"].iloc[0] == "dynamic-near"


def test_unmatched_static_candidate_remains_static_only() -> None:
    static_candidates = extract_static_candidates(
        [Point(0, 0).buffer(30)],
        candidate_ids=["static-only"],
    )
    dynamic_objects = _dynamic_objects([("dynamic-far", Point(500, 0).buffer(20), 2.0)])

    fused = fuse_static_dynamic_candidates(
        static_candidates,
        dynamic_objects,
        config=CandidateFusionConfig(iou_min=0.5, centroid_distance_max_m=20.0),
        crs=CRS,
    )

    static_row = fused[fused["evidence_class"] == "static_only"].iloc[0]
    assert static_row["source_static_candidate_id"] == "static-only"
    assert static_row["source_dynamic_object_id"] == ""


def test_unmatched_dynamic_object_remains_dynamic_only() -> None:
    fused = fuse_static_dynamic_candidates(
        [],
        _dynamic_objects([("dynamic-only", Point(0, 0).buffer(20), 2.0)]),
        crs=CRS,
    )

    assert len(fused) == 1
    assert fused["evidence_class"].iloc[0] == "dynamic_only"
    assert fused["source_static_candidate_ids"].iloc[0] == ()
    assert fused["source_dynamic_object_ids"].iloc[0] == ("dynamic-only",)
    assert fused["dynamic_max_anomaly"].iloc[0] == 2.0


def test_fusion_preserves_false_positive_filter_outputs() -> None:
    static_candidates = extract_static_candidates(
        [Point(0, 0).buffer(30)],
        candidate_ids=["static-match"],
        landcover_context="built_up",
    )
    dynamic_objects = _dynamic_objects([("dynamic-match", Point(20, 0).buffer(25), 3.0)])
    dynamic_objects["false_positive_flags"] = [["road_risk"]]
    dynamic_objects["false_positive_penalty"] = [0.3]
    dynamic_objects["missing_data_flags"] = [["missing_context_water"]]

    fused = fuse_static_dynamic_candidates(static_candidates, dynamic_objects, crs=CRS)

    matched = fused[fused["evidence_class"] == "static_dynamic"].iloc[0]
    assert matched["false_positive_flags"] == ("built_up_risk", "road_risk")
    assert matched["false_positive_penalty"] == 0.3
    assert matched["missing_data_flags"] == ("missing_context_water",)


def test_dynamic_object_is_not_reused_by_multiple_static_candidates() -> None:
    static_candidates = extract_static_candidates(
        [Point(0, 0).buffer(30), Point(20, 0).buffer(30)],
        candidate_ids=["static-a", "static-b"],
    )
    dynamic_objects = _dynamic_objects([("dynamic-shared", Point(5, 0).buffer(25), 3.0)])

    fused = fuse_static_dynamic_candidates(static_candidates, dynamic_objects, crs=CRS)

    assert list(fused["evidence_class"]).count("static_dynamic") == 1
    assert list(fused["evidence_class"]).count("static_only") == 1
    assert list(fused["source_dynamic_object_id"]).count("dynamic-shared") == 1


def test_empty_static_and_dynamic_inputs_return_empty_fused_geodataframe() -> None:
    empty_dynamic = gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=CRS)

    fused = fuse_static_dynamic_candidates([], empty_dynamic, crs=CRS)

    assert fused.empty
    assert str(fused.crs) == CRS
    assert "geometry" in fused.columns
    assert "candidate_id" in fused.columns
    assert "evidence_class" in fused.columns
    assert "source_static_candidate_ids" in fused.columns
    assert "source_dynamic_object_ids" in fused.columns


def _dynamic_objects(objects: list[tuple[str, object, float]]) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "object_id": [object_id for object_id, _, _ in objects],
            "mean_anomaly": [max_anomaly / 2 for _, _, max_anomaly in objects],
            "max_anomaly": [max_anomaly for _, _, max_anomaly in objects],
            "dynamic_object_flags": [[] for _ in objects],
            "anomalous_dates": [("2024-05-01",) for _ in objects],
            "source_layer_ids": [(f"{object_id}_layer",) for object_id, _, _ in objects],
            "geometry": [geometry for _, geometry, _ in objects],
        },
        geometry="geometry",
        crs=CRS,
    )
