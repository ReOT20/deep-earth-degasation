from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import LineString, Point, box

from deep_earth_degasation.context.false_positive import (
    FalsePositiveContext,
    FalsePositiveFilterConfig,
    apply_false_positive_filters,
)
from deep_earth_degasation.context.fields import ContextAssignmentError
from deep_earth_degasation.io.candidates import (
    candidate_object_to_score_row,
    static_candidate_to_feature,
    static_candidate_to_score_row,
    write_candidate_object_scores_csv,
    write_candidate_objects_geojson,
)
from deep_earth_degasation.io.labeling import score_row_to_labeling_row
from deep_earth_degasation.morphology.static_detector import extract_static_candidates
from deep_earth_degasation.reports.passport import render_candidate_passport

CRS = "EPSG:32637"


def test_spatial_context_layers_add_false_positive_flags_and_penalties() -> None:
    objects = _objects([box(0, 0, 10, 10)])
    context = FalsePositiveContext(
        roads=_layer([LineString([(12, 0), (12, 10)])]),
        water=_layer([box(100, 100, 110, 110)]),
        built_up=_layer([box(0, 0, 5, 5)]),
        excluded_zones=_layer([box(2, 2, 8, 8)]),
        quarries=_layer([box(1, 1, 3, 3)]),
        woody_patches=_layer([box(90, 90, 95, 95)]),
    )
    config = FalsePositiveFilterConfig(
        flag_cloud_shadows=False,
        flag_harvest_patterns=False,
        flag_irrigation=False,
        road_buffer_m=5.0,
    )

    filtered = apply_false_positive_filters(objects, context, config)

    assert filtered["false_positive_flags"].iloc[0] == [
        "built_up_risk",
        "excluded_zone_risk",
        "quarry_risk",
        "road_risk",
    ]
    assert filtered["false_positive_penalty"].iloc[0] == pytest.approx(1.45)
    assert filtered["road_distance_m"].iloc[0] == pytest.approx(2.0)
    assert filtered["built_up_distance_m"].iloc[0] == pytest.approx(0.0)
    assert filtered["quarry_distance_m"].iloc[0] == pytest.approx(0.0)
    profile = filtered["false_positive_profile"].iloc[0]
    assert profile["road_distance_m"] == pytest.approx(2.0)
    assert profile["flags"] == [
        "built_up_risk",
        "excluded_zone_risk",
        "quarry_risk",
        "road_risk",
    ]


def test_woody_patch_context_adds_false_positive_flag_and_penalty() -> None:
    objects = _objects([box(0, 0, 10, 10)])
    context = FalsePositiveContext(woody_patches=_layer([box(15, 0, 25, 10)]))

    filtered = apply_false_positive_filters(
        objects,
        context,
        FalsePositiveFilterConfig(
            flag_roads=False,
            flag_water=False,
            flag_built_up=False,
            flag_excluded_zones=False,
            flag_quarries=False,
            flag_cloud_shadows=False,
            flag_harvest_patterns=False,
            flag_irrigation=False,
            woody_patch_buffer_m=10.0,
        ),
    )

    assert filtered["false_positive_flags"].iloc[0] == ["woody_patch_risk"]
    assert filtered["false_positive_penalty"].iloc[0] == pytest.approx(0.25)


def test_water_buffer_flags_near_objects_without_intersection() -> None:
    objects = _objects([box(0, 0, 10, 10)])
    context = FalsePositiveContext(water=_layer([box(15, 0, 25, 10)]))

    filtered = apply_false_positive_filters(
        objects,
        context,
        FalsePositiveFilterConfig(
            flag_roads=False,
            flag_built_up=False,
            flag_excluded_zones=False,
            flag_quarries=False,
            flag_woody_patches=False,
            flag_cloud_shadows=False,
            flag_harvest_patterns=False,
            flag_irrigation=False,
            water_buffer_m=10.0,
        ),
    )

    assert filtered["false_positive_flags"].iloc[0] == ["water_risk"]


def test_field_edge_and_linear_object_risks_use_existing_object_columns() -> None:
    objects = _objects(
        [box(0, 0, 10, 10), box(30, 0, 80, 5)],
        near_field_edge=[True, False],
        distances=[5.0, 50.0],
        elongations=[1.2, 10.0],
    )

    filtered = apply_false_positive_filters(
        objects,
        FalsePositiveContext(),
        FalsePositiveFilterConfig(
            flag_roads=False,
            flag_water=False,
            flag_built_up=False,
            flag_excluded_zones=False,
            flag_quarries=False,
            flag_woody_patches=False,
            flag_cloud_shadows=False,
            flag_harvest_patterns=False,
            flag_irrigation=False,
        ),
    )

    assert filtered["false_positive_flags"].iloc[0] == ["field_edge_risk"]
    assert filtered["false_positive_flags"].iloc[1] == ["linear_object_risk"]
    assert filtered["false_positive_penalty"].iloc[0] == pytest.approx(0.20)
    assert filtered["false_positive_penalty"].iloc[1] == pytest.approx(0.25)


def test_missing_enabled_context_layers_add_missing_data_flags() -> None:
    filtered = apply_false_positive_filters(
        _objects([box(0, 0, 10, 10)], missing_flags=[["missing_component_thermal"]]),
        FalsePositiveContext(),
        FalsePositiveFilterConfig(
            flag_roads=True,
            flag_water=True,
            flag_built_up=False,
            flag_excluded_zones=False,
            flag_quarries=False,
            flag_woody_patches=False,
            flag_cloud_shadows=False,
            flag_harvest_patterns=False,
            flag_irrigation=False,
        ),
    )

    assert filtered["false_positive_flags"].iloc[0] == []
    assert filtered["missing_data_flags"].iloc[0] == [
        "missing_component_thermal",
        "missing_context_roads",
        "missing_context_water",
    ]
    assert filtered["false_positive_profile"].iloc[0]["missing_context"] == ["roads", "water"]


def test_shape_profile_flags_small_objects_and_broad_patches() -> None:
    filtered = apply_false_positive_filters(
        _objects(
            [box(0, 0, 10, 10)],
            support_pixels=[5],
            dynamic_flags=[["broad_patch"]],
        ),
        FalsePositiveContext(),
        FalsePositiveFilterConfig(
            flag_roads=False,
            flag_water=False,
            flag_built_up=False,
            flag_excluded_zones=False,
            flag_quarries=False,
            flag_woody_patches=False,
            flag_cloud_shadows=False,
            flag_harvest_patterns=False,
            flag_irrigation=False,
        ),
    )

    assert filtered["false_positive_flags"].iloc[0] == ["broad_patch_risk", "small_object_risk"]
    profile = filtered["false_positive_profile"].iloc[0]
    assert profile["support_pixel_count"] == 5
    assert profile["flags"] == ["broad_patch_risk", "small_object_risk"]


def test_missing_support_pixel_count_does_not_trigger_small_object_profile() -> None:
    filtered = apply_false_positive_filters(
        _objects([box(0, 0, 10, 10)], support_pixels=[float("nan")]),
        FalsePositiveContext(),
        FalsePositiveFilterConfig(
            flag_roads=False,
            flag_water=False,
            flag_built_up=False,
            flag_excluded_zones=False,
            flag_quarries=False,
            flag_woody_patches=False,
            flag_cloud_shadows=False,
            flag_harvest_patterns=False,
            flag_irrigation=False,
        ),
    )

    assert "small_object_risk" not in filtered["false_positive_flags"].iloc[0]
    profile = filtered["false_positive_profile"].iloc[0]
    assert profile["support_pixel_count"] is None
    assert "small_object_risk" not in profile["flags"]


def test_disabled_context_layers_do_not_add_missing_data_flags() -> None:
    filtered = apply_false_positive_filters(
        _objects([box(0, 0, 10, 10)]),
        FalsePositiveContext(),
        FalsePositiveFilterConfig(
            flag_roads=False,
            flag_water=False,
            flag_built_up=False,
            flag_excluded_zones=False,
            flag_quarries=False,
            flag_woody_patches=False,
            flag_cloud_shadows=False,
            flag_harvest_patterns=False,
            flag_irrigation=False,
        ),
    )

    assert filtered["missing_data_flags"].iloc[0] == []


def test_context_crs_mismatch_raises() -> None:
    objects = _objects([box(0, 0, 10, 10)])
    context = FalsePositiveContext(
        roads=_layer([LineString([(0, 0), (10, 0)])]).to_crs("EPSG:3857")
    )

    with pytest.raises(ContextAssignmentError, match="same CRS"):
        apply_false_positive_filters(
            objects,
            context,
            FalsePositiveFilterConfig(
                flag_water=False,
                flag_built_up=False,
                flag_excluded_zones=False,
                flag_quarries=False,
                flag_woody_patches=False,
                flag_cloud_shadows=False,
                flag_harvest_patterns=False,
                flag_irrigation=False,
            ),
        )


def test_false_positive_flags_appear_in_candidate_geojson_feature() -> None:
    candidate = extract_static_candidates(
        [Point(0, 0).buffer(30)],
        candidate_ids=["built-up"],
        landcover_context="built_up",
    )[0]

    feature = static_candidate_to_feature(candidate)

    assert feature["properties"]["false_positive_flags"] == ["built_up_risk"]


def test_false_positive_flags_flow_to_score_labeling_and_passport_surfaces() -> None:
    candidate = extract_static_candidates(
        [Point(0, 0).buffer(30)],
        candidate_ids=["built-up"],
        landcover_context="built_up",
    )[0]

    score_row = static_candidate_to_score_row(candidate, rank=1)
    labeling_row = score_row_to_labeling_row(score_row)
    passport = render_candidate_passport(score_row)

    assert json.loads(str(score_row["false_positive_flags"])) == ["built_up_risk"]
    assert labeling_row["false_positive_flags"] == '["built_up_risk"]'
    assert "built_up_risk" in passport


def test_context_false_positive_flags_flow_to_dynamic_review_artifacts(tmp_path: Path) -> None:
    filtered = apply_false_positive_filters(
        _objects([box(0, 0, 10, 10)]),
        FalsePositiveContext(roads=_layer([LineString([(12, 0), (12, 10)])])),
        FalsePositiveFilterConfig(
            flag_water=False,
            flag_built_up=False,
            flag_excluded_zones=False,
            flag_quarries=False,
            flag_woody_patches=False,
            flag_cloud_shadows=False,
            flag_harvest_patterns=False,
            flag_irrigation=False,
            road_buffer_m=5.0,
        ),
    )
    geojson_path = tmp_path / "candidates.geojson"
    scores_path = tmp_path / "candidate_scores.csv"

    write_candidate_objects_geojson(filtered, geojson_path)
    write_candidate_object_scores_csv(filtered, scores_path)
    score_row = candidate_object_to_score_row(filtered.iloc[0], rank=1)
    labeling_row = score_row_to_labeling_row(score_row)
    passport = render_candidate_passport(score_row)

    geojson = json.loads(geojson_path.read_text(encoding="utf-8"))
    csv_text = scores_path.read_text(encoding="utf-8")
    assert geojson["features"][0]["properties"]["false_positive_flags"] == ["road_risk"]
    assert geojson["features"][0]["properties"]["road_distance_m"] == 2.0
    assert geojson["features"][0]["properties"]["false_positive_profile"]["flags"] == ["road_risk"]
    assert '"road_risk"' in csv_text
    assert "false_positive_profile" in csv_text
    assert labeling_row["false_positive_flags"] == '["road_risk"]'
    assert "road_distance_m" in labeling_row
    assert "false_positive_profile" in labeling_row
    assert "road_risk" in passport
    assert "false_positive_profile" in passport


def _objects(
    geometries: list[object],
    *,
    near_field_edge: list[bool] | None = None,
    distances: list[float] | None = None,
    elongations: list[float] | None = None,
    support_pixels: list[int | float] | None = None,
    dynamic_flags: list[list[str]] | None = None,
    missing_flags: list[list[str]] | None = None,
) -> gpd.GeoDataFrame:
    count = len(geometries)
    return gpd.GeoDataFrame(
        {
            "object_id": [f"object-{index}" for index in range(count)],
            "near_field_edge": near_field_edge or [False] * count,
            "distance_to_field_edge_m": distances or [100.0] * count,
            "elongation": elongations or [1.0] * count,
            "support_pixel_count": support_pixels or [100] * count,
            "dynamic_object_flags": dynamic_flags or [[] for _ in range(count)],
            "missing_data_flags": missing_flags or [[] for _ in range(count)],
            "geometry": geometries,
        },
        geometry="geometry",
        crs=CRS,
    )


def _layer(geometries: list[object]) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({"geometry": geometries}, geometry="geometry", crs=CRS)
