from __future__ import annotations

import geopandas as gpd
import pytest
from shapely.geometry import box

from deep_earth_degasation.context.fields import (
    ContextAssignmentError,
    assign_field_context,
)
from deep_earth_degasation.context.landcover import assign_landcover_context

METRIC_CRS = "EPSG:32637"


def test_assign_field_context_adds_field_id_and_edge_distance() -> None:
    objects = _objects([box(20, 20, 40, 40)])
    fields = _fields([("field-a", box(0, 0, 100, 100))])

    assigned = assign_field_context(
        objects,
        fields,
        min_distance_to_field_edge_m=15,
    )

    assert assigned["field_id"].iloc[0] == "field-a"
    assert assigned["distance_to_field_edge_m"].iloc[0] == pytest.approx(20.0)
    assert bool(assigned["near_field_edge"].iloc[0]) is False
    assert assigned["field_context_flags"].iloc[0] == []


def test_assign_field_context_retains_and_flags_near_edge_objects() -> None:
    objects = _objects([box(95, 20, 110, 40)])
    fields = _fields([("field-a", box(0, 0, 100, 100))])

    assigned = assign_field_context(
        objects,
        fields,
        min_distance_to_field_edge_m=15,
    )

    assert len(assigned) == 1
    assert assigned["field_id"].iloc[0] == "field-a"
    assert assigned["distance_to_field_edge_m"].iloc[0] == pytest.approx(0.0)
    assert bool(assigned["near_field_edge"].iloc[0]) is True
    assert assigned["field_context_flags"].iloc[0] == ["near_field_edge"]


def test_assign_field_context_retains_unassigned_objects_with_flag() -> None:
    objects = _objects([box(200, 200, 220, 220)])
    fields = _fields([("field-a", box(0, 0, 100, 100))])

    assigned = assign_field_context(objects, fields)

    assert len(assigned) == 1
    assert assigned["field_id"].iloc[0] == ""
    assert assigned["distance_to_field_edge_m"].iloc[0] is None
    assert bool(assigned["near_field_edge"].iloc[0]) is False
    assert assigned["field_context_flags"].iloc[0] == ["missing_field_context"]


def test_assign_field_context_uses_largest_intersection_area() -> None:
    objects = _objects([box(80, 0, 130, 100)])
    fields = _fields(
        [
            ("field-a", box(0, 0, 100, 100)),
            ("field-b", box(100, 0, 200, 100)),
        ]
    )

    assigned = assign_field_context(objects, fields)

    assert assigned["field_id"].iloc[0] == "field-b"


def test_assign_landcover_context_adds_single_branch() -> None:
    objects = _objects([box(10, 10, 50, 50)])
    landcover = _landcover([("crops", box(0, 0, 100, 100))])

    assigned = assign_landcover_context(
        objects,
        landcover,
        classes={"cropland": ["crops"], "forest": ["trees"]},
    )

    assert assigned["landcover_branch"].iloc[0] == "cropland"
    assert assigned["dominant_landcover_branch"].iloc[0] == "cropland"
    assert assigned["landcover_proportions"].iloc[0] == {"cropland": 1.0}
    assert assigned["landcover_context_flags"].iloc[0] == []


def test_assign_landcover_context_retains_mixed_branch_proportions() -> None:
    objects = _objects([box(0, 0, 100, 100)])
    landcover = _landcover(
        [
            ("crops", box(0, 0, 60, 100)),
            ("trees", box(60, 0, 100, 100)),
        ]
    )

    assigned = assign_landcover_context(
        objects,
        landcover,
        classes={"cropland": ["crops"], "forest": ["trees"]},
    )

    assert assigned["landcover_branch"].iloc[0] == "mixed"
    assert assigned["dominant_landcover_branch"].iloc[0] == "cropland"
    assert assigned["landcover_proportions"].iloc[0] == {
        "cropland": 0.6,
        "forest": 0.4,
    }
    assert assigned["landcover_context_flags"].iloc[0] == ["mixed_landcover"]


def test_assign_landcover_context_retains_unknown_when_context_missing() -> None:
    objects = _objects([box(200, 200, 250, 250)])
    landcover = _landcover([("crops", box(0, 0, 100, 100))])

    assigned = assign_landcover_context(
        objects,
        landcover,
        classes={"cropland": ["crops"]},
    )

    assert assigned["landcover_branch"].iloc[0] == "unknown"
    assert assigned["dominant_landcover_branch"].iloc[0] == "unknown"
    assert assigned["landcover_proportions"].iloc[0] == {}
    assert assigned["landcover_context_flags"].iloc[0] == ["missing_landcover_context"]


def test_context_assignment_rejects_mismatched_crs() -> None:
    objects = _objects([box(10, 10, 20, 20)])
    fields = _fields([("field-a", box(0, 0, 100, 100))]).to_crs("EPSG:3857")

    with pytest.raises(ContextAssignmentError, match="same CRS"):
        assign_field_context(objects, fields)


def test_context_assignment_rejects_missing_required_columns() -> None:
    objects = _objects([box(10, 10, 20, 20)])
    fields = gpd.GeoDataFrame(
        {"geometry": [box(0, 0, 100, 100)]},
        crs=METRIC_CRS,
    )

    with pytest.raises(ContextAssignmentError, match="field_id"):
        assign_field_context(objects, fields)


def _objects(geometries: list[object]) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "object_id": [f"object-{index}" for index in range(len(geometries))],
            "geometry": geometries,
        },
        crs=METRIC_CRS,
    )


def _fields(fields: list[tuple[str, object]]) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "field_id": [field_id for field_id, _ in fields],
            "geometry": [geometry for _, geometry in fields],
        },
        crs=METRIC_CRS,
    )


def _landcover(classes: list[tuple[str, object]]) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "landcover_class": [landcover_class for landcover_class, _ in classes],
            "geometry": [geometry for _, geometry in classes],
        },
        crs=METRIC_CRS,
    )
