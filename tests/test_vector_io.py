from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import GeometryCollection, Polygon, box

from deep_earth_degasation.geo.crs import CRSValidationError, ensure_metric_crs
from deep_earth_degasation.io.vector import (
    VectorDataError,
    assign_stable_ids,
    load_vector_layer,
)

METRIC_CRS = "EPSG:32637"


def test_load_vector_layer_preserves_role_crs_and_source_ids(tmp_path: Path) -> None:
    path = _write_layer(
        tmp_path / "aoi.gpkg",
        gpd.GeoDataFrame(
            {"source_id": ["aoi-1"], "geometry": [box(500_000, 5_600_000, 500_100, 5_600_100)]},
            crs=METRIC_CRS,
        ),
    )

    layer = load_vector_layer(path, name="pilot_aoi", role="aoi", id_field="source_id")

    assert layer.name == "pilot_aoi"
    assert layer.role == "aoi"
    assert layer.data.crs.to_epsg() == 32637
    assert layer.data["stable_id"].to_list() == ["aoi-1"]


@pytest.mark.parametrize(
    "role",
    ["aoi", "fields", "excluded_zones", "roads", "water", "built_up", "woody_patches"],
)
def test_load_vector_layer_supports_common_context_roles(tmp_path: Path, role: str) -> None:
    path = _write_layer(
        tmp_path / f"{role}.gpkg",
        gpd.GeoDataFrame(
            {"geometry": [box(500_000, 5_600_000, 500_030, 5_600_030)]},
            crs=METRIC_CRS,
        ),
    )

    layer = load_vector_layer(path, name=role, role=role)

    assert layer.role == role
    assert layer.data["stable_id"].iloc[0].startswith("geom-")


def test_geographic_crs_is_rejected_without_explicit_reprojection(tmp_path: Path) -> None:
    path = _write_layer(
        tmp_path / "lonlat.gpkg",
        gpd.GeoDataFrame(
            {"geometry": [box(37.0, 51.0, 37.01, 51.01)]},
            crs="EPSG:4326",
        ),
    )

    with pytest.raises(CRSValidationError, match="projected metre coordinates are required"):
        load_vector_layer(path, name="lonlat", role="aoi", target_crs=METRIC_CRS)


def test_geographic_crs_can_be_reprojected_when_enabled(tmp_path: Path) -> None:
    path = _write_layer(
        tmp_path / "lonlat.gpkg",
        gpd.GeoDataFrame(
            {"geometry": [box(37.0, 51.0, 37.01, 51.01)]},
            crs="EPSG:4326",
        ),
    )

    layer = load_vector_layer(
        path,
        name="lonlat",
        role="aoi",
        target_crs=METRIC_CRS,
        allow_reprojection=True,
    )

    assert layer.data.crs.to_epsg() == 32637
    assert layer.data.total_bounds[0] > 300_000


def test_metric_source_is_reprojected_to_explicit_target_crs(tmp_path: Path) -> None:
    path = _write_layer(
        tmp_path / "metric_other_zone.gpkg",
        gpd.GeoDataFrame(
            {"geometry": [box(37.0, 6_621_000, 1_150.0, 6_622_200)]},
            crs="EPSG:3857",
        ),
    )

    layer = load_vector_layer(
        path,
        name="metric_other_zone",
        role="aoi",
        target_crs=METRIC_CRS,
        allow_reprojection=True,
    )

    assert layer.data.crs.to_epsg() == 32637
    assert layer.data.total_bounds[0] != pytest.approx(37.0)


def test_missing_crs_is_rejected_before_metric_operations() -> None:
    gdf = gpd.GeoDataFrame(
        {"geometry": [box(0, 0, 10, 10)]},
        crs=None,
    )

    with pytest.raises(CRSValidationError, match="has no CRS"):
        ensure_metric_crs(gdf, layer_name="missing_crs")


def test_invalid_geometry_is_repaired_deterministically(tmp_path: Path) -> None:
    bowtie = Polygon([(0, 0), (2, 2), (2, 0), (0, 2), (0, 0)])
    path = _write_layer(
        tmp_path / "invalid.gpkg",
        gpd.GeoDataFrame({"geometry": [bowtie]}, crs=METRIC_CRS),
    )

    layer = load_vector_layer(path, name="invalid", role="fields")

    assert layer.data.geometry.iloc[0].is_valid
    assert layer.issues[0].reason == "invalid_geometry_repaired"


def test_empty_geometry_is_rejected(tmp_path: Path) -> None:
    path = _write_layer(
        tmp_path / "empty.gpkg",
        gpd.GeoDataFrame({"geometry": [GeometryCollection()]}, crs=METRIC_CRS),
    )

    with pytest.raises(VectorDataError, match="empty geometry"):
        load_vector_layer(path, name="empty", role="fields")


def test_stable_ids_do_not_depend_on_feature_order() -> None:
    first = box(0, 0, 10, 10)
    second = box(20, 20, 30, 30)
    gdf = gpd.GeoDataFrame({"geometry": [first, second]}, crs=METRIC_CRS)

    forward = assign_stable_ids(gdf)
    reversed_ids = assign_stable_ids(gdf.iloc[::-1])

    forward_by_geometry = dict(zip(forward.geometry.to_wkt(), forward["stable_id"], strict=True))
    reversed_by_geometry = dict(
        zip(reversed_ids.geometry.to_wkt(), reversed_ids["stable_id"], strict=True)
    )

    assert forward_by_geometry == reversed_by_geometry


def test_blank_source_ids_fall_back_to_geometry_hash() -> None:
    gdf = gpd.GeoDataFrame(
        {"source_id": ["field-1", ""], "geometry": [box(0, 0, 10, 10), box(20, 20, 30, 30)]},
        crs=METRIC_CRS,
    )

    with_ids = assign_stable_ids(gdf, id_field="source_id")

    assert with_ids["stable_id"].iloc[0] == "field-1"
    assert with_ids["stable_id"].iloc[1].startswith("geom-")


def _write_layer(path: Path, gdf: gpd.GeoDataFrame) -> Path:
    gdf.to_file(path, driver="GPKG")
    return path
