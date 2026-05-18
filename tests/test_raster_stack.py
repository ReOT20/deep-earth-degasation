from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
import yaml
from rasterio.transform import from_origin
from shapely.geometry import box

from deep_earth_degasation.io.raster_stack import (
    RasterStackError,
    load_raster_stack,
    write_run_manifest,
)
from deep_earth_degasation.pipeline.manifest import (
    ManifestError,
    load_prepared_stack_manifest,
)

METRIC_CRS = "EPSG:32637"
BASE_TRANSFORM = from_origin(500_000, 5_600_040, 10, 10)
COARSE_TRANSFORM = from_origin(500_000, 5_600_060, 30, 30)


def test_prepared_stack_manifest_template_loads() -> None:
    manifest = load_prepared_stack_manifest(
        Path("data/manifests/prepared_stack_manifest.template.yaml")
    )

    assert manifest.run.id == "lipetsk_voronezh_dynamic_pilot_template"
    assert manifest.crs == METRIC_CRS
    assert manifest.raster_layers[0].feature_name == "NDVI"
    assert manifest.vectors["quarries"].role == "quarries"
    assert manifest.vectors["woody_patches"].role == "woody_patches"
    assert manifest.weather_context is not None
    assert (
        manifest.weather_context.path
        == (Path.cwd() / "data/context/weather_events.template.yaml").resolve()
    )
    assert (
        manifest.raster_layers[0].path
        == (Path.cwd() / "data/prepared/sentinel2/2024-05-15_NDVI.tif").resolve()
    )


def test_load_raster_stack_applies_nodata_mask_and_records_provenance(tmp_path: Path) -> None:
    ndvi = np.array(
        [
            [1.0, -9999.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ],
        dtype=np.float32,
    )
    ndmi = np.full((4, 4), 2.0, dtype=np.float32)
    mask = np.ones((4, 4), dtype=np.uint8)
    mask[1, 1] = 0
    _write_raster(tmp_path / "ndvi.tif", ndvi, nodata=-9999)
    _write_raster(tmp_path / "ndmi.tif", ndmi, nodata=-9999)
    _write_raster(tmp_path / "valid_mask.tif", mask, nodata=0, dtype="uint8")
    _write_aoi(tmp_path / "aoi.geojson")
    manifest_path = _write_manifest(tmp_path)

    manifest = load_prepared_stack_manifest(manifest_path)
    stack = load_raster_stack(manifest)
    run_manifest_path = tmp_path / "run_manifest.json"
    write_run_manifest(stack, run_manifest_path)
    provenance = json.loads(run_manifest_path.read_text(encoding="utf-8"))

    assert len(stack.layers) == 2
    assert stack.layers[0].shape == (4, 4)
    assert np.isnan(stack.layers[0].data[0, 1])
    assert np.isnan(stack.layers[0].data[1, 1])
    assert stack.layers[1].spec.date == "2024-05-20"
    assert provenance["run_id"] == "synthetic_dynamic_test"
    assert provenance["layers"][0]["feature_name"] == "NDVI"
    assert provenance["layers"][0]["quality_mask_path"].endswith("valid_mask.tif")


def test_load_raster_stack_allows_mixed_resolution_layers(tmp_path: Path) -> None:
    _write_aoi(tmp_path / "aoi.geojson")
    _write_raster(tmp_path / "ndvi.tif", np.ones((4, 4), dtype=np.float32), nodata=-9999)
    _write_raster(
        tmp_path / "lst.tif",
        np.full((2, 2), 300.0, dtype=np.float32),
        nodata=-9999,
        transform=COARSE_TRANSFORM,
    )
    manifest = load_prepared_stack_manifest(
        _write_manifest(
            tmp_path,
            layers=[
                _layer("ndvi", "NDVI", "2024-05-15", "ndvi.tif", resolution_m=10),
                _layer("lst", "LST", "2024-05-17", "lst.tif", resolution_m=30),
            ],
        )
    )

    stack = load_raster_stack(manifest)

    assert [layer.spec.id for layer in stack.layers] == ["ndvi", "lst"]
    assert stack.layers[0].shape == (4, 4)
    assert stack.layers[1].shape == (2, 2)


def test_manifest_rejects_duplicate_layer_ids(tmp_path: Path) -> None:
    _write_aoi(tmp_path / "aoi.geojson")
    _write_raster(tmp_path / "ndvi.tif", np.ones((4, 4), dtype=np.float32), nodata=-9999)
    manifest_path = _write_manifest(
        tmp_path,
        layers=[
            _layer("duplicate", "NDVI", "2024-05-15", "ndvi.tif"),
            _layer("duplicate", "NDMI", "2024-05-20", "ndvi.tif"),
        ],
    )

    with pytest.raises(ManifestError, match="Duplicate raster layer id"):
        load_prepared_stack_manifest(manifest_path)


def test_manifest_rejects_quoted_quality_boolean(tmp_path: Path) -> None:
    _write_aoi(tmp_path / "aoi.geojson")
    _write_raster(tmp_path / "ndvi.tif", np.ones((4, 4), dtype=np.float32), nodata=-9999)
    manifest_path = _write_manifest(
        tmp_path, layers=[_layer("ndvi", "NDVI", "2024-05-15", "ndvi.tif")]
    )
    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    data["quality"]["reject_mismatched_grid"] = "false"
    manifest_path.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(ManifestError, match=r"quality\.reject_mismatched_grid"):
        load_prepared_stack_manifest(manifest_path)


def test_manifest_rejects_quoted_vector_required_boolean(tmp_path: Path) -> None:
    _write_aoi(tmp_path / "aoi.geojson")
    _write_raster(tmp_path / "ndvi.tif", np.ones((4, 4), dtype=np.float32), nodata=-9999)
    manifest_path = _write_manifest(
        tmp_path, layers=[_layer("ndvi", "NDVI", "2024-05-15", "ndvi.tif")]
    )
    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    data["vectors"]["fields"]["required"] = "false"
    manifest_path.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(ManifestError, match=r"vectors\.fields\.required"):
        load_prepared_stack_manifest(manifest_path)


def test_load_raster_stack_rejects_crs_mismatch(tmp_path: Path) -> None:
    _write_aoi(tmp_path / "aoi.geojson")
    _write_raster(
        tmp_path / "ndvi.tif", np.ones((4, 4), dtype=np.float32), nodata=-9999, crs="EPSG:3857"
    )
    manifest = load_prepared_stack_manifest(
        _write_manifest(
            tmp_path,
            layers=[
                _layer("ndvi", "NDVI", "2024-05-15", "ndvi.tif"),
                _layer("ndmi", "NDMI", "2024-05-20", "ndmi.tif"),
            ],
        )
    )

    with pytest.raises(RasterStackError, match="CRS"):
        load_raster_stack(manifest)


def test_load_raster_stack_rejects_grid_mismatch(tmp_path: Path) -> None:
    _write_aoi(tmp_path / "aoi.geojson")
    _write_raster(tmp_path / "ndvi.tif", np.ones((4, 4), dtype=np.float32), nodata=-9999)
    shifted_transform = from_origin(500_010, 5_600_040, 10, 10)
    _write_raster(
        tmp_path / "ndmi.tif",
        np.ones((4, 4), dtype=np.float32),
        nodata=-9999,
        transform=shifted_transform,
    )
    manifest = load_prepared_stack_manifest(
        _write_manifest(
            tmp_path,
            layers=[
                _layer("ndvi", "NDVI", "2024-05-15", "ndvi.tif"),
                _layer("ndmi", "NDMI", "2024-05-20", "ndmi.tif"),
            ],
        )
    )

    with pytest.raises(RasterStackError, match="grid"):
        load_raster_stack(manifest)


def test_load_raster_stack_rejects_resolution_mismatch(tmp_path: Path) -> None:
    _write_aoi(tmp_path / "aoi.geojson")
    _write_raster(tmp_path / "ndvi.tif", np.ones((4, 4), dtype=np.float32), nodata=-9999)
    manifest = load_prepared_stack_manifest(
        _write_manifest(
            tmp_path, layers=[_layer("ndvi", "NDVI", "2024-05-15", "ndvi.tif", resolution_m=20)]
        )
    )

    with pytest.raises(RasterStackError, match="resolution"):
        load_raster_stack(manifest)


def test_load_raster_stack_rejects_raster_that_does_not_cover_aoi(tmp_path: Path) -> None:
    _write_aoi(tmp_path / "aoi.geojson", bounds=(500_100, 5_600_100, 500_120, 5_600_120))
    _write_raster(tmp_path / "ndvi.tif", np.ones((4, 4), dtype=np.float32), nodata=-9999)
    manifest = load_prepared_stack_manifest(
        _write_manifest(tmp_path, layers=[_layer("ndvi", "NDVI", "2024-05-15", "ndvi.tif")])
    )

    with pytest.raises(RasterStackError, match="AOI"):
        load_raster_stack(manifest)


def _write_manifest(tmp_path: Path, *, layers: list[dict[str, object]] | None = None) -> Path:
    manifest = {
        "run": {
            "id": "synthetic_dynamic_test",
            "description": "Synthetic public test manifest.",
            "created_by": "pytest",
        },
        "crs": METRIC_CRS,
        "resolution_m": 10,
        "aoi": {"path": "aoi.geojson", "crs": METRIC_CRS},
        "vectors": {
            "fields": {
                "path": "fields.geojson",
                "crs": METRIC_CRS,
                "role": "fields",
                "required": False,
                "id_field": "field_id",
            }
        },
        "raster_layers": layers
        or [
            _layer("ndvi", "NDVI", "2024-05-15", "ndvi.tif", quality_mask_path="valid_mask.tif"),
            _layer("ndmi", "NDMI", "2024-05-20", "ndmi.tif"),
        ],
        "quality": {
            "reject_mismatched_crs": True,
            "reject_mismatched_grid": True,
            "allow_resampling": False,
        },
        "notes": ["synthetic test only"],
    }
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    return path


def _layer(
    layer_id: str,
    feature_name: str,
    date: str,
    path: str,
    *,
    resolution_m: int = 10,
    quality_mask_path: str | None = None,
) -> dict[str, object]:
    layer = {
        "id": layer_id,
        "sensor": "sentinel2",
        "feature_name": feature_name,
        "date": date,
        "path": path,
        "crs": METRIC_CRS,
        "resolution_m": resolution_m,
        "nodata": -9999,
        "role": "input_feature",
    }
    if quality_mask_path is not None:
        layer["quality_mask_path"] = quality_mask_path
    return layer


def _write_aoi(path: Path, *, bounds: tuple[float, float, float, float] | None = None) -> None:
    minx, miny, maxx, maxy = bounds or (500_005, 5_600_005, 500_035, 5_600_035)
    gdf = gpd.GeoDataFrame(
        {"name": ["synthetic_aoi"], "geometry": [box(minx, miny, maxx, maxy)]}, crs=METRIC_CRS
    )
    gdf.to_file(path, driver="GeoJSON")


def _write_raster(
    path: Path,
    data: np.ndarray,
    *,
    nodata: float,
    crs: str = METRIC_CRS,
    transform: object = BASE_TRANSFORM,
    dtype: str = "float32",
) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(data.astype(dtype), 1)
