from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
from affine import Affine
from rasterio.coords import BoundingBox

from deep_earth_degasation.features.sar import sentinel1_features_from_stack
from deep_earth_degasation.features.spectral import (
    compute_brightness,
    compute_bsi,
    compute_msi,
    compute_ndmi,
    compute_ndvi,
    compute_ndwi,
    sentinel2_features_from_stack,
)
from deep_earth_degasation.features.thermal import landsat_thermal_features_from_stack
from deep_earth_degasation.features.types import DynamicFeatureLayer
from deep_earth_degasation.io.raster_stack import RasterLayer, RasterStack
from deep_earth_degasation.pipeline.manifest import (
    AOISpec,
    PreparedStackManifest,
    QualitySpec,
    RasterLayerSpec,
    RunManifestSpec,
)


def test_sentinel2_array_feature_helpers_compute_expected_values() -> None:
    nir = np.array([[0.8, 0.4]], dtype=float)
    red = np.array([[0.2, 0.2]], dtype=float)
    swir1 = np.array([[0.2, 0.4]], dtype=float)
    green = np.array([[0.5, 0.3]], dtype=float)
    blue = np.array([[0.1, 0.2]], dtype=float)

    assert np.allclose(compute_ndvi(nir, red).data, np.array([[0.6, 1 / 3]]), rtol=1e-6)
    assert np.allclose(compute_ndmi(nir, swir1).data, np.array([[0.6, 0.0]]), rtol=1e-6)
    assert np.allclose(
        compute_ndwi(nir, green).data, np.array([[-0.23076923, -0.14285714]]), rtol=1e-6
    )
    assert np.allclose(compute_msi(nir, swir1).data, np.array([[0.25, 1.0]]), rtol=1e-6)
    assert np.isfinite(compute_bsi(red, nir, swir1, blue).data).all()
    assert np.allclose(compute_brightness(nir, red).data, np.array([[0.5, 0.3]]), rtol=1e-6)


def test_sentinel2_features_from_stack_preserve_provenance_and_missing_optional_flags() -> None:
    stack = _stack(
        [
            _layer("s2_ndvi", "sentinel2", "NDVI", "2024-05-15", [[0.1, 0.2]]),
            _layer("s2_ndmi", "sentinel2", "NDMI", "2024-05-15", [[0.3, 0.4]]),
            _layer("s2_ndwi", "sentinel2", "NDWI", "2024-05-15", [[0.5, 0.6]]),
            _layer("s2_msi", "sentinel2", "MSI", "2024-05-15", [[0.7, 0.8]]),
            _layer("s2_bsi", "sentinel2", "BSI", "2024-05-15", [[0.9, 1.0]]),
            _layer("s2_brightness", "sentinel2", "brightness", "2024-05-15", [[1.1, 1.2]]),
        ]
    )

    result = sentinel2_features_from_stack(stack)
    ndvi = _feature(result.features, "NDVI")

    assert len(result.features) == 6
    assert ndvi.source_layer_ids == ("s2_ndvi",)
    assert ndvi.date == "2024-05-15"
    assert ndvi.evidence_direction == "lower_values_indicate_vegetation_stress"
    assert "missing_sentinel2_red_edge" in result.missing_data_flags
    assert "missing_sentinel2_EVI" in result.missing_data_flags


def test_sentinel2_features_from_stack_preserve_all_dates() -> None:
    stack = _stack(
        [
            _layer("s2_ndvi_early", "sentinel2", "NDVI", "2024-05-15", [[0.1]]),
            _layer("s2_ndvi_late", "sentinel2", "NDVI", "2024-06-15", [[0.2]]),
        ]
    )

    result = sentinel2_features_from_stack(stack)
    ndvi_features = [feature for feature in result.features if feature.name == "NDVI"]

    assert [feature.date for feature in ndvi_features] == ["2024-05-15", "2024-06-15"]
    assert [feature.source_layer_ids for feature in ndvi_features] == [
        ("s2_ndvi_early",),
        ("s2_ndvi_late",),
    ]
    assert "missing_sentinel2_NDVI" not in result.missing_data_flags


def test_sentinel2_features_from_stack_flags_missing_required_prepared_layers() -> None:
    result = sentinel2_features_from_stack(_stack([]))

    assert result.features == ()
    assert "missing_sentinel2_NDVI" in result.missing_data_flags
    assert "missing_sentinel2_brightness" in result.missing_data_flags


def test_sentinel1_features_compute_ratio_and_temporal_difference() -> None:
    stack = _stack(
        [
            _layer("vv_1", "sentinel1", "VV", "2024-05-01", [[2.0, 4.0]]),
            _layer("vv_2", "sentinel1", "VV", "2024-05-15", [[3.0, 1.0]]),
            _layer("vh_1", "sentinel1", "VH", "2024-05-01", [[1.0, 2.0]]),
        ]
    )

    result = sentinel1_features_from_stack(stack)

    assert np.allclose(_feature(result.features, "VV_VH_ratio").data, np.array([[2.0, 2.0]]))
    assert np.allclose(
        _feature(result.features, "temporal_difference").data, np.array([[1.0, -3.0]])
    )
    assert _feature(result.features, "VV_VH_ratio").source_layer_ids == ("vv_1", "vh_1")


def test_sentinel1_vv_vh_ratio_pairs_only_matching_dates() -> None:
    stack = _stack(
        [
            _layer("vv_early", "sentinel1", "VV", "2024-05-01", [[2.0]]),
            _layer("vv_late", "sentinel1", "VV", "2024-05-15", [[4.0]]),
            _layer("vh_late", "sentinel1", "VH", "2024-05-15", [[2.0]]),
        ]
    )

    result = sentinel1_features_from_stack(stack)
    ratio_features = [feature for feature in result.features if feature.name == "VV_VH_ratio"]

    assert len(ratio_features) == 1
    assert ratio_features[0].date == "2024-05-15"
    assert ratio_features[0].source_layer_ids == ("vv_late", "vh_late")
    assert np.allclose(ratio_features[0].data, np.array([[2.0]]))
    assert "missing_sentinel1_VH_for_ratio_2024-05-01" in result.missing_data_flags


def test_sentinel1_vv_vh_ratio_does_not_cross_mismatched_dates() -> None:
    stack = _stack(
        [
            _layer("vv_early", "sentinel1", "VV", "2024-05-01", [[2.0]]),
            _layer("vh_late", "sentinel1", "VH", "2024-05-15", [[1.0]]),
        ]
    )

    result = sentinel1_features_from_stack(stack)

    assert all(feature.name != "VV_VH_ratio" for feature in result.features)
    assert "missing_sentinel1_VH_for_ratio_2024-05-01" in result.missing_data_flags
    assert "missing_sentinel1_VV_for_ratio_2024-05-15" in result.missing_data_flags


def test_sentinel1_missing_inputs_return_flags() -> None:
    result = sentinel1_features_from_stack(_stack([]))

    assert result.features == ()
    assert "missing_sentinel1_VV" in result.missing_data_flags
    assert "missing_sentinel1_VH" in result.missing_data_flags
    assert "missing_sentinel1_VV_VH_ratio_inputs" in result.missing_data_flags


def test_sentinel1_shape_mismatch_is_flagged_without_broadcasting() -> None:
    stack = _stack(
        [
            _layer("vv_1", "sentinel1", "VV", "2024-05-01", [[1.0, 2.0]]),
            _layer("vh_1", "sentinel1", "VH", "2024-05-01", [[1.0], [2.0]]),
        ]
    )

    result = sentinel1_features_from_stack(stack)

    assert "VV_VH_ratio_shape_mismatch_2024-05-01" in result.missing_data_flags
    assert all(feature.name != "VV_VH_ratio" for feature in result.features)


def test_landsat_lst_feature_preserves_provenance_and_direction() -> None:
    stack = _stack([_layer("lst_1", "landsat", "LST", "2024-05-17", [[300.0]])])

    result = landsat_thermal_features_from_stack(stack)
    feature = _feature(result.features, "LST")

    assert feature.source_layer_ids == ("lst_1",)
    assert feature.date == "2024-05-17"
    assert feature.evidence_direction == "higher_values_indicate_heat_or_thermal_support"


def test_missing_landsat_lst_returns_missing_flag() -> None:
    result = landsat_thermal_features_from_stack(_stack([]))

    assert result.features == ()
    assert result.missing_data_flags == ("missing_landsat_LST",)


def _feature(features: tuple[DynamicFeatureLayer, ...], name: str) -> DynamicFeatureLayer:
    for feature in features:
        if feature.name == name:
            return feature
    raise AssertionError(f"Feature {name!r} not found.")


def _stack(layers: list[RasterLayer]) -> RasterStack:
    return RasterStack(
        manifest=PreparedStackManifest(
            path=Path(__file__),
            run=RunManifestSpec(id="test", description=None, created_by=None, notes=None),
            crs="EPSG:32637",
            resolution_m=10,
            aoi=AOISpec(path=Path(__file__), crs="EPSG:32637"),
            vectors={},
            raster_layers=tuple(layer.spec for layer in layers),
            quality=QualitySpec(),
            notes=(),
        ),
        layers=tuple(layers),
    )


def _layer(
    layer_id: str,
    sensor: str,
    feature_name: str,
    date: str,
    data: list[list[float]],
) -> RasterLayer:
    base_spec = RasterLayerSpec(
        id=layer_id,
        sensor=sensor,
        feature_name=feature_name,
        date=date,
        path=Path(__file__),
        crs="EPSG:32637",
        resolution_m=10,
        nodata=None,
        role="input_feature",
    )
    return RasterLayer(
        spec=replace(base_spec),
        data=np.array(data, dtype=float),
        crs="EPSG:32637",
        transform=Affine.identity(),
        bounds=BoundingBox(left=0.0, bottom=0.0, right=1.0, top=1.0),
        resolution_m=(10.0, 10.0),
        shape=np.array(data).shape,
    )
