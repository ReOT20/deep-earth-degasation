from __future__ import annotations

import geopandas as gpd
import numpy as np
from shapely.geometry import box

from deep_earth_degasation.anomaly.residuals import (
    peer_residual_layers,
    residual_layers_metadata,
    temporal_residual_layers,
)
from deep_earth_degasation.features.types import DynamicFeatureLayer


def test_peer_residual_layers_use_available_field_context() -> None:
    fields = gpd.GeoDataFrame(
        {
            "crop_type": ["wheat", "wheat"],
            "phenology_stage": ["early", "early"],
            "geometry": [box(0, 0, 1, 1), box(1, 0, 2, 1)],
        },
        geometry="geometry",
        crs="EPSG:32637",
    )
    field_ids = np.array([[1, 1, 2, 2]])
    feature = _feature("NDMI", [[0.5, 0.5, 0.5, 0.1]])

    result = peer_residual_layers(
        (feature,),
        field_ids,
        fields,
        component_for_feature=lambda _: "moisture",
    )

    assert result.missing_data_flags == ()
    assert len(result.layers) == 1
    layer = result.layers[0]
    assert layer.residual_type == "peer"
    assert layer.data[0, 3] > 1.0


def test_peer_residual_layers_flag_missing_context() -> None:
    fields = gpd.GeoDataFrame({"geometry": [box(0, 0, 1, 1)]}, geometry="geometry")

    result = peer_residual_layers(
        (_feature("NDMI", [[0.5, 0.1]]),),
        np.array([[1, 1]]),
        fields,
        component_for_feature=lambda _: "moisture",
    )

    assert result.layers == ()
    assert "missing_peer_residual_context" in result.missing_data_flags


def test_temporal_residual_layers_use_repeated_feature_dates() -> None:
    early = _feature("NDMI", [[0.5, 0.5]], date="2024-05-01")
    late = _feature("NDMI", [[0.5, 0.1]], date="2024-06-01")

    result = temporal_residual_layers(
        (early, late),
        component_for_feature=lambda _: "moisture",
    )

    assert result.missing_data_flags == ()
    assert len(result.layers) == 2
    assert {layer.residual_type for layer in result.layers} == {"temporal"}
    assert result.layers[1].data[0, 1] > 0.0


def test_temporal_residual_layers_flag_missing_repeated_dates() -> None:
    result = temporal_residual_layers(
        (_feature("NDMI", [[0.5, 0.1]], date="2024-05-01"),),
        component_for_feature=lambda _: "moisture",
    )

    assert result.layers == ()
    assert "missing_temporal_residual_context" in result.missing_data_flags


def test_temporal_residual_layers_require_distinct_dates() -> None:
    result = temporal_residual_layers(
        (
            _feature("NDMI", [[0.5, 0.1]], date="2024-05-01"),
            _feature("NDMI", [[0.4, 0.2]], date="2024-05-01"),
        ),
        component_for_feature=lambda _: "moisture",
    )

    assert result.layers == ()
    assert "missing_temporal_residual_NDMI" in result.missing_data_flags
    assert "missing_temporal_residual_context" in result.missing_data_flags


def test_temporal_residual_layers_drop_constant_all_nan_products() -> None:
    result = temporal_residual_layers(
        (
            _feature("NDMI", [[0.5, 0.5]], date="2024-05-01"),
            _feature("NDMI", [[0.5, 0.5]], date="2024-06-01"),
        ),
        component_for_feature=lambda _: "moisture",
    )

    assert result.layers == ()
    assert "missing_temporal_residual_NDMI" in result.missing_data_flags
    assert "missing_temporal_residual_context" in result.missing_data_flags


def test_temporal_residual_layers_drop_all_nan_products() -> None:
    result = temporal_residual_layers(
        (
            _feature("NDMI", [[np.nan, np.nan]], date="2024-05-01"),
            _feature("NDMI", [[np.nan, np.nan]], date="2024-06-01"),
        ),
        component_for_feature=lambda _: "moisture",
    )

    assert result.layers == ()
    assert "missing_temporal_residual_NDMI" in result.missing_data_flags
    assert "missing_temporal_residual_context" in result.missing_data_flags


def test_residual_metadata_is_json_ready() -> None:
    result = temporal_residual_layers(
        (
            _feature("NDMI", [[0.5, 0.5]], date="2024-05-01"),
            _feature("NDMI", [[0.5, 0.1]], date="2024-06-01"),
        ),
        component_for_feature=lambda _: "moisture",
    )

    metadata = residual_layers_metadata(result.layers, missing_data_flags=("missing_peer",))

    assert metadata["schema_version"] == "hierarchical_residuals.v1"
    assert metadata["missing_data_flags"] == ["missing_peer"]
    layers = metadata["layers"]
    assert isinstance(layers, list)
    assert layers[0]["residual_type"] == "temporal"


def _feature(
    name: str,
    data: list[list[float]],
    *,
    date: str | None = "2024-05-01",
) -> DynamicFeatureLayer:
    return DynamicFeatureLayer(
        name=name,
        data=np.array(data, dtype=float),
        sensor="sentinel2",
        date=date,
        source_layer_ids=(f"{name}_{date}",),
        evidence_direction="lower_values_indicate_moisture_stress",
    )
