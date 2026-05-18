from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from deep_earth_degasation.anomaly.composite import composite_anomaly_map
from deep_earth_degasation.anomaly.field_normalization import (
    FieldAnomalyLayer,
    field_normalized_anomaly,
)
from deep_earth_degasation.features.types import DynamicFeatureLayer


@dataclass(frozen=True)
class _ComponentConfig:
    inputs: list[str]
    weight: float


def test_field_normalization_suppresses_regional_uniform_shift() -> None:
    early = _feature(
        "NDMI",
        [[0.40, 0.42, 0.44, 0.46], [0.50, 0.52, 0.54, 0.56]],
        direction="lower_values_indicate_moisture_stress",
        date="2024-05-01",
    )
    drought_shift = _feature(
        "NDMI",
        [[0.20, 0.22, 0.24, 0.26], [0.30, 0.32, 0.34, 0.36]],
        direction="lower_values_indicate_moisture_stress",
        date="2024-06-01",
    )
    field_ids = np.array([["a", "a", "a", "a"], ["b", "b", "b", "b"]], dtype=object)

    early_anomaly = field_normalized_anomaly(early, field_ids, component="moisture")
    drought_anomaly = field_normalized_anomaly(drought_shift, field_ids, component="moisture")

    assert np.allclose(early_anomaly.data, drought_anomaly.data)


def test_lower_value_stress_features_become_positive_local_anomalies() -> None:
    feature = _feature(
        "NDMI",
        [[0.50, 0.51, 0.49, 0.20]],
        direction="lower_values_indicate_moisture_stress",
    )
    field_ids = np.array([["field-a", "field-a", "field-a", "field-a"]], dtype=object)

    anomaly = field_normalized_anomaly(feature, field_ids, component="moisture")

    assert anomaly.data[0, 3] > 1.0
    assert anomaly.component == "moisture"
    assert anomaly.residual_type == "field"
    assert anomaly.source_feature_names == ("NDMI",)
    assert anomaly.source_layer_ids == ("NDMI_layer",)


def test_numeric_field_ids_are_used_for_masks() -> None:
    feature = _feature(
        "NDMI",
        [[0.50, 0.51, 0.49, 0.20]],
        direction="lower_values_indicate_moisture_stress",
    )
    field_ids = np.array([[1, 1, 1, 1]])

    anomaly = field_normalized_anomaly(feature, field_ids, component="moisture")

    assert anomaly.data[0, 3] > 1.0
    assert not np.isnan(anomaly.data).all()


def test_zero_label_background_is_not_normalized_as_a_field() -> None:
    feature = _feature(
        "NDMI",
        [[0.50, 0.51, 0.20, 0.99]],
        direction="lower_values_indicate_moisture_stress",
    )
    field_ids = np.array([[1, 1, 1, 0]])

    anomaly = field_normalized_anomaly(feature, field_ids, component="moisture")

    assert anomaly.data[0, 2] > 1.0
    assert np.isnan(anomaly.data[0, 3])


def test_float_field_ids_skip_nan_background() -> None:
    feature = _feature(
        "NDMI",
        [[0.50, 0.51, 0.20, 0.99]],
        direction="lower_values_indicate_moisture_stress",
    )
    field_ids = np.array([[1.0, 1.0, 1.0, np.nan]])

    anomaly = field_normalized_anomaly(feature, field_ids, component="moisture")

    assert anomaly.data[0, 2] > 1.0
    assert np.isnan(anomaly.data[0, 3])


def test_higher_value_stress_features_keep_positive_direction() -> None:
    feature = _feature(
        "MSI",
        [[1.0, 1.1, 0.9, 2.0]],
        direction="higher_values_indicate_moisture_stress",
    )
    field_ids = np.array([["field-a", "field-a", "field-a", "field-a"]], dtype=object)

    anomaly = field_normalized_anomaly(feature, field_ids, component="moisture")

    assert anomaly.data[0, 3] > 1.0


def test_sar_two_sided_features_use_absolute_anomaly() -> None:
    feature = _feature(
        "VV_VH_ratio",
        [[1.0, 1.1, 0.9, 0.1]],
        direction="higher_or_lower_values_indicate_sar_anomaly",
    )
    field_ids = np.array([["field-a", "field-a", "field-a", "field-a"]], dtype=object)

    anomaly = field_normalized_anomaly(feature, field_ids, component="sar")

    assert anomaly.data[0, 3] > 1.0


def test_minimum_valid_pixels_skip_low_quality_fields() -> None:
    feature = _feature(
        "NDMI",
        [[0.5, np.nan, 0.4, 0.2]],
        direction="lower_values_indicate_moisture_stress",
    )
    field_ids = np.array([["field-a", "field-a", "field-b", "field-b"]], dtype=object)

    anomaly = field_normalized_anomaly(
        feature,
        field_ids,
        component="moisture",
        min_valid_pixels=2,
    )

    assert np.isnan(anomaly.data[0, 0])
    assert "insufficient_valid_pixels_field-a" in anomaly.missing_data_flags
    assert np.isfinite(anomaly.data[0, 2])


def test_constant_fields_are_flagged_after_mad_and_std_fallbacks() -> None:
    feature = _feature(
        "NDMI",
        [[0.5, 0.5, 0.5, 0.5]],
        direction="lower_values_indicate_moisture_stress",
    )
    field_ids = np.array([["field-a", "field-a", "field-a", "field-a"]], dtype=object)

    anomaly = field_normalized_anomaly(feature, field_ids, component="moisture")

    assert np.isnan(anomaly.data).all()
    assert anomaly.missing_data_flags == ("constant_field_field-a",)


def test_mad_falls_back_to_std_for_nonconstant_low_mad_fields() -> None:
    feature = _feature(
        "LST",
        [[300.0, 300.0, 300.0, 304.0]],
        direction="higher_values_indicate_heat_or_thermal_support",
    )
    field_ids = np.array([["field-a", "field-a", "field-a", "field-a"]], dtype=object)

    anomaly = field_normalized_anomaly(feature, field_ids, component="thermal")

    assert "mad_fallback_std_field-a" in anomaly.missing_data_flags
    assert anomaly.data[0, 3] > 1.0


def test_shape_mismatch_is_rejected() -> None:
    feature = _feature("NDMI", [[0.5, 0.4]], direction="lower_values_indicate_moisture_stress")
    field_ids = np.array([["field-a"], ["field-a"]], dtype=object)

    with pytest.raises(ValueError, match="matching shapes"):
        field_normalized_anomaly(feature, field_ids, component="moisture")


def test_composite_anomaly_map_combines_configured_weighted_components() -> None:
    moisture = _anomaly_layer("NDMI", "moisture", [[2.0, -1.0]])
    vegetation = _anomaly_layer("NDVI", "vegetation_stress", [[1.0, 3.0]])
    brightness = _anomaly_layer("BSI", "soil_brightness", [[0.5, 1.0]])
    thermal = _anomaly_layer("LST", "thermal", [[4.0, 0.0]])
    sar = _anomaly_layer("VV_VH_ratio", "sar", [[1.0, 2.0]])

    composite = composite_anomaly_map(
        (moisture, vegetation, brightness, thermal, sar),
        {
            "moisture": _ComponentConfig(inputs=["NDMI", "NDWI", "MSI"], weight=0.20),
            "vegetation_stress": _ComponentConfig(inputs=["NDVI"], weight=0.20),
            "soil_brightness": _ComponentConfig(inputs=["BSI", "brightness"], weight=0.20),
            "thermal": _ComponentConfig(inputs=["LST"], weight=0.20),
            "sar": _ComponentConfig(inputs=["VV", "VH", "VV_VH_ratio"], weight=0.20),
        },
    )

    assert composite.component_weights == {
        "moisture": 0.2,
        "vegetation_stress": 0.2,
        "soil_brightness": 0.2,
        "thermal": 0.2,
        "sar": 0.2,
    }
    assert np.allclose(composite.data, np.array([[1.7, 1.2]]))
    assert composite.source_feature_names == ("BSI", "LST", "NDMI", "NDVI", "VV_VH_ratio")


def test_composite_anomaly_map_reports_missing_inputs_without_crashing() -> None:
    composite = composite_anomaly_map(
        (_anomaly_layer("NDMI", "moisture", [[2.0]]),),
        {
            "moisture": _ComponentConfig(inputs=["NDMI"], weight=0.2),
            "thermal": _ComponentConfig(inputs=["LST"], weight=0.1),
        },
    )

    assert np.allclose(composite.data, np.array([[2.0]]))
    assert composite.component_weights == {"moisture": 1.0}
    assert composite.missing_data_flags == ("missing_component_thermal",)


def _feature(
    name: str,
    data: list[list[float]],
    *,
    direction: str,
    date: str | None = "2024-05-01",
) -> DynamicFeatureLayer:
    return DynamicFeatureLayer(
        name=name,
        data=np.array(data, dtype=float),
        sensor="test",
        date=date,
        source_layer_ids=(f"{name}_layer",),
        evidence_direction=direction,
    )


def _anomaly_layer(
    feature_name: str,
    component: str,
    data: list[list[float]],
) -> FieldAnomalyLayer:
    return FieldAnomalyLayer(
        name=f"{feature_name}_field_anomaly",
        component=component,
        data=np.array(data, dtype=float),
        date="2024-05-01",
        source_feature_names=(feature_name,),
        source_layer_ids=(f"{feature_name}_layer",),
        evidence_direction="higher_values_indicate_stronger_local_anomaly_support",
    )
