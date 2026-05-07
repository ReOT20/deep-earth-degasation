from __future__ import annotations

import numpy as np
import pytest

from deep_earth_degasation.features.types import DynamicFeatureLayer
from deep_earth_degasation.features.weather import (
    POST_RAIN_DRYING_DIRECTION,
    RAIN_EVENT_DIRECTION,
    RAINFALL_DIRECTION,
    WeatherFeatureError,
    days_since_last_rain,
    load_weather_context,
    load_weather_context_manifest,
    post_rain_drying_delta,
    post_rain_drying_features,
    weather_features_from_context,
)


def test_load_weather_context_from_prepared_records() -> None:
    context = load_weather_context(
        [
            {"date": "2024-05-12", "rainfall_mm": 0.5, "source_id": "manual_2"},
            {"date": "2024-05-10", "rainfall_mm": 12.0, "source_id": "manual_1"},
        ]
    )

    assert [event.date for event in context.events] == ["2024-05-10", "2024-05-12"]
    assert context.events[0].rainfall_mm == 12.0
    assert context.events[0].context_id == "manual_1"


def test_load_weather_context_from_small_yaml_manifest(tmp_path) -> None:
    manifest_path = tmp_path / "weather.yaml"
    manifest_path.write_text(
        """
events:
  - date: "2024-05-10"
    precipitation_mm: 12.0
    source_id: manual_1
    source: prepared
  - date: "2024-05-12"
    rainfall_mm: 0.5
""",
        encoding="utf-8",
    )

    context = load_weather_context_manifest(manifest_path)

    assert [event.date for event in context.events] == ["2024-05-10", "2024-05-12"]
    assert [event.rainfall_mm for event in context.events] == [12.0, 0.5]
    assert context.events[0].source == "prepared"


def test_load_weather_context_manifest_rejects_unknown_keys(tmp_path) -> None:
    manifest_path = tmp_path / "weather.yaml"
    manifest_path.write_text(
        """
events:
  - date: "2024-05-10"
    rainfall_mm: 12.0
    confidence: high
""",
        encoding="utf-8",
    )

    with pytest.raises(WeatherFeatureError, match="unknown keys"):
        load_weather_context_manifest(manifest_path)


def test_load_weather_context_manifest_rejects_invalid_rainfall_values(tmp_path) -> None:
    manifest_path = tmp_path / "weather.yaml"
    manifest_path.write_text(
        """
events:
  - date: "2024-05-10"
    rainfall_mm: trace
""",
        encoding="utf-8",
    )

    with pytest.raises(WeatherFeatureError, match="must be a number"):
        load_weather_context_manifest(manifest_path)


def test_weather_context_missing_inputs_return_flags() -> None:
    result = weather_features_from_context(None)

    assert result.features == ()
    assert result.missing_data_flags == ("missing_weather_context",)


def test_weather_context_exports_scalar_supporting_layers_and_precipitation_flags() -> None:
    context = load_weather_context(
        [
            {"date": "2024-05-10", "rainfall_mm": 12.0, "source_id": "manual_1"},
            {"date": "2024-05-12"},
        ]
    )

    result = weather_features_from_context(context)

    assert [feature.name for feature in result.features] == [
        "rain_event",
        "rainfall_mm",
        "rain_event",
    ]
    assert result.features[0].evidence_direction == RAIN_EVENT_DIRECTION
    assert result.features[1].evidence_direction == RAINFALL_DIRECTION
    assert np.allclose(result.features[1].data, np.array([[12.0]]))
    assert "missing_rainfall_mm_2024-05-12" in result.missing_data_flags
    assert all("proof" not in feature.evidence_direction for feature in result.features)


def test_days_since_last_rain_uses_threshold() -> None:
    context = load_weather_context(
        [
            {"date": "2024-05-10", "rainfall_mm": 0.5},
            {"date": "2024-05-12", "rainfall_mm": 8.0},
        ]
    )

    assert days_since_last_rain("2024-05-15", context, rainfall_threshold_mm=1.0) == 3


def test_post_rain_drying_delta_uses_weather_and_moisture_features() -> None:
    context = load_weather_context(
        [{"date": "2024-05-10", "rainfall_mm": 15.0, "source_id": "rain_1"}]
    )
    before = _moisture("before", "NDMI", "2024-05-11", [[0.8, 0.6]])
    after = _moisture("after", "NDMI", "2024-05-15", [[0.3, 0.4]])

    result = post_rain_drying_delta(before, after, context=context)
    feature = result.features[0]

    assert np.allclose(feature.data, np.array([[0.5, 0.2]]))
    assert feature.name == "post_rain_drying"
    assert feature.date == "2024-05-15"
    assert feature.source_layer_ids == ("rain_1", "before", "after")
    assert feature.evidence_direction == POST_RAIN_DRYING_DIRECTION
    assert "not_standalone" in feature.evidence_direction
    assert result.missing_data_flags == ()


def test_post_rain_drying_delta_uses_msi_direction() -> None:
    context = load_weather_context([{"date": "2024-05-10", "rainfall_mm": 15.0}])
    before = _moisture("before", "MSI", "2024-05-11", [[0.8]])
    after = _moisture("after", "MSI", "2024-05-15", [[1.1]])

    result = post_rain_drying_delta(before, after, context=context)

    assert np.allclose(result.features[0].data, np.array([[0.3]]))


def test_post_rain_drying_missing_weather_does_not_crash() -> None:
    result = post_rain_drying_delta(
        _moisture("before", "NDMI", "2024-05-11", [[0.8]]),
        _moisture("after", "NDMI", "2024-05-15", [[0.3]]),
        context=None,
    )

    assert result.features == ()
    assert result.missing_data_flags == ("missing_weather_context",)


def test_post_rain_drying_requires_matching_shapes() -> None:
    context = load_weather_context([{"date": "2024-05-10", "rainfall_mm": 15.0}])
    result = post_rain_drying_delta(
        _moisture("before", "NDMI", "2024-05-11", [[0.8, 0.7]]),
        _moisture("after", "NDMI", "2024-05-15", [[0.3], [0.2]]),
        context=context,
    )

    assert result.features == ()
    assert result.missing_data_flags == ("post_rain_drying_shape_mismatch",)


def test_post_rain_drying_flags_missing_prior_rain_event() -> None:
    context = load_weather_context([{"date": "2024-05-20", "rainfall_mm": 15.0}])
    result = post_rain_drying_delta(
        _moisture("before", "NDMI", "2024-05-11", [[0.8]]),
        _moisture("after", "NDMI", "2024-05-15", [[0.3]]),
        context=context,
    )

    assert result.features == ()
    assert result.missing_data_flags == ("missing_prior_rain_event",)


def test_post_rain_drying_features_pair_moisture_observations_around_rain() -> None:
    context = load_weather_context(
        [{"date": "2024-05-10", "rainfall_mm": 15.0, "source_id": "rain_1"}]
    )
    result = post_rain_drying_features(
        (
            _moisture("ndmi_before", "NDMI", "2024-05-01", [[0.8]]),
            _moisture("ndmi_after", "NDMI", "2024-05-15", [[0.5]]),
            _moisture("msi_before", "MSI", "2024-05-01", [[0.7]]),
            _moisture("msi_after", "MSI", "2024-05-15", [[1.0]]),
        ),
        context,
    )

    assert [feature.name for feature in result.features] == [
        "post_rain_drying_NDMI",
        "post_rain_drying_MSI",
    ]
    assert np.allclose(result.features[0].data, np.array([[0.3]]))
    assert np.allclose(result.features[1].data, np.array([[0.3]]))
    assert result.features[0].source_layer_ids == ("rain_1", "ndmi_before", "ndmi_after")
    assert result.missing_data_flags == ()


def test_post_rain_drying_features_missing_inputs_return_flags() -> None:
    context = load_weather_context([{"date": "2024-05-10", "rainfall_mm": 15.0}])

    assert post_rain_drying_features((), context).missing_data_flags == (
        "missing_moisture_features_for_post_rain_drying",
    )
    assert post_rain_drying_features(
        (_moisture("late", "NDMI", "2024-05-15", [[0.3]]),), context
    ).missing_data_flags == ("missing_moisture_pair_for_post_rain_drying_2024-05-10",)


def test_post_rain_drying_features_flags_shape_mismatch() -> None:
    context = load_weather_context([{"date": "2024-05-10", "rainfall_mm": 15.0}])
    result = post_rain_drying_features(
        (
            _moisture("before", "NDMI", "2024-05-01", [[0.8, 0.7]]),
            _moisture("after", "NDMI", "2024-05-15", [[0.5], [0.4]]),
        ),
        context,
    )

    assert result.features == ()
    assert result.missing_data_flags == ("post_rain_drying_shape_mismatch_2024-05-10_NDMI",)


def _moisture(
    layer_id: str,
    feature_name: str,
    feature_date: str,
    data: list[list[float]],
) -> DynamicFeatureLayer:
    return DynamicFeatureLayer(
        name=feature_name,
        data=np.array(data, dtype=float),
        sensor="sentinel2",
        date=feature_date,
        source_layer_ids=(layer_id,),
        evidence_direction="lower_values_indicate_moisture_stress",
    )
