from __future__ import annotations

from datetime import date

import numpy as np

from deep_earth_degasation.features.types import DynamicFeatureLayer, DynamicFeatureResult
from deep_earth_degasation.features.weather import WeatherContext
from deep_earth_degasation.io.raster_stack import RasterLayer, RasterStack

SAR_EVIDENCE_DIRECTIONS = {
    "VV": "backscatter_supporting_context",
    "VH": "backscatter_supporting_context",
    "VV_VH_ratio": "higher_or_lower_values_indicate_sar_anomaly",
    "temporal_difference": "larger_absolute_values_indicate_temporal_sar_change",
    "sar_event_response": "larger_absolute_values_indicate_supporting_sar_event_response_not_moisture_proof",
}


def sentinel1_features_from_stack(stack: RasterStack) -> DynamicFeatureResult:
    """Compute available Sentinel-1 SAR feature layers from a prepared stack."""
    layers = [layer for layer in stack.layers if layer.spec.sensor.lower() == "sentinel1"]
    flags: list[str] = []
    features: list[DynamicFeatureLayer] = []
    by_feature = _layers_by_feature(layers)

    _append_prepared_sar(features, by_feature, "VV", flags)
    _append_prepared_sar(features, by_feature, "VH", flags)

    ratio_features, ratio_flags = vv_vh_ratio_features(by_feature)
    features.extend(ratio_features)
    flags.extend(ratio_flags)
    if not ratio_features and not ratio_flags:
        flags.append("missing_sentinel1_VV_VH_ratio_inputs")

    temporal = temporal_difference(by_feature)
    if temporal is None:
        flags.append("missing_sentinel1_temporal_difference_inputs")
    else:
        features.append(temporal)

    return DynamicFeatureResult(features=tuple(features), missing_data_flags=tuple(flags))


def sentinel1_event_response_features(
    sar_features: tuple[DynamicFeatureLayer, ...],
    context: WeatherContext | None,
    *,
    rainfall_threshold_mm: float = 1.0,
) -> DynamicFeatureResult:
    """Compute supporting SAR changes around prepared weather events."""
    if context is None or not context.events:
        return DynamicFeatureResult(features=(), missing_data_flags=("missing_weather_context",))
    event_features = tuple(
        feature
        for feature in sar_features
        if feature.sensor.lower() == "sentinel1"
        and feature.name in {"VV", "VH", "VV_VH_ratio"}
        and feature.date is not None
    )
    if not event_features:
        return DynamicFeatureResult(
            features=(), missing_data_flags=("missing_sentinel1_event_response_inputs",)
        )

    rain_events = [
        event
        for event in context.events
        if event.rainfall_mm is not None and event.rainfall_mm >= rainfall_threshold_mm
    ]
    if not rain_events:
        return DynamicFeatureResult(
            features=(), missing_data_flags=("missing_prior_rain_event_for_sar_event_response",)
        )

    features: list[DynamicFeatureLayer] = []
    flags: list[str] = []
    by_name = _features_by_name(event_features)
    for event in rain_events:
        event_had_pair = False
        for feature_name, dated_features in by_name.items():
            before, after = _bracketing_features(dated_features, event.date)
            if before is None or after is None:
                continue
            event_had_pair = True
            if before.data.shape != after.data.shape:
                flags.append(f"sar_event_response_shape_mismatch_{event.date}_{feature_name}")
                continue
            features.append(
                DynamicFeatureLayer(
                    name=f"sar_event_response_{feature_name}",
                    data=np.abs(after.data - before.data).astype(np.float64),
                    sensor="sentinel1",
                    date=after.date,
                    source_layer_ids=(
                        event.context_id,
                        *before.source_layer_ids,
                        *after.source_layer_ids,
                    ),
                    evidence_direction=SAR_EVIDENCE_DIRECTIONS["sar_event_response"],
                )
            )
        if not event_had_pair:
            flags.append(f"missing_sar_pair_for_event_response_{event.date}")

    return DynamicFeatureResult(features=tuple(features), missing_data_flags=tuple(flags))


def vv_vh_ratio_features(
    by_feature: dict[str, list[RasterLayer]],
) -> tuple[list[DynamicFeatureLayer], list[str]]:
    vv_by_date = _layers_by_date(by_feature.get("VV", []))
    vh_by_date = _layers_by_date(by_feature.get("VH", []))
    dates = sorted(set(vv_by_date) | set(vh_by_date), key=lambda date: "" if date is None else date)
    features: list[DynamicFeatureLayer] = []
    flags: list[str] = []

    for date_key in dates:
        vv = vv_by_date.get(date_key)
        vh = vh_by_date.get(date_key)
        date_label = "undated" if date_key is None else date_key
        if vv is None:
            flags.append(f"missing_sentinel1_VV_for_ratio_{date_label}")
            continue
        if vh is None:
            flags.append(f"missing_sentinel1_VH_for_ratio_{date_label}")
            continue
        if vv.data.shape != vh.data.shape:
            flags.append(f"VV_VH_ratio_shape_mismatch_{date_label}")
            continue
        features.append(
            DynamicFeatureLayer(
                name="VV_VH_ratio",
                data=vv.data / (vh.data + 1e-9),
                sensor="sentinel1",
                date=date_key,
                source_layer_ids=(vv.spec.id, vh.spec.id),
                evidence_direction=SAR_EVIDENCE_DIRECTIONS["VV_VH_ratio"],
            )
        )

    return features, flags


def temporal_difference(by_feature: dict[str, list[RasterLayer]]) -> DynamicFeatureLayer | None:
    for feature_name in ("VV", "VH"):
        layers = sorted(by_feature.get(feature_name, []), key=lambda layer: layer.spec.date or "")
        if len(layers) < 2:
            continue
        first, second = layers[0], layers[-1]
        if first.data.shape != second.data.shape:
            return DynamicFeatureLayer(
                name="temporal_difference",
                data=np.full(first.data.shape, np.nan),
                sensor="sentinel1",
                date=second.spec.date,
                source_layer_ids=(first.spec.id, second.spec.id),
                evidence_direction=SAR_EVIDENCE_DIRECTIONS["temporal_difference"],
                missing_data_flags=("temporal_difference_shape_mismatch",),
            )
        return DynamicFeatureLayer(
            name="temporal_difference",
            data=second.data - first.data,
            sensor="sentinel1",
            date=second.spec.date,
            source_layer_ids=(first.spec.id, second.spec.id),
            evidence_direction=SAR_EVIDENCE_DIRECTIONS["temporal_difference"],
        )
    return None


def _append_prepared_sar(
    features: list[DynamicFeatureLayer],
    by_feature: dict[str, list[RasterLayer]],
    feature_name: str,
    flags: list[str],
) -> None:
    layers = by_feature.get(feature_name, [])
    if not layers:
        flags.append(f"missing_sentinel1_{feature_name}")
        return
    for layer in sorted(layers, key=lambda item: (item.spec.date or "", item.spec.id)):
        features.append(
            DynamicFeatureLayer(
                name=feature_name,
                data=layer.data,
                sensor="sentinel1",
                date=layer.spec.date,
                source_layer_ids=(layer.spec.id,),
                evidence_direction=SAR_EVIDENCE_DIRECTIONS[feature_name],
            )
        )


def _layers_by_feature(layers: list[RasterLayer]) -> dict[str, list[RasterLayer]]:
    by_feature: dict[str, list[RasterLayer]] = {}
    for layer in layers:
        by_feature.setdefault(layer.spec.feature_name, []).append(layer)
    return by_feature


def _layers_by_date(layers: list[RasterLayer]) -> dict[str | None, RasterLayer]:
    by_date: dict[str | None, RasterLayer] = {}
    for layer in sorted(layers, key=lambda item: (item.spec.date or "", item.spec.id)):
        by_date.setdefault(layer.spec.date, layer)
    return by_date


def _features_by_name(
    features: tuple[DynamicFeatureLayer, ...],
) -> dict[str, list[DynamicFeatureLayer]]:
    by_name: dict[str, list[DynamicFeatureLayer]] = {}
    for feature in features:
        by_name.setdefault(feature.name, []).append(feature)
    return by_name


def _bracketing_features(
    features: list[DynamicFeatureLayer],
    event_date: str,
) -> tuple[DynamicFeatureLayer | None, DynamicFeatureLayer | None]:
    event = date.fromisoformat(event_date)
    dated = sorted(
        (feature for feature in features if feature.date is not None),
        key=lambda feature: feature.date or "",
    )
    before = next(
        (feature for feature in reversed(dated) if date.fromisoformat(feature.date or "") <= event),
        None,
    )
    after = next(
        (feature for feature in dated if date.fromisoformat(feature.date or "") > event),
        None,
    )
    return before, after
