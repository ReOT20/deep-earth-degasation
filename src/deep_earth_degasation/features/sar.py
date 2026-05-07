from __future__ import annotations

import numpy as np

from deep_earth_degasation.features.types import DynamicFeatureLayer, DynamicFeatureResult
from deep_earth_degasation.io.raster_stack import RasterLayer, RasterStack

SAR_EVIDENCE_DIRECTIONS = {
    "VV": "backscatter_supporting_context",
    "VH": "backscatter_supporting_context",
    "VV_VH_ratio": "higher_or_lower_values_indicate_sar_anomaly",
    "temporal_difference": "larger_absolute_values_indicate_temporal_sar_change",
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


def vv_vh_ratio_features(
    by_feature: dict[str, list[RasterLayer]],
) -> tuple[list[DynamicFeatureLayer], list[str]]:
    vv_by_date = _layers_by_date(by_feature.get("VV", []))
    vh_by_date = _layers_by_date(by_feature.get("VH", []))
    dates = sorted(set(vv_by_date) | set(vh_by_date), key=lambda date: "" if date is None else date)
    features: list[DynamicFeatureLayer] = []
    flags: list[str] = []

    for date in dates:
        vv = vv_by_date.get(date)
        vh = vh_by_date.get(date)
        date_label = "undated" if date is None else date
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
                date=date,
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
    layer = _first_layer(by_feature.get(feature_name))
    if layer is None:
        flags.append(f"missing_sentinel1_{feature_name}")
        return
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


def _first_layer(layers: list[RasterLayer] | None) -> RasterLayer | None:
    if not layers:
        return None
    return sorted(layers, key=lambda layer: layer.spec.date or "")[0]
