from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from deep_earth_degasation.anomaly.field_normalization import FieldAnomalyLayer


@dataclass(frozen=True)
class ObjectZonalStats:
    mean_anomaly: float
    max_anomaly: float
    per_feature_mean: dict[str, float]
    per_feature_max: dict[str, float]
    source_feature_names: tuple[str, ...]
    source_layer_ids: tuple[str, ...]
    anomalous_dates: tuple[str, ...]
    supporting_observation_count: int


def compute_object_zonal_stats(
    layers: tuple[FieldAnomalyLayer, ...],
    mask: np.ndarray,
) -> ObjectZonalStats:
    """Compute deterministic NaN-aware object stats across source anomaly layers."""
    per_feature_mean: dict[str, float] = {}
    per_feature_max: dict[str, float] = {}
    per_feature_values: dict[str, list[np.ndarray]] = {}
    source_feature_names: set[str] = set()
    source_layer_ids: set[str] = set()
    dates: set[str] = set()
    object_values: list[np.ndarray] = []
    supporting_observation_count = 0

    for layer in layers:
        if layer.data.shape != mask.shape:
            raise ValueError("Object mask and anomaly layers must have matching shapes.")
        values = layer.data[mask]
        valid_values = values[np.isfinite(values)]
        positive_values = valid_values[valid_values > 0]
        for feature_name in layer.source_feature_names:
            source_feature_names.add(feature_name)
            if valid_values.size:
                per_feature_values.setdefault(feature_name, []).append(valid_values)
        source_layer_ids.update(layer.source_layer_ids)
        if positive_values.size:
            supporting_observation_count += 1
        if layer.date is not None and positive_values.size:
            dates.add(layer.date)
        if valid_values.size:
            object_values.append(valid_values)

    for feature_name, value_chunks in per_feature_values.items():
        merged_feature_values = np.concatenate(value_chunks)
        per_feature_mean[feature_name] = _nan_stat(merged_feature_values, stat="mean")
        per_feature_max[feature_name] = _nan_stat(merged_feature_values, stat="max")

    merged_values = np.concatenate(object_values) if object_values else np.array([], dtype=float)
    return ObjectZonalStats(
        mean_anomaly=_nan_stat(merged_values, stat="mean"),
        max_anomaly=_nan_stat(merged_values, stat="max"),
        per_feature_mean=per_feature_mean,
        per_feature_max=per_feature_max,
        source_feature_names=tuple(sorted(source_feature_names)),
        source_layer_ids=tuple(sorted(source_layer_ids)),
        anomalous_dates=tuple(sorted(dates)),
        supporting_observation_count=supporting_observation_count,
    )


def _nan_stat(values: np.ndarray, *, stat: str) -> float:
    if values.size == 0:
        return float("nan")
    if stat == "max":
        return float(np.nanmax(values))
    return float(np.nanmean(values))
