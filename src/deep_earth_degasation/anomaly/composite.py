from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from deep_earth_degasation.anomaly.field_normalization import FieldAnomalyLayer
from deep_earth_degasation.io.raster_stack import FloatArray


class ComponentConfig(Protocol):
    @property
    def inputs(self) -> Sequence[str]: ...

    @property
    def weight(self) -> float: ...


@dataclass(frozen=True)
class CompositeAnomalyMap:
    data: FloatArray
    component_maps: dict[str, FloatArray]
    component_weights: dict[str, float]
    source_feature_names: tuple[str, ...]
    source_layer_ids: tuple[str, ...]
    missing_data_flags: tuple[str, ...] = ()


def composite_anomaly_map(
    layers: tuple[FieldAnomalyLayer, ...],
    component_config: Mapping[str, ComponentConfig],
) -> CompositeAnomalyMap:
    """Combine field-normalized anomaly layers into a weighted support map."""
    if not layers:
        return CompositeAnomalyMap(
            data=np.empty((0, 0), dtype=np.float64),
            component_maps={},
            component_weights={},
            source_feature_names=(),
            source_layer_ids=(),
            missing_data_flags=("missing_anomaly_layers",),
        )

    reference_shape = layers[0].data.shape
    _require_matching_shapes(layers, reference_shape)
    component_maps: dict[str, FloatArray] = {}
    raw_weights: dict[str, float] = {}
    flags: list[str] = []

    for component_name, config in component_config.items():
        matching_layers = [
            layer
            for layer in layers
            if layer.component == component_name
            and any(feature_name in config.inputs for feature_name in layer.source_feature_names)
        ]
        if not matching_layers:
            flags.append(f"missing_component_{component_name}")
            continue

        component_maps[component_name] = _nanmean_stack(
            tuple(_positive_support(layer.data) for layer in matching_layers)
        )
        if config.weight > 0:
            raw_weights[component_name] = float(config.weight)

    if not component_maps or not raw_weights:
        return CompositeAnomalyMap(
            data=np.full(reference_shape, np.nan, dtype=np.float64),
            component_maps=component_maps,
            component_weights={},
            source_feature_names=_source_feature_names(layers),
            source_layer_ids=_source_layer_ids(layers),
            missing_data_flags=(*tuple(flags), "missing_weighted_components"),
        )

    total_weight = sum(raw_weights.values())
    weights = {name: weight / total_weight for name, weight in raw_weights.items()}
    output = np.zeros(reference_shape, dtype=np.float64)
    weight_sum = np.zeros(reference_shape, dtype=np.float64)
    for component_name, weight in weights.items():
        component = component_maps[component_name]
        valid_mask = np.isfinite(component)
        output[valid_mask] += component[valid_mask] * weight
        weight_sum[valid_mask] += weight

    output[weight_sum == 0] = np.nan
    valid_output = weight_sum > 0
    output[valid_output] = output[valid_output] / weight_sum[valid_output]

    return CompositeAnomalyMap(
        data=output,
        component_maps=component_maps,
        component_weights=weights,
        source_feature_names=_source_feature_names(layers),
        source_layer_ids=_source_layer_ids(layers),
        missing_data_flags=tuple(flags),
    )


def _positive_support(data: FloatArray) -> FloatArray:
    return np.clip(data, 0.0, None)


def _nanmean_stack(arrays: tuple[FloatArray, ...]) -> FloatArray:
    stack = np.stack(arrays, axis=0)
    valid_count = np.count_nonzero(np.isfinite(stack), axis=0)
    total = np.nansum(stack, axis=0)
    output = np.full(arrays[0].shape, np.nan, dtype=np.float64)
    np.divide(total, valid_count, out=output, where=valid_count > 0)
    return output


def _require_matching_shapes(
    layers: tuple[FieldAnomalyLayer, ...],
    reference_shape: tuple[int, ...],
) -> None:
    for layer in layers:
        if layer.data.shape != reference_shape:
            raise ValueError("All anomaly layers must have matching shapes.")


def _source_feature_names(layers: tuple[FieldAnomalyLayer, ...]) -> tuple[str, ...]:
    names = {name for layer in layers for name in layer.source_feature_names}
    return tuple(sorted(names))


def _source_layer_ids(layers: tuple[FieldAnomalyLayer, ...]) -> tuple[str, ...]:
    layer_ids = {layer_id for layer in layers for layer_id in layer.source_layer_ids}
    return tuple(sorted(layer_ids))
