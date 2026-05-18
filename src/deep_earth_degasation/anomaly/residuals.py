from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import geopandas as gpd
import numpy as np

from deep_earth_degasation.anomaly.field_normalization import FieldAnomalyLayer
from deep_earth_degasation.features.types import DynamicFeatureLayer
from deep_earth_degasation.io.raster_stack import FloatArray

_MAD_SCALE = 1.4826
PEER_CONTEXT_COLUMNS = ("crop_type", "phenology_stage", "weather_stratum")


@dataclass(frozen=True)
class ResidualLayerResult:
    layers: tuple[FieldAnomalyLayer, ...]
    missing_data_flags: tuple[str, ...] = ()


def peer_residual_layers(
    features: Sequence[DynamicFeatureLayer],
    field_ids: np.ndarray,
    fields: gpd.GeoDataFrame,
    *,
    component_for_feature: Callable[[DynamicFeatureLayer], str | None],
    min_valid_pixels: int = 1,
    mad_epsilon: float = 1.0e-6,
) -> ResidualLayerResult:
    """Build peer-stratum residuals when field context columns are available."""
    context_columns = tuple(column for column in PEER_CONTEXT_COLUMNS if column in fields.columns)
    if not context_columns:
        return ResidualLayerResult((), ("missing_peer_residual_context",))

    strata_by_field = _strata_by_field(fields, context_columns)
    if not strata_by_field:
        return ResidualLayerResult((), ("missing_peer_residual_context",))

    layers: list[FieldAnomalyLayer] = []
    flags: list[str] = []
    for feature in features:
        component = component_for_feature(feature)
        if component is None:
            continue
        residual = _peer_residual_layer(
            feature,
            field_ids,
            strata_by_field,
            component=component,
            min_valid_pixels=min_valid_pixels,
            mad_epsilon=mad_epsilon,
        )
        if residual is None:
            flags.append(f"missing_peer_residual_{feature.name}_{feature.date or 'undated'}")
            continue
        layers.append(residual)
        flags.extend(residual.missing_data_flags)

    if not layers:
        flags.append("missing_peer_residual_context")
    return ResidualLayerResult(tuple(layers), tuple(sorted(set(flags))))


def temporal_residual_layers(
    features: Sequence[DynamicFeatureLayer],
    *,
    component_for_feature: Callable[[DynamicFeatureLayer], str | None],
    min_valid_observations: int = 2,
    mad_epsilon: float = 1.0e-6,
) -> ResidualLayerResult:
    """Build per-feature temporal residual layers from repeated prepared dates."""
    groups: dict[str, list[DynamicFeatureLayer]] = {}
    for feature in features:
        if feature.date is None:
            continue
        if component_for_feature(feature) is None:
            continue
        groups.setdefault(feature.name, []).append(feature)

    layers: list[FieldAnomalyLayer] = []
    flags: list[str] = []
    for feature_name, group in sorted(groups.items()):
        dated_group = tuple(sorted(group, key=lambda feature: feature.date or ""))
        distinct_dates = {feature.date for feature in dated_group if feature.date is not None}
        if len(distinct_dates) < min_valid_observations:
            flags.append(f"missing_temporal_residual_{feature_name}")
            continue
        group_layers = _finite_layers(
            _temporal_residual_group(
                dated_group,
                component=component_for_feature(dated_group[0]) or "",
                mad_epsilon=mad_epsilon,
            )
        )
        if not group_layers:
            flags.append(f"missing_temporal_residual_{feature_name}")
            continue
        layers.extend(group_layers)

    if not layers:
        flags.append("missing_temporal_residual_context")
    return ResidualLayerResult(tuple(layers), tuple(sorted(set(flags))))


def residual_layers_metadata(
    layers: Sequence[FieldAnomalyLayer],
    *,
    missing_data_flags: Sequence[str] = (),
) -> dict[str, object]:
    return {
        "schema_version": "hierarchical_residuals.v1",
        "missing_data_flags": sorted(set(missing_data_flags)),
        "layers": [
            {
                "name": layer.name,
                "residual_type": layer.residual_type,
                "component": layer.component,
                "date": layer.date,
                "source_feature_names": list(layer.source_feature_names),
                "source_layer_ids": list(layer.source_layer_ids),
                "missing_data_flags": list(layer.missing_data_flags),
            }
            for layer in layers
        ],
    }


def _peer_residual_layer(
    feature: DynamicFeatureLayer,
    field_ids: np.ndarray,
    strata_by_field: dict[int, tuple[str, ...]],
    *,
    component: str,
    min_valid_pixels: int,
    mad_epsilon: float,
) -> FieldAnomalyLayer | None:
    output = np.full(feature.data.shape, np.nan, dtype=np.float64)
    flags: list[str] = []
    field_labels = sorted(set(_valid_field_labels(field_ids)) & set(strata_by_field))
    if not field_labels:
        return None

    labels_by_stratum: dict[tuple[str, ...], list[int]] = {}
    for field_label in field_labels:
        labels_by_stratum.setdefault(strata_by_field[field_label], []).append(field_label)

    for stratum, labels in labels_by_stratum.items():
        if len(labels) < 2:
            flags.append(f"insufficient_peer_fields_{_stratum_label(stratum)}")
            continue
        mask = np.isin(field_ids, labels)
        valid_mask = mask & np.isfinite(feature.data)
        if int(np.count_nonzero(valid_mask)) < min_valid_pixels:
            flags.append(f"insufficient_peer_valid_pixels_{_stratum_label(stratum)}")
            continue
        residual_values = _robust_residual(
            feature.data[valid_mask],
            feature.evidence_direction,
            mad_epsilon=mad_epsilon,
        )
        if residual_values is None:
            flags.append(f"constant_peer_stratum_{_stratum_label(stratum)}")
            continue
        output[valid_mask] = residual_values

    if np.isnan(output).all():
        return None
    return FieldAnomalyLayer(
        name=f"{feature.name}_peer_residual",
        component=component,
        data=output,
        date=feature.date,
        source_feature_names=(feature.name,),
        source_layer_ids=feature.source_layer_ids,
        evidence_direction="higher_values_indicate_stronger_peer_anomaly_support",
        residual_type="peer",
        missing_data_flags=tuple(sorted(set(flags))),
    )


def _temporal_residual_group(
    group: Sequence[DynamicFeatureLayer],
    *,
    component: str,
    mad_epsilon: float,
) -> tuple[FieldAnomalyLayer, ...]:
    stack = np.stack([feature.data for feature in group], axis=0)
    valid_count = np.count_nonzero(np.isfinite(stack), axis=0)
    if not np.any(valid_count >= 2):
        return ()
    median = np.nanmedian(stack, axis=0)
    mad = np.nanmedian(np.abs(stack - median), axis=0)
    scale = _MAD_SCALE * mad
    std = np.nanstd(stack, axis=0)
    scale = np.where(scale > mad_epsilon, scale, std)
    valid_scale = (scale > mad_epsilon) & (valid_count >= 2)

    layers: list[FieldAnomalyLayer] = []
    for feature in group:
        residual = np.full(feature.data.shape, np.nan, dtype=np.float64)
        valid_mask = valid_scale & np.isfinite(feature.data)
        raw = (feature.data[valid_mask] - median[valid_mask]) / scale[valid_mask]
        residual[valid_mask] = _orient_zscore(raw, feature.evidence_direction)
        layers.append(
            FieldAnomalyLayer(
                name=f"{feature.name}_temporal_residual",
                component=component,
                data=residual,
                date=feature.date,
                source_feature_names=(feature.name,),
                source_layer_ids=feature.source_layer_ids,
                evidence_direction="higher_values_indicate_stronger_temporal_anomaly_support",
                residual_type="temporal",
            )
        )
    return tuple(layers)


def _finite_layers(layers: Sequence[FieldAnomalyLayer]) -> tuple[FieldAnomalyLayer, ...]:
    return tuple(layer for layer in layers if np.isfinite(layer.data).any())


def _strata_by_field(
    fields: gpd.GeoDataFrame,
    context_columns: Sequence[str],
) -> dict[int, tuple[str, ...]]:
    strata: dict[int, tuple[str, ...]] = {}
    for label, (_, row) in enumerate(fields.iterrows(), start=1):
        values = tuple(_text(row[column]) for column in context_columns)
        if all(values):
            strata[label] = values
    return strata


def _valid_field_labels(field_ids: np.ndarray) -> tuple[int, ...]:
    labels = []
    for value in np.unique(field_ids):
        if isinstance(value, np.integer):
            value = int(value)
        if isinstance(value, int) and value > 0:
            labels.append(value)
    return tuple(labels)


def _robust_residual(
    values: FloatArray,
    evidence_direction: str,
    *,
    mad_epsilon: float,
) -> FloatArray | None:
    median = float(np.nanmedian(values))
    mad = float(np.nanmedian(np.abs(values - median)))
    scale = _MAD_SCALE * mad
    if scale <= mad_epsilon:
        std = float(np.nanstd(values))
        if std <= mad_epsilon:
            return None
        scale = std
    return _orient_zscore((values - median) / scale, evidence_direction)


def _orient_zscore(zscore: FloatArray, evidence_direction: str) -> FloatArray:
    direction = evidence_direction.lower()
    if "higher_or_lower" in direction or "larger_absolute" in direction:
        return np.abs(zscore)
    if "lower_values" in direction:
        return -zscore
    return zscore


def _stratum_label(stratum: Sequence[str]) -> str:
    return "_".join(value.replace(" ", "_") for value in stratum)


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return str(value).strip()
