from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from deep_earth_degasation.features.types import DynamicFeatureLayer
from deep_earth_degasation.io.raster_stack import FloatArray

_MAD_SCALE = 1.4826
FieldId = object


@dataclass(frozen=True)
class FieldAnomalyLayer:
    name: str
    component: str
    data: FloatArray
    date: str | None
    source_feature_names: tuple[str, ...]
    source_layer_ids: tuple[str, ...]
    evidence_direction: str
    residual_type: str = "field"
    missing_data_flags: tuple[str, ...] = ()


def field_normalized_anomaly(
    feature: DynamicFeatureLayer,
    field_ids: np.ndarray,
    *,
    component: str,
    min_valid_pixels: int = 1,
    mad_epsilon: float = 1.0e-6,
) -> FieldAnomalyLayer:
    """Normalize a dynamic feature inside each field with robust z-scores."""
    _require_matching_shape(feature.data, field_ids, feature_name=feature.name)
    output = np.full(feature.data.shape, np.nan, dtype=np.float64)
    flags: list[str] = []

    for field_id in _unique_field_ids(field_ids):
        field_mask = field_ids == field_id
        valid_mask = field_mask & np.isfinite(feature.data)
        valid_count = int(np.count_nonzero(valid_mask))
        if valid_count < min_valid_pixels:
            flags.append(f"insufficient_valid_pixels_{_field_id_label(field_id)}")
            continue

        values = feature.data[valid_mask]
        median = float(np.nanmedian(values))
        mad = float(np.nanmedian(np.abs(values - median)))
        scale = _MAD_SCALE * mad
        if scale <= mad_epsilon:
            std = float(np.nanstd(values))
            if std <= mad_epsilon:
                flags.append(f"constant_field_{_field_id_label(field_id)}")
                continue
            scale = std
            flags.append(f"mad_fallback_std_{_field_id_label(field_id)}")

        zscore = (values - median) / scale
        output[valid_mask] = _orient_zscore(zscore, feature.evidence_direction)

    return FieldAnomalyLayer(
        name=f"{feature.name}_field_anomaly",
        component=component,
        data=output,
        date=feature.date,
        source_feature_names=(feature.name,),
        source_layer_ids=feature.source_layer_ids,
        evidence_direction="higher_values_indicate_stronger_local_anomaly_support",
        residual_type="field",
        missing_data_flags=tuple(flags),
    )


def _orient_zscore(zscore: FloatArray, evidence_direction: str) -> FloatArray:
    direction = evidence_direction.lower()
    if "higher_or_lower" in direction or "larger_absolute" in direction:
        return np.abs(zscore)
    if "lower_values" in direction:
        return -zscore
    return zscore


def _unique_field_ids(field_ids: np.ndarray) -> tuple[FieldId, ...]:
    ids: dict[str, FieldId] = {}
    for raw_id in field_ids.ravel():
        if _is_missing_field_id(raw_id):
            continue
        ids.setdefault(
            _field_id_sort_key(raw_id), raw_id.item() if hasattr(raw_id, "item") else raw_id
        )
    return tuple(ids[key] for key in sorted(ids))


def _is_missing_field_id(field_id: FieldId) -> bool:
    if field_id is None:
        return True
    if isinstance(field_id, str):
        return field_id == ""
    if isinstance(field_id, int):
        return field_id == 0
    if isinstance(field_id, np.integer):
        return bool(field_id == 0)
    if isinstance(field_id, float | np.floating):
        return bool(np.isnan(field_id) or field_id == 0.0)
    return False


def _field_id_sort_key(field_id: FieldId) -> str:
    return f"{type(field_id).__name__}:{field_id}"


def _field_id_label(field_id: FieldId) -> str:
    return str(field_id)


def _require_matching_shape(
    feature_data: FloatArray,
    field_ids: np.ndarray,
    *,
    feature_name: str,
) -> None:
    if feature_data.shape != field_ids.shape:
        raise ValueError(f"{feature_name} feature data and field_ids must have matching shapes.")
