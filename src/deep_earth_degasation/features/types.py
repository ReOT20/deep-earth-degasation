from __future__ import annotations

from dataclasses import dataclass

from deep_earth_degasation.io.raster_stack import FloatArray


@dataclass(frozen=True)
class DynamicFeatureLayer:
    name: str
    data: FloatArray
    sensor: str
    date: str | None
    source_layer_ids: tuple[str, ...]
    evidence_direction: str
    missing_data_flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class DynamicFeatureResult:
    features: tuple[DynamicFeatureLayer, ...]
    missing_data_flags: tuple[str, ...] = ()
