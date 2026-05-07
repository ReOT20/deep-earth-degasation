from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from deep_earth_degasation.features.types import DynamicFeatureLayer, DynamicFeatureResult

POST_RAIN_DRYING_DIRECTION = (
    "higher_values_indicate_supporting_post_rain_drying_evidence_not_standalone"
)
RAIN_EVENT_DIRECTION = "supporting_weather_event_context_not_standalone_evidence"
RAINFALL_DIRECTION = "supporting_rainfall_amount_context_not_standalone_evidence"


class WeatherFeatureError(ValueError):
    """Raised when prepared weather context violates the manifest contract."""


@dataclass(frozen=True)
class WeatherEvent:
    date: str
    rainfall_mm: float | None = None
    source_id: str | None = None
    source: str | None = None

    @property
    def context_id(self) -> str:
        return self.source_id or f"weather_event_{self.date}"


@dataclass(frozen=True)
class WeatherContext:
    events: tuple[WeatherEvent, ...]


def load_weather_context(records: list[dict[str, object]]) -> WeatherContext:
    """Load prepared weather/event context records without external downloads."""
    events = tuple(
        _parse_weather_event(record, index=index) for index, record in enumerate(records)
    )
    return WeatherContext(events=_sort_events(events))


def load_weather_context_manifest(path: str | Path) -> WeatherContext:
    """Load a small prepared weather manifest from YAML.

    Expected schema:

    events:
      - date: "2024-05-10"
        rainfall_mm: 12.0
        source_id: "manual_event_1"
        source: "prepared"
    """
    manifest_path = Path(path)
    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    root = _require_mapping(data, "weather context manifest")
    _require_keys(root, required={"events"}, optional=set(), section="weather context manifest")
    raw_events = root["events"]
    if not isinstance(raw_events, list):
        raise WeatherFeatureError("events must be a list.")
    return load_weather_context(
        [_require_mapping(event, f"events[{index}]") for index, event in enumerate(raw_events)]
    )


def weather_features_from_context(context: WeatherContext | None) -> DynamicFeatureResult:
    if context is None or not context.events:
        return DynamicFeatureResult(features=(), missing_data_flags=("missing_weather_context",))

    features: list[DynamicFeatureLayer] = []
    flags: list[str] = []
    for event in context.events:
        features.append(
            DynamicFeatureLayer(
                name="rain_event",
                data=np.array([[1.0]], dtype=np.float64),
                sensor="weather",
                date=event.date,
                source_layer_ids=(event.context_id,),
                evidence_direction=RAIN_EVENT_DIRECTION,
            )
        )
        if event.rainfall_mm is None:
            flags.append(f"missing_rainfall_mm_{event.date}")
            continue
        features.append(
            DynamicFeatureLayer(
                name="rainfall_mm",
                data=np.array([[event.rainfall_mm]], dtype=np.float64),
                sensor="weather",
                date=event.date,
                source_layer_ids=(event.context_id,),
                evidence_direction=RAINFALL_DIRECTION,
            )
        )
    return DynamicFeatureResult(features=tuple(features), missing_data_flags=tuple(flags))


def days_since_last_rain(
    observation_date: str,
    context: WeatherContext | None,
    *,
    rainfall_threshold_mm: float = 1.0,
) -> int | None:
    if context is None:
        return None
    observation = date.fromisoformat(observation_date)
    rain_event = _latest_rain_event_on_or_before(
        observation_date,
        context,
        rainfall_threshold_mm=rainfall_threshold_mm,
    )
    if rain_event is None:
        return None
    return (observation - date.fromisoformat(rain_event.date)).days


def post_rain_drying_delta(
    before_moisture: DynamicFeatureLayer,
    after_moisture: DynamicFeatureLayer,
    *,
    context: WeatherContext | None,
    rainfall_threshold_mm: float = 1.0,
) -> DynamicFeatureResult:
    """Compute a supporting post-rain drying delta when weather and moisture are available."""
    flags: list[str] = []
    if context is None or not context.events:
        flags.append("missing_weather_context")
    if before_moisture.date is None or after_moisture.date is None:
        flags.append("missing_moisture_feature_date")
    if before_moisture.data.shape != after_moisture.data.shape:
        flags.append("post_rain_drying_shape_mismatch")
    if flags:
        return DynamicFeatureResult(features=(), missing_data_flags=tuple(flags))

    after_date = after_moisture.date
    if after_date is None:
        return DynamicFeatureResult(
            features=(), missing_data_flags=("missing_moisture_feature_date",)
        )
    if context is None:
        return DynamicFeatureResult(features=(), missing_data_flags=("missing_weather_context",))

    days_since_rain = days_since_last_rain(
        after_date,
        context,
        rainfall_threshold_mm=rainfall_threshold_mm,
    )
    rain_event = _latest_rain_event_on_or_before(
        after_date,
        context,
        rainfall_threshold_mm=rainfall_threshold_mm,
    )
    if days_since_rain is None or rain_event is None:
        return DynamicFeatureResult(features=(), missing_data_flags=("missing_prior_rain_event",))

    drying_delta = _moisture_drying_delta(before_moisture, after_moisture)
    feature = DynamicFeatureLayer(
        name="post_rain_drying",
        data=drying_delta.astype(np.float64),
        sensor="weather",
        date=after_date,
        source_layer_ids=(
            rain_event.context_id,
            *before_moisture.source_layer_ids,
            *after_moisture.source_layer_ids,
        ),
        evidence_direction=POST_RAIN_DRYING_DIRECTION,
    )
    return DynamicFeatureResult(features=(feature,))


def post_rain_drying_features(
    moisture_features: tuple[DynamicFeatureLayer, ...],
    context: WeatherContext | None,
    *,
    rainfall_threshold_mm: float = 1.0,
) -> DynamicFeatureResult:
    """Compute supporting drying deltas for moisture features bracketing rain events."""
    if context is None or not context.events:
        return DynamicFeatureResult(features=(), missing_data_flags=("missing_weather_context",))
    if not moisture_features:
        return DynamicFeatureResult(
            features=(), missing_data_flags=("missing_moisture_features_for_post_rain_drying",)
        )

    features: list[DynamicFeatureLayer] = []
    flags: list[str] = []
    moisture_by_name = _moisture_features_by_name(moisture_features)
    rain_events = [
        event
        for event in context.events
        if event.rainfall_mm is not None and event.rainfall_mm >= rainfall_threshold_mm
    ]
    if not rain_events:
        return DynamicFeatureResult(features=(), missing_data_flags=("missing_prior_rain_event",))

    for event in rain_events:
        event_had_pair = False
        for feature_name, dated_features in moisture_by_name.items():
            before, after = _bracketing_features(dated_features, event.date)
            if before is None or after is None:
                continue
            event_had_pair = True
            if before.data.shape != after.data.shape:
                flags.append(f"post_rain_drying_shape_mismatch_{event.date}_{feature_name}")
                continue
            features.append(
                DynamicFeatureLayer(
                    name=f"post_rain_drying_{feature_name}",
                    data=_moisture_drying_delta(before, after).astype(np.float64),
                    sensor="weather",
                    date=after.date,
                    source_layer_ids=(
                        event.context_id,
                        *before.source_layer_ids,
                        *after.source_layer_ids,
                    ),
                    evidence_direction=POST_RAIN_DRYING_DIRECTION,
                )
            )
        if not event_had_pair:
            flags.append(f"missing_moisture_pair_for_post_rain_drying_{event.date}")

    return DynamicFeatureResult(features=tuple(features), missing_data_flags=tuple(flags))


def _parse_weather_event(record: dict[str, object], *, index: int) -> WeatherEvent:
    _require_keys(
        record,
        required={"date"},
        optional={"rainfall_mm", "precipitation_mm", "source_id", "source"},
        section=f"events[{index}]",
    )
    rainfall = record.get("rainfall_mm", record.get("precipitation_mm"))
    return WeatherEvent(
        date=_parse_date(record["date"], f"events[{index}].date"),
        rainfall_mm=_optional_float(rainfall, f"events[{index}].rainfall_mm"),
        source_id=None if record.get("source_id") is None else str(record["source_id"]),
        source=None if record.get("source") is None else str(record["source"]),
    )


def _parse_date(value: object, field_name: str) -> str:
    text = str(value)
    try:
        date.fromisoformat(text)
    except ValueError as exc:
        raise WeatherFeatureError(f"{field_name} must be an ISO date.") from exc
    return text


def _optional_float(value: object, field_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise WeatherFeatureError(f"{field_name} must be a number.")
    if isinstance(value, int | float | str):
        try:
            return float(value)
        except ValueError as exc:
            raise WeatherFeatureError(f"{field_name} must be a number.") from exc
    raise WeatherFeatureError(f"{field_name} must be a number.")


def _latest_rain_event_on_or_before(
    observation_date: str,
    context: WeatherContext,
    *,
    rainfall_threshold_mm: float,
) -> WeatherEvent | None:
    observation = date.fromisoformat(observation_date)
    rain_events = [
        event
        for event in context.events
        if event.rainfall_mm is not None
        and event.rainfall_mm >= rainfall_threshold_mm
        and date.fromisoformat(event.date) <= observation
    ]
    if not rain_events:
        return None
    return max(rain_events, key=lambda event: (event.date, event.context_id))


def _sort_events(events: tuple[WeatherEvent, ...]) -> tuple[WeatherEvent, ...]:
    return tuple(sorted(events, key=lambda event: (event.date, event.context_id)))


def _moisture_features_by_name(
    features: tuple[DynamicFeatureLayer, ...],
) -> dict[str, tuple[DynamicFeatureLayer, ...]]:
    by_name: dict[str, list[DynamicFeatureLayer]] = {}
    for feature in features:
        if feature.name not in {"NDMI", "NDWI", "MSI"} or feature.date is None:
            continue
        by_name.setdefault(feature.name, []).append(feature)
    return {
        name: tuple(sorted(layers, key=lambda layer: (layer.date or "", layer.source_layer_ids)))
        for name, layers in by_name.items()
    }


def _bracketing_features(
    features: tuple[DynamicFeatureLayer, ...],
    event_date: str,
) -> tuple[DynamicFeatureLayer | None, DynamicFeatureLayer | None]:
    before = [
        feature for feature in features if feature.date is not None and feature.date < event_date
    ]
    after = [
        feature for feature in features if feature.date is not None and feature.date > event_date
    ]
    return (before[-1] if before else None, after[0] if after else None)


def _moisture_drying_delta(
    before_moisture: DynamicFeatureLayer,
    after_moisture: DynamicFeatureLayer,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    if before_moisture.name == "MSI":
        return after_moisture.data - before_moisture.data
    return before_moisture.data - after_moisture.data


def _require_mapping(value: object, section: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise WeatherFeatureError(f"{section} must be a mapping.")
    return value


def _require_keys(
    data: dict[str, Any],
    *,
    required: set[str],
    optional: set[str],
    section: str,
) -> None:
    keys = set(data)
    missing = required - keys
    if missing:
        raise WeatherFeatureError(f"{section} is missing required keys: {sorted(missing)}.")
    unknown = keys - required - optional
    if unknown:
        raise WeatherFeatureError(f"{section} contains unknown keys: {sorted(unknown)}.")
