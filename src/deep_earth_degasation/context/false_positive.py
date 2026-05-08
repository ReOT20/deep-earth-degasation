from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import geopandas as gpd
from shapely.geometry.base import BaseGeometry

from deep_earth_degasation.context.fields import ContextAssignmentError
from deep_earth_degasation.geo.crs import is_projected_metric_crs, parse_crs


@dataclass(frozen=True)
class FalsePositiveContext:
    roads: gpd.GeoDataFrame | None = None
    water: gpd.GeoDataFrame | None = None
    built_up: gpd.GeoDataFrame | None = None
    excluded_zones: gpd.GeoDataFrame | None = None
    quarries: gpd.GeoDataFrame | None = None
    cloud_shadows: gpd.GeoDataFrame | None = None
    harvest_patterns: gpd.GeoDataFrame | None = None
    irrigation: gpd.GeoDataFrame | None = None


@dataclass(frozen=True)
class FalsePositiveFilterConfig:
    flag_roads: bool = True
    flag_water: bool = True
    flag_built_up: bool = True
    flag_excluded_zones: bool = True
    flag_quarries: bool = True
    flag_cloud_shadows: bool = True
    flag_harvest_patterns: bool = True
    flag_irrigation: bool = True
    flag_field_edges: bool = True
    flag_linear_objects: bool = True
    road_buffer_m: float = 20.0
    water_buffer_m: float = 20.0
    builtup_buffer_m: float = 50.0
    field_edge_buffer_m: float = 20.0
    max_elongation_without_penalty: float = 4.0
    penalties: dict[str, float] = field(
        default_factory=lambda: {
            "field_edge": 0.20,
            "road": 0.30,
            "water": 0.30,
            "built_up": 0.35,
            "excluded_zone": 0.40,
            "quarry": 0.40,
            "linear_object": 0.25,
            "cloud_shadow": 0.20,
            "harvest_pattern": 0.20,
            "irrigation": 0.20,
        }
    )


@dataclass(frozen=True)
class _SpatialRisk:
    context_name: str
    flag: str
    penalty_key: str
    enabled: bool
    buffer_m: float


def apply_false_positive_filters(
    objects: gpd.GeoDataFrame,
    context: FalsePositiveContext,
    config: FalsePositiveFilterConfig | None = None,
) -> gpd.GeoDataFrame:
    """Annotate candidate objects with false-positive context flags."""
    filter_config = config or FalsePositiveFilterConfig()
    _require_metric_crs(objects, "objects")

    output = objects.copy()
    all_flags: list[list[str]] = []
    all_missing_flags: list[list[str]] = []
    penalties: list[float] = []

    for _, row in output.iterrows():
        geometry = cast(BaseGeometry, row.geometry)
        flags: list[str] = []
        missing_flags: list[str] = []
        penalty = 0.0

        for risk in _spatial_risks(filter_config):
            if not risk.enabled:
                continue
            layer = getattr(context, risk.context_name)
            if layer is None:
                missing_flags.append(f"missing_context_{risk.context_name}")
                continue
            _require_compatible_crs(objects, layer, risk.context_name)
            if _intersects_context(geometry, layer, buffer_m=risk.buffer_m):
                flags.append(risk.flag)
                penalty += filter_config.penalties.get(risk.penalty_key, 0.0)

        if filter_config.flag_field_edges and _has_field_edge_risk(row, filter_config):
            flags.append("field_edge_risk")
            penalty += filter_config.penalties.get("field_edge", 0.0)
        if filter_config.flag_linear_objects and _has_linear_object_risk(row, filter_config):
            flags.append("linear_object_risk")
            penalty += filter_config.penalties.get("linear_object", 0.0)

        all_flags.append(sorted(set(flags)))
        all_missing_flags.append(sorted(set(missing_flags)))
        penalties.append(round(penalty, 10))

    output["false_positive_flags"] = all_flags
    output["false_positive_penalty"] = penalties
    output["missing_data_flags"] = all_missing_flags
    return output


def _spatial_risks(config: FalsePositiveFilterConfig) -> tuple[_SpatialRisk, ...]:
    return (
        _SpatialRisk("roads", "road_risk", "road", config.flag_roads, config.road_buffer_m),
        _SpatialRisk("water", "water_risk", "water", config.flag_water, config.water_buffer_m),
        _SpatialRisk(
            "built_up",
            "built_up_risk",
            "built_up",
            config.flag_built_up,
            config.builtup_buffer_m,
        ),
        _SpatialRisk(
            "excluded_zones",
            "excluded_zone_risk",
            "excluded_zone",
            config.flag_excluded_zones,
            0.0,
        ),
        _SpatialRisk("quarries", "quarry_risk", "quarry", config.flag_quarries, 0.0),
        _SpatialRisk(
            "cloud_shadows",
            "cloud_shadow_risk",
            "cloud_shadow",
            config.flag_cloud_shadows,
            0.0,
        ),
        _SpatialRisk(
            "harvest_patterns",
            "harvest_pattern_risk",
            "harvest_pattern",
            config.flag_harvest_patterns,
            0.0,
        ),
        _SpatialRisk("irrigation", "irrigation_risk", "irrigation", config.flag_irrigation, 0.0),
    )


def _intersects_context(
    geometry: BaseGeometry, context_layer: gpd.GeoDataFrame, *, buffer_m: float
) -> bool:
    prepared_geometry = geometry.buffer(buffer_m) if buffer_m > 0 else geometry
    return any(
        prepared_geometry.intersects(cast(BaseGeometry, context_geometry))
        for context_geometry in context_layer.geometry
    )


def _has_field_edge_risk(row: object, config: FalsePositiveFilterConfig) -> bool:
    if _row_value(row, "near_field_edge") is True:
        return True
    distance = _optional_float(_row_value(row, "distance_to_field_edge_m"))
    if distance is None:
        return False
    return distance < config.field_edge_buffer_m


def _has_linear_object_risk(row: object, config: FalsePositiveFilterConfig) -> bool:
    elongation = _optional_float(_row_value(row, "elongation"))
    if elongation is None:
        return False
    return elongation > config.max_elongation_without_penalty


def _row_value(row: Any, key: str) -> object | None:
    if hasattr(row, "index") and key in row.index:
        return row[key]
    return None


def _optional_float(value: object | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _require_compatible_crs(
    objects: gpd.GeoDataFrame, context_layer: gpd.GeoDataFrame, context_name: str
) -> None:
    _require_metric_crs(context_layer, context_name)
    if not parse_crs(objects.crs).equals(parse_crs(context_layer.crs)):
        raise ContextAssignmentError(
            f"objects and {context_name} must use the same CRS before false-positive filtering."
        )


def _require_metric_crs(gdf: gpd.GeoDataFrame, layer_name: str) -> None:
    if gdf.crs is None:
        raise ContextAssignmentError(f"{layer_name} has no CRS metadata.")
    crs = parse_crs(gdf.crs)
    if not is_projected_metric_crs(crs):
        raise ContextAssignmentError(f"{layer_name} must use a projected metre CRS.")
