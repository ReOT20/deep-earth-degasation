from __future__ import annotations

import math
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
    woody_patches: gpd.GeoDataFrame | None = None
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
    flag_woody_patches: bool = True
    flag_cloud_shadows: bool = True
    flag_harvest_patterns: bool = True
    flag_irrigation: bool = True
    flag_field_edges: bool = True
    flag_linear_objects: bool = True
    road_buffer_m: float = 20.0
    water_buffer_m: float = 20.0
    builtup_buffer_m: float = 50.0
    woody_patch_buffer_m: float = 20.0
    field_edge_buffer_m: float = 20.0
    max_elongation_without_penalty: float = 4.0
    small_object_max_support_pixels: int = 12
    penalties: dict[str, float] = field(
        default_factory=lambda: {
            "field_edge": 0.20,
            "road": 0.30,
            "water": 0.30,
            "built_up": 0.35,
            "excluded_zone": 0.40,
            "quarry": 0.40,
            "woody_patch": 0.25,
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
    distance_field: str


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
    profiles: list[dict[str, object]] = []
    distances_by_field: dict[str, list[float | None]] = {
        risk.distance_field: [] for risk in _spatial_risks(filter_config)
    }

    for _, row in output.iterrows():
        geometry = cast(BaseGeometry, row.geometry)
        flags: list[str] = []
        missing_flags = _existing_missing_flags(row)
        penalty = 0.0
        profile: dict[str, object] = {}

        for risk in _spatial_risks(filter_config):
            if not risk.enabled:
                distances_by_field[risk.distance_field].append(None)
                continue
            layer = getattr(context, risk.context_name)
            if layer is None:
                missing_flags.append(f"missing_context_{risk.context_name}")
                profile[f"{risk.context_name}_context"] = "missing"
                distances_by_field[risk.distance_field].append(None)
                continue
            _require_compatible_crs(objects, layer, risk.context_name)
            distance = _nearest_context_distance_m(geometry, layer)
            distances_by_field[risk.distance_field].append(distance)
            profile[risk.distance_field] = distance
            if _intersects_context(geometry, layer, buffer_m=risk.buffer_m):
                flags.append(risk.flag)
                penalty += filter_config.penalties.get(risk.penalty_key, 0.0)

        if filter_config.flag_field_edges and _has_field_edge_risk(row, filter_config):
            flags.append("field_edge_risk")
            penalty += filter_config.penalties.get("field_edge", 0.0)
        if filter_config.flag_linear_objects and _has_linear_object_risk(row, filter_config):
            flags.append("linear_object_risk")
            penalty += filter_config.penalties.get("linear_object", 0.0)
        if _has_small_object_risk(row, filter_config):
            flags.append("small_object_risk")
        if _has_broad_patch_risk(row):
            flags.append("broad_patch_risk")

        final_flags = sorted(set(flags))
        final_missing_flags = sorted(set(missing_flags))
        profile["flags"] = final_flags
        profile["missing_context"] = [
            flag.removeprefix("missing_context_")
            for flag in final_missing_flags
            if flag.startswith("missing_context_")
        ]
        profile["distance_to_field_edge_m"] = _optional_float(
            _row_value(row, "distance_to_field_edge_m")
        )
        profile["elongation"] = _optional_float(_row_value(row, "elongation"))
        profile["support_pixel_count"] = _optional_int(_row_value(row, "support_pixel_count"))
        all_flags.append(final_flags)
        all_missing_flags.append(final_missing_flags)
        penalties.append(round(penalty, 10))
        profiles.append(profile)

    output["false_positive_flags"] = all_flags
    output["false_positive_penalty"] = penalties
    output["missing_data_flags"] = all_missing_flags
    for field_name, values in distances_by_field.items():
        output[field_name] = values
    output["false_positive_profile"] = profiles
    return output


def _spatial_risks(config: FalsePositiveFilterConfig) -> tuple[_SpatialRisk, ...]:
    return (
        _SpatialRisk(
            "roads", "road_risk", "road", config.flag_roads, config.road_buffer_m, "road_distance_m"
        ),
        _SpatialRisk(
            "water",
            "water_risk",
            "water",
            config.flag_water,
            config.water_buffer_m,
            "water_distance_m",
        ),
        _SpatialRisk(
            "built_up",
            "built_up_risk",
            "built_up",
            config.flag_built_up,
            config.builtup_buffer_m,
            "built_up_distance_m",
        ),
        _SpatialRisk(
            "excluded_zones",
            "excluded_zone_risk",
            "excluded_zone",
            config.flag_excluded_zones,
            0.0,
            "excluded_zone_distance_m",
        ),
        _SpatialRisk(
            "quarries",
            "quarry_risk",
            "quarry",
            config.flag_quarries,
            0.0,
            "quarry_distance_m",
        ),
        _SpatialRisk(
            "woody_patches",
            "woody_patch_risk",
            "woody_patch",
            config.flag_woody_patches,
            config.woody_patch_buffer_m,
            "woody_patch_distance_m",
        ),
        _SpatialRisk(
            "cloud_shadows",
            "cloud_shadow_risk",
            "cloud_shadow",
            config.flag_cloud_shadows,
            0.0,
            "cloud_shadow_distance_m",
        ),
        _SpatialRisk(
            "harvest_patterns",
            "harvest_pattern_risk",
            "harvest_pattern",
            config.flag_harvest_patterns,
            0.0,
            "harvest_pattern_distance_m",
        ),
        _SpatialRisk(
            "irrigation",
            "irrigation_risk",
            "irrigation",
            config.flag_irrigation,
            0.0,
            "irrigation_distance_m",
        ),
    )


def _intersects_context(
    geometry: BaseGeometry, context_layer: gpd.GeoDataFrame, *, buffer_m: float
) -> bool:
    prepared_geometry = geometry.buffer(buffer_m) if buffer_m > 0 else geometry
    return any(
        prepared_geometry.intersects(cast(BaseGeometry, context_geometry))
        for context_geometry in context_layer.geometry
    )


def _nearest_context_distance_m(
    geometry: BaseGeometry, context_layer: gpd.GeoDataFrame
) -> float | None:
    distances = [
        geometry.distance(cast(BaseGeometry, context_geometry))
        for context_geometry in context_layer.geometry
    ]
    if not distances:
        return None
    return round(float(min(distances)), 6)


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


def _has_small_object_risk(row: object, config: FalsePositiveFilterConfig) -> bool:
    support_pixels = _optional_int(_row_value(row, "support_pixel_count"))
    if support_pixels is None:
        return False
    return support_pixels <= config.small_object_max_support_pixels


def _has_broad_patch_risk(row: object) -> bool:
    flags = _row_value(row, "dynamic_object_flags")
    if isinstance(flags, str):
        return flags == "broad_patch"
    if isinstance(flags, list | tuple | set):
        return "broad_patch" in {str(flag) for flag in flags}
    return False


def _existing_missing_flags(row: object) -> list[str]:
    value = _row_value(row, "missing_data_flags")
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list | tuple | set):
        return [str(flag) for flag in value]
    return []


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


def _optional_int(value: object | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return int(value)
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
