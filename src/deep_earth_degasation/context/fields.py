from __future__ import annotations

from typing import cast

import geopandas as gpd
from shapely.geometry.base import BaseGeometry

from deep_earth_degasation.geo.crs import is_projected_metric_crs, parse_crs


class ContextAssignmentError(ValueError):
    """Raised when spatial context cannot be assigned deterministically."""


def assign_field_context(
    objects: gpd.GeoDataFrame,
    fields: gpd.GeoDataFrame,
    *,
    field_id_column: str = "field_id",
    min_distance_to_field_edge_m: float = 20.0,
) -> gpd.GeoDataFrame:
    """Attach field ID and field-edge context to object geometries."""
    _require_compatible_metric_crs(objects, fields, left_name="objects", right_name="fields")
    _require_column(fields, field_id_column, "fields")

    output = objects.copy()
    field_ids: list[str] = []
    distances: list[float | None] = []
    near_edge_flags: list[bool] = []
    context_flags: list[list[str]] = []

    for geometry in output.geometry:
        object_geometry = cast(BaseGeometry, geometry)
        best_field = _largest_intersecting_field(
            object_geometry,
            fields,
            field_id_column=field_id_column,
        )
        flags: list[str] = []
        if best_field is None:
            field_ids.append("")
            distances.append(None)
            near_edge_flags.append(False)
            context_flags.append(["missing_field_context"])
            continue

        best_field_id, field_geometry = best_field
        distance_to_edge = float(object_geometry.distance(field_geometry.boundary))
        is_near_edge = distance_to_edge < min_distance_to_field_edge_m
        if is_near_edge:
            flags.append("near_field_edge")

        field_ids.append(best_field_id)
        distances.append(distance_to_edge)
        near_edge_flags.append(is_near_edge)
        context_flags.append(flags)

    output["field_id"] = field_ids
    output["distance_to_field_edge_m"] = distances
    output["near_field_edge"] = near_edge_flags
    output["field_context_flags"] = context_flags
    return output


def _largest_intersecting_field(
    object_geometry: BaseGeometry,
    fields: gpd.GeoDataFrame,
    *,
    field_id_column: str,
) -> tuple[str, BaseGeometry] | None:
    best_field = None
    best_area = 0.0
    for _, field in fields.iterrows():
        field_geometry = cast(BaseGeometry, field.geometry)
        intersection_area = float(object_geometry.intersection(field_geometry).area)
        if intersection_area > best_area:
            best_area = intersection_area
            best_field = (str(field[field_id_column]), field_geometry)
    return best_field


def _require_compatible_metric_crs(
    left: gpd.GeoDataFrame,
    right: gpd.GeoDataFrame,
    *,
    left_name: str,
    right_name: str,
) -> None:
    if left.crs is None:
        raise ContextAssignmentError(f"{left_name} has no CRS metadata.")
    if right.crs is None:
        raise ContextAssignmentError(f"{right_name} has no CRS metadata.")

    left_crs = parse_crs(left.crs)
    right_crs = parse_crs(right.crs)
    if not is_projected_metric_crs(left_crs):
        raise ContextAssignmentError(f"{left_name} must use a projected metre CRS.")
    if not is_projected_metric_crs(right_crs):
        raise ContextAssignmentError(f"{right_name} must use a projected metre CRS.")
    if not left_crs.equals(right_crs):
        raise ContextAssignmentError(
            f"{left_name} and {right_name} must use the same CRS before context assignment."
        )


def _require_column(gdf: gpd.GeoDataFrame, column: str, layer_name: str) -> None:
    if column not in gdf.columns:
        raise ContextAssignmentError(f"{layer_name} is missing required column {column!r}.")
