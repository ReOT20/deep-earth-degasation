from __future__ import annotations

from typing import Any

import geopandas as gpd
from pyproj import CRS


class CRSValidationError(ValueError):
    """Raised when a vector layer CRS is not valid for metric analysis."""


def parse_crs(value: Any) -> CRS:
    """Parse a CRS value into a pyproj CRS."""
    try:
        return CRS.from_user_input(value)
    except Exception as exc:
        raise CRSValidationError(f"Invalid CRS: {value!r}") from exc


def is_geographic_crs(value: Any) -> bool:
    """Return True when a CRS uses angular lon/lat coordinates."""
    return parse_crs(value).is_geographic


def is_projected_metric_crs(value: Any) -> bool:
    """Return True when a CRS is projected and uses metre linear units."""
    crs = parse_crs(value)
    metric_units = {"meter", "metre"}
    return bool(
        crs.is_projected
        and crs.axis_info
        and all(axis.unit_name.lower() in metric_units for axis in crs.axis_info)
    )


def require_projected_metric_crs(value: Any, layer_name: str = "vector") -> CRS:
    """Require a projected metre CRS before metric operations."""
    crs = parse_crs(value)
    if crs.is_geographic:
        raise CRSValidationError(
            f"{layer_name} uses geographic CRS {crs.to_string()}; "
            "projected metre coordinates are required for metric operations."
        )
    if not is_projected_metric_crs(crs):
        raise CRSValidationError(
            f"{layer_name} uses non-metric CRS {crs.to_string()}; "
            "projected metre coordinates are required for metric operations."
        )
    return crs


def ensure_metric_crs(
    gdf: gpd.GeoDataFrame,
    *,
    target_crs: Any | None = None,
    allow_reprojection: bool = False,
    layer_name: str = "vector",
) -> gpd.GeoDataFrame:
    """Return a GeoDataFrame in a projected metre CRS.

    GeoPandas geometry operations are Cartesian, so metric analysis requires
    projected metre coordinates. Reprojection is explicit to avoid accidental
    unit changes.
    """
    if gdf.crs is None:
        raise CRSValidationError(f"{layer_name} has no CRS metadata.")

    source_crs = parse_crs(gdf.crs)
    target = None
    if target_crs is not None:
        target = require_projected_metric_crs(target_crs, f"{layer_name} target_crs")

    if is_projected_metric_crs(source_crs):
        if target is not None and allow_reprojection and not source_crs.equals(target):
            return gdf.to_crs(target)
        return gdf

    if not allow_reprojection:
        require_projected_metric_crs(source_crs, layer_name)

    if target is None:
        raise CRSValidationError(
            f"{layer_name} requires target_crs when allow_reprojection is enabled."
        )

    return gdf.to_crs(target)
