from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
from shapely import make_valid

from deep_earth_degasation.geo.crs import ensure_metric_crs


class VectorDataError(ValueError):
    """Raised when vector data cannot be loaded deterministically."""


@dataclass(frozen=True)
class VectorIssue:
    feature_index: int
    reason: str


@dataclass(frozen=True)
class VectorLayer:
    name: str
    role: str
    path: Path
    crs: str
    data: gpd.GeoDataFrame
    issues: tuple[VectorIssue, ...] = ()


def load_vector_layer(
    path: str | Path,
    *,
    name: str,
    role: str,
    id_field: str | None = None,
    stable_id_column: str = "stable_id",
    target_crs: Any | None = None,
    allow_reprojection: bool = False,
    repair_invalid: bool = True,
) -> VectorLayer:
    """Load a vector layer with CRS validation, geometry repair and stable IDs."""
    layer_path = Path(path)
    gdf = gpd.read_file(layer_path)
    if gdf.empty:
        raise VectorDataError(f"{name} contains no features.")
    if gdf.geometry.name is None:
        raise VectorDataError(f"{name} has no active geometry column.")

    metric_gdf = ensure_metric_crs(
        gdf,
        target_crs=target_crs,
        allow_reprojection=allow_reprojection,
        layer_name=name,
    ).copy()
    repaired_gdf, issues = _repair_geometries(metric_gdf, name=name, repair_invalid=repair_invalid)
    with_ids = assign_stable_ids(
        repaired_gdf,
        id_field=id_field,
        stable_id_column=stable_id_column,
    )
    return VectorLayer(
        name=name,
        role=role,
        path=layer_path,
        crs=str(with_ids.crs),
        data=with_ids,
        issues=tuple(issues),
    )


def assign_stable_ids(
    gdf: gpd.GeoDataFrame,
    *,
    id_field: str | None = None,
    stable_id_column: str = "stable_id",
) -> gpd.GeoDataFrame:
    """Assign deterministic vector IDs from a source field or geometry hash.

    Source IDs are preserved when present and nonblank. Missing or blank source
    IDs fall back to a geometry hash so ID generation remains deterministic
    without requiring every input layer to carry a complete identifier column.
    """
    output = gdf.copy()
    stable_ids: list[str] = []
    if id_field is not None and id_field not in output.columns:
        raise VectorDataError(f"id_field {id_field!r} does not exist.")

    for _, row in output.iterrows():
        source_id = row[id_field] if id_field is not None else None
        if source_id is None or str(source_id).strip() == "":
            stable_ids.append(_geometry_hash(row.geometry))
        else:
            stable_ids.append(str(source_id))

    output[stable_id_column] = stable_ids
    return output


def _repair_geometries(
    gdf: gpd.GeoDataFrame, *, name: str, repair_invalid: bool
) -> tuple[gpd.GeoDataFrame, list[VectorIssue]]:
    output = gdf.copy()
    issues: list[VectorIssue] = []
    repaired_geometries = []

    for index, geometry in enumerate(output.geometry, start=1):
        if geometry is None or geometry.is_empty:
            raise VectorDataError(f"{name} feature {index} has empty geometry.")
        if geometry.is_valid:
            repaired_geometries.append(geometry)
            continue
        if not repair_invalid:
            raise VectorDataError(f"{name} feature {index} has invalid geometry.")

        repaired = make_valid(geometry)
        if repaired is None or repaired.is_empty or not repaired.is_valid:
            raise VectorDataError(f"{name} feature {index} could not be repaired.")
        issues.append(VectorIssue(feature_index=index, reason="invalid_geometry_repaired"))
        repaired_geometries.append(repaired)

    output = output.set_geometry(
        gpd.GeoSeries(repaired_geometries, crs=output.crs, index=output.index)
    )
    return output, issues


def _geometry_hash(geometry: Any) -> str:
    normalized = geometry.normalize()
    digest = hashlib.sha256(normalized.wkb).hexdigest()[:16]
    return f"geom-{digest}"
