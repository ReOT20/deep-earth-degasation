from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from affine import Affine
from shapely.geometry import LinearRing, LineString, MultiLineString, MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry

from deep_earth_degasation.io.raster_stack import FloatArray

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def write_quicklook_png(
    raster: FloatArray,
    output_path: Path,
    *,
    candidate_geometry: BaseGeometry | None = None,
    transform: Affine | None = None,
    title: str | None = None,
) -> None:
    """Write an optional candidate quicklook PNG from a prepared raster array."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(4, 4), dpi=100)
    view, extent = _candidate_view(raster, candidate_geometry, transform)
    masked = np.ma.masked_invalid(view)
    if extent is None:
        axis.imshow(masked, cmap="viridis")
        if candidate_geometry is not None and transform is None:
            _plot_geometry_bounds(axis, candidate_geometry)
    else:
        axis.imshow(masked, cmap="viridis", extent=extent, origin="upper")
        if candidate_geometry is not None:
            _plot_geometry(axis, candidate_geometry)
    if title:
        axis.set_title(title)
    axis.set_axis_off()
    figure.tight_layout(pad=0)
    figure.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close(figure)


def _candidate_view(
    raster: FloatArray,
    candidate_geometry: BaseGeometry | None,
    transform: Affine | None,
    *,
    padding_pixels: int = 5,
) -> tuple[FloatArray, tuple[float, float, float, float] | None]:
    if candidate_geometry is None or transform is None or candidate_geometry.is_empty:
        return raster, None

    minx, miny, maxx, maxy = candidate_geometry.bounds
    inverse = ~transform
    pixels = [
        inverse * (minx, miny),
        inverse * (minx, maxy),
        inverse * (maxx, miny),
        inverse * (maxx, maxy),
    ]
    columns = [point[0] for point in pixels]
    rows = [point[1] for point in pixels]
    row_start = max(0, int(np.floor(min(rows))) - padding_pixels)
    row_stop = min(raster.shape[0], int(np.ceil(max(rows))) + padding_pixels)
    column_start = max(0, int(np.floor(min(columns))) - padding_pixels)
    column_stop = min(raster.shape[1], int(np.ceil(max(columns))) + padding_pixels)
    if row_start >= row_stop or column_start >= column_stop:
        return raster, None

    left, top = transform * (column_start, row_start)
    right, bottom = transform * (column_stop, row_stop)
    extent = (left, right, bottom, top)
    return raster[row_start:row_stop, column_start:column_stop], extent


def _plot_geometry(axis: Any, geometry: BaseGeometry) -> None:
    if isinstance(geometry, Polygon):
        _plot_line(axis, geometry.exterior.coords)
        for interior in geometry.interiors:
            _plot_line(axis, interior.coords)
        return
    if isinstance(geometry, MultiPolygon):
        for part in geometry.geoms:
            _plot_geometry(axis, part)
        return
    if isinstance(geometry, LineString | LinearRing):
        _plot_line(axis, geometry.coords)
        return
    if isinstance(geometry, MultiLineString):
        for part in geometry.geoms:
            _plot_geometry(axis, part)
        return
    minx, miny, maxx, maxy = geometry.bounds
    _plot_bounds(axis, minx, miny, maxx, maxy)


def _plot_geometry_bounds(axis: Any, geometry: BaseGeometry) -> None:
    minx, miny, maxx, maxy = geometry.bounds
    _plot_bounds(axis, minx, miny, maxx, maxy)


def _plot_bounds(axis: Any, minx: float, miny: float, maxx: float, maxy: float) -> None:
    _plot_line(axis, [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy), (minx, miny)])


def _plot_line(axis: Any, coords: Any) -> None:
    xs, ys = zip(*coords, strict=True)
    axis.plot(xs, ys, color="white", linewidth=1.5)
