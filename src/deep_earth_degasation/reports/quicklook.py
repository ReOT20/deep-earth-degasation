from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
from shapely.geometry.base import BaseGeometry

from deep_earth_degasation.io.raster_stack import FloatArray

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def write_quicklook_png(
    raster: FloatArray,
    output_path: Path,
    *,
    candidate_geometry: BaseGeometry | None = None,
    title: str | None = None,
) -> None:
    """Write an optional candidate quicklook PNG from a prepared raster array."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(4, 4), dpi=100)
    masked = np.ma.masked_invalid(raster)
    axis.imshow(masked, cmap="viridis")
    if candidate_geometry is not None:
        minx, miny, maxx, maxy = candidate_geometry.bounds
        axis.plot(
            [minx, maxx, maxx, minx, minx],
            [miny, miny, maxy, maxy, miny],
            color="white",
            linewidth=1.5,
        )
    if title:
        axis.set_title(title)
    axis.set_axis_off()
    figure.tight_layout(pad=0)
    figure.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close(figure)
