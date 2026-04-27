from __future__ import annotations

import math
from dataclasses import dataclass

from shapely.geometry.base import BaseGeometry


@dataclass(frozen=True)
class ShapeMetrics:
    area: float
    perimeter: float
    equivalent_diameter: float
    circularity: float
    compactness: float
    bbox_width: float
    bbox_height: float
    elongation: float


def compute_shape_metrics(geometry: BaseGeometry) -> ShapeMetrics:
    """Compute basic object shape metrics for projected geometries."""
    if geometry.is_empty:
        raise ValueError("Geometry is empty")
    area = float(geometry.area)
    perimeter = float(geometry.length)
    equivalent_diameter = 2.0 * math.sqrt(area / math.pi) if area > 0 else 0.0
    circularity = (4.0 * math.pi * area / (perimeter**2)) if perimeter > 0 else 0.0
    compactness = (area / (perimeter**2)) if perimeter > 0 else 0.0
    minx, miny, maxx, maxy = geometry.bounds
    bbox_width = float(maxx - minx)
    bbox_height = float(maxy - miny)
    shortest = max(min(bbox_width, bbox_height), 1e-9)
    longest = max(bbox_width, bbox_height)
    elongation = float(longest / shortest)
    return ShapeMetrics(
        area,
        perimeter,
        equivalent_diameter,
        circularity,
        compactness,
        bbox_width,
        bbox_height,
        elongation,
    )


def ringness_score(
    annulus_anomaly: float, center_anomaly: float, background_anomaly: float
) -> float:
    """Simple ringness proxy. Positive means annulus is more anomalous than center/background."""
    return float(annulus_anomaly - max(center_anomaly, background_anomaly))
