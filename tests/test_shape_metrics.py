from __future__ import annotations

import pytest
from shapely.geometry import GeometryCollection, Point, box

from deep_earth_degasation.morphology.shape import compute_shape_metrics, ringness_score


def test_circle_like_geometry_has_high_circularity() -> None:
    circle = Point(0.0, 0.0).buffer(10.0, quad_segs=32)

    metrics = compute_shape_metrics(circle)

    assert metrics.circularity > 0.95
    assert metrics.equivalent_diameter > 19.0


def test_rectangle_has_expected_elongation() -> None:
    rectangle = box(0.0, 0.0, 10.0, 2.0)

    metrics = compute_shape_metrics(rectangle)

    assert metrics.elongation == pytest.approx(5.0)


def test_empty_geometry_raises() -> None:
    with pytest.raises(ValueError, match="Geometry is empty"):
        compute_shape_metrics(GeometryCollection())


def test_ringness_positive_when_annulus_is_stronger() -> None:
    assert ringness_score(annulus_anomaly=0.9, center_anomaly=0.2, background_anomaly=0.3) > 0
