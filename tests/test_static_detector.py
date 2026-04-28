from __future__ import annotations

import pytest
from shapely import affinity
from shapely.geometry import GeometryCollection, MultiPolygon, Point, Polygon, box

from deep_earth_degasation.morphology.static_detector import (
    StaticDetectorConfig,
    extract_static_candidates,
)


def test_circle_like_geometry_is_static_candidate() -> None:
    circle = Point(0.0, 0.0).buffer(50.0, quad_segs=32)

    candidate = extract_static_candidates([circle])[0]

    assert candidate.candidate_id == "static-000001"
    assert candidate.morphology_type == "circle"
    assert candidate.static_score > 0.9
    assert candidate.evidence["circularity"] > 0.95


def test_polygon_with_hole_is_ring_candidate() -> None:
    geometry = Polygon(
        [(-50.0, -50.0), (50.0, -50.0), (50.0, 50.0), (-50.0, 50.0)],
        holes=[[(-15.0, -15.0), (15.0, -15.0), (15.0, 15.0), (-15.0, 15.0)]],
    )

    candidate = extract_static_candidates([geometry], candidate_ids=["ring-1"])[0]

    assert candidate.candidate_id == "ring-1"
    assert candidate.morphology_type == "ring"
    assert candidate.static_score == pytest.approx(0.95)


def test_ellipse_like_geometry_is_classified_as_ellipse() -> None:
    circle = Point(0.0, 0.0).buffer(50.0, quad_segs=32)
    ellipse = affinity.scale(circle, xfact=2.0, yfact=1.0, origin=(0.0, 0.0))

    candidate = extract_static_candidates([ellipse])[0]

    assert candidate.morphology_type == "ellipse"
    assert 1.5 < candidate.shape_metrics.elongation < 2.5


def test_multi_part_elongated_geometry_is_chain_candidate() -> None:
    geometry = MultiPolygon(
        [
            Point(0.0, 0.0).buffer(15.0),
            Point(80.0, 0.0).buffer(15.0),
            Point(160.0, 0.0).buffer(15.0),
        ]
    )

    candidate = extract_static_candidates([geometry])[0]

    assert candidate.morphology_type == "chain"
    assert "multi_part" in candidate.flags
    assert "elongated" in candidate.flags


def test_irregular_geometry_is_classified_as_irregular() -> None:
    geometry = Polygon(
        [
            (0.0, 0.0),
            (120.0, 0.0),
            (95.0, 35.0),
            (140.0, 90.0),
            (45.0, 65.0),
            (0.0, 120.0),
        ]
    )

    candidate = extract_static_candidates([geometry])[0]

    assert candidate.morphology_type == "irregular"
    assert 0.0 <= candidate.static_score <= 1.0


def test_size_filter_flags_zero_score_candidates() -> None:
    tiny = Point(0.0, 0.0).buffer(2.0)
    large = Point(0.0, 0.0).buffer(5000.0)

    candidates = extract_static_candidates([tiny, large])

    assert "too_small" in candidates[0].flags
    assert candidates[0].static_score == 0.0
    assert "too_large" in candidates[1].flags
    assert candidates[1].static_score == 0.0


def test_empty_geometry_raises() -> None:
    with pytest.raises(ValueError, match="Geometry is empty"):
        extract_static_candidates([GeometryCollection()])


def test_built_up_context_adds_false_positive_flag() -> None:
    geometry = box(0.0, 0.0, 60.0, 60.0)

    candidate = extract_static_candidates([geometry], landcover_context="built_up")[0]

    assert "built_up_risk" in candidate.flags
    assert candidate.static_score < 1.0


def test_custom_config_controls_size_thresholds() -> None:
    geometry = Point(0.0, 0.0).buffer(20.0)
    config = StaticDetectorConfig(min_diameter_m=10.0, min_area=10.0)

    candidate = extract_static_candidates([geometry], config=config)[0]

    assert "too_small" not in candidate.flags
    assert candidate.static_score > 0.0
