from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry

from deep_earth_degasation.morphology.shape import ShapeMetrics, compute_shape_metrics


@dataclass(frozen=True)
class StaticDetectorConfig:
    min_diameter_m: float = 40.0
    max_diameter_m: float = 1500.0
    min_area: float = 100.0
    max_area: float = 2_000_000.0
    circle_min_circularity: float = 0.85
    circle_max_elongation: float = 1.25
    ellipse_min_circularity: float = 0.45
    ellipse_max_elongation: float = 3.0
    chain_min_elongation: float = 3.0


@dataclass(frozen=True)
class StaticCandidate:
    candidate_id: str
    geometry: BaseGeometry
    morphology_type: str
    shape_metrics: ShapeMetrics
    static_score: float
    evidence: dict[str, float]
    flags: list[str] = field(default_factory=list)


def extract_static_candidates(
    geometries: Iterable[BaseGeometry],
    candidate_ids: Sequence[str] | None = None,
    landcover_context: str | None = None,
    config: StaticDetectorConfig | None = None,
) -> list[StaticCandidate]:
    detector_config = config or StaticDetectorConfig()
    candidates: list[StaticCandidate] = []

    for index, geometry in enumerate(geometries, start=1):
        candidate_id = (
            candidate_ids[index - 1]
            if candidate_ids is not None and index - 1 < len(candidate_ids)
            else f"static-{index:06d}"
        )
        candidates.append(
            _build_candidate(candidate_id, geometry, landcover_context, detector_config)
        )

    return candidates


def _build_candidate(
    candidate_id: str,
    geometry: BaseGeometry,
    landcover_context: str | None,
    config: StaticDetectorConfig,
) -> StaticCandidate:
    metrics = compute_shape_metrics(geometry)
    flags = _flags(metrics, geometry, landcover_context, config)
    morphology_type = _classify_morphology(metrics, geometry, config)
    evidence = {
        "circularity": metrics.circularity,
        "elongation": metrics.elongation,
        "equivalent_diameter": metrics.equivalent_diameter,
        "area": metrics.area,
    }
    static_score = _static_score(morphology_type, metrics, flags, config)
    return StaticCandidate(
        candidate_id=candidate_id,
        geometry=geometry,
        morphology_type=morphology_type,
        shape_metrics=metrics,
        static_score=static_score,
        evidence=evidence,
        flags=flags,
    )


def _classify_morphology(
    metrics: ShapeMetrics, geometry: BaseGeometry, config: StaticDetectorConfig
) -> str:
    if _has_polygon_hole(geometry):
        return "ring"
    if metrics.elongation >= config.chain_min_elongation or _is_multi_part(geometry):
        return "chain"
    if (
        metrics.circularity >= config.circle_min_circularity
        and metrics.elongation <= config.circle_max_elongation
    ):
        return "circle"
    if (
        metrics.circularity >= config.ellipse_min_circularity
        and metrics.elongation <= config.ellipse_max_elongation
    ):
        return "ellipse"
    return "irregular"


def _flags(
    metrics: ShapeMetrics,
    geometry: BaseGeometry,
    landcover_context: str | None,
    config: StaticDetectorConfig,
) -> list[str]:
    flags: list[str] = []
    if metrics.equivalent_diameter < config.min_diameter_m or metrics.area < config.min_area:
        flags.append("too_small")
    if metrics.equivalent_diameter > config.max_diameter_m or metrics.area > config.max_area:
        flags.append("too_large")
    if metrics.elongation >= config.chain_min_elongation:
        flags.append("elongated")
    if _is_multi_part(geometry):
        flags.append("multi_part")
    if landcover_context is not None and landcover_context.lower() == "built_up":
        flags.append("built_up_risk")
    return flags


def _static_score(
    morphology_type: str,
    metrics: ShapeMetrics,
    flags: list[str],
    config: StaticDetectorConfig,
) -> float:
    if "too_small" in flags or "too_large" in flags:
        return 0.0
    if morphology_type == "ring":
        base_score = 0.95
    elif morphology_type == "circle":
        base_score = metrics.circularity
    elif morphology_type == "ellipse":
        base_score = 0.65
    elif morphology_type == "chain":
        base_score = 0.55
    else:
        base_score = 0.25
    if "built_up_risk" in flags:
        base_score -= 0.2
    if metrics.elongation >= config.chain_min_elongation and morphology_type != "chain":
        base_score -= 0.2
    return max(0.0, min(1.0, float(base_score)))


def _has_polygon_hole(geometry: BaseGeometry) -> bool:
    if isinstance(geometry, Polygon):
        return len(geometry.interiors) > 0
    if isinstance(geometry, MultiPolygon):
        return any(len(polygon.interiors) > 0 for polygon in geometry.geoms)
    return False


def _is_multi_part(geometry: BaseGeometry) -> bool:
    return len(getattr(geometry, "geoms", ())) > 1
