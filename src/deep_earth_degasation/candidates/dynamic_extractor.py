from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

import geopandas as gpd
import numpy as np
from affine import Affine
from rasterio.features import shapes
from scipy.ndimage import binary_dilation
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry
from skimage.measure import label, regionprops

from deep_earth_degasation.anomaly.composite import CompositeAnomalyMap
from deep_earth_degasation.anomaly.field_normalization import FieldAnomalyLayer
from deep_earth_degasation.candidates.object_features import compute_object_zonal_stats
from deep_earth_degasation.context.fields import assign_field_context
from deep_earth_degasation.io.raster_stack import FloatArray
from deep_earth_degasation.morphology.shape import compute_shape_metrics, ringness_score


@dataclass(frozen=True)
class DynamicExtractionConfig:
    anomaly_percentile: float = 95.0
    support_percentile: float | None = None
    min_support_pixels: int = 1
    min_area_m2: float = 1_000.0
    max_area_m2: float = 800_000.0
    min_diameter_m: float = 40.0
    max_diameter_m: float = 1_000.0
    max_elongation: float = 4.0
    broad_patch_min_area_m2: float | None = 50_000.0
    broad_patch_max_circularity: float = 1.0
    broad_patch_max_ringness: float = 0.0
    merge_distance_m: float = 30.0
    merge_across_dates: bool = True
    min_distance_to_field_edge_m: float = 20.0
    connectivity: int = 2


def extract_dynamic_objects(
    anomaly_layers: tuple[FieldAnomalyLayer, ...],
    field_ids: np.ndarray,
    *,
    transform: Affine,
    crs: str,
    config: DynamicExtractionConfig | None = None,
    fields: gpd.GeoDataFrame | None = None,
    field_id_column: str = "field_id",
) -> gpd.GeoDataFrame:
    """Extract review candidate objects from field-normalized anomaly maps."""
    return _extract_dynamic_objects_from_layers(
        anomaly_layers,
        anomaly_layers,
        field_ids,
        transform=transform,
        crs=crs,
        config=config,
        fields=fields,
        field_id_column=field_id_column,
        merge_detections=True,
    )


def extract_dynamic_objects_from_composite(
    composite: CompositeAnomalyMap,
    anomaly_layers: tuple[FieldAnomalyLayer, ...],
    field_ids: np.ndarray,
    *,
    transform: Affine,
    crs: str,
    config: DynamicExtractionConfig | None = None,
    fields: gpd.GeoDataFrame | None = None,
    field_id_column: str = "field_id",
) -> gpd.GeoDataFrame:
    """Extract review objects from the composite map and explain them with source layers."""
    composite_layer = FieldAnomalyLayer(
        name="composite_anomaly",
        component="composite",
        data=composite.data,
        date=None,
        source_feature_names=composite.source_feature_names,
        source_layer_ids=composite.source_layer_ids,
        evidence_direction="higher_values_indicate_stronger_local_anomaly_support",
        missing_data_flags=composite.missing_data_flags,
    )
    return _extract_dynamic_objects_from_layers(
        (composite_layer,),
        anomaly_layers,
        field_ids,
        transform=transform,
        crs=crs,
        config=config,
        fields=fields,
        field_id_column=field_id_column,
        merge_detections=False,
    )


def _extract_dynamic_objects_from_layers(
    detection_layers: tuple[FieldAnomalyLayer, ...],
    stat_layers: tuple[FieldAnomalyLayer, ...],
    field_ids: np.ndarray,
    *,
    transform: Affine,
    crs: str,
    config: DynamicExtractionConfig | None,
    fields: gpd.GeoDataFrame | None,
    field_id_column: str,
    merge_detections: bool,
) -> gpd.GeoDataFrame:
    extraction_config = config or DynamicExtractionConfig()
    if not detection_layers:
        return _empty_objects(crs)
    _require_common_shapes(detection_layers, field_ids)
    _require_common_shapes(stat_layers, field_ids)

    detections: list[dict[str, Any]] = []
    for layer in detection_layers:
        detection_labels = _field_local_labels(layer.data, field_ids, extraction_config)
        for region in regionprops(detection_labels):
            component_mask = detection_labels == region.label
            geometry = _component_geometry(component_mask, transform)
            if geometry is None or geometry.is_empty:
                continue
            detections.append(
                _object_row(
                    object_id=f"dynamic-detection-{len(detections) + 1:06d}",
                    geometry=geometry,
                    field_id=_majority_field_id(field_ids[component_mask]),
                    layers=(layer,) if merge_detections else stat_layers,
                    mask=component_mask,
                    detection_data=layer.data,
                    config=extraction_config,
                    crs=crs,
                )
            )

    if not detections:
        return _empty_objects(crs)

    detections_gdf = gpd.GeoDataFrame(detections, geometry="geometry", crs=crs)
    merged_gdf = (
        _merge_repeated_detections(detections_gdf, extraction_config)
        if merge_detections and extraction_config.merge_across_dates
        else detections_gdf
    )
    if fields is not None and not merged_gdf.empty:
        merged_gdf = assign_field_context(
            merged_gdf,
            fields,
            field_id_column=field_id_column,
            min_distance_to_field_edge_m=extraction_config.min_distance_to_field_edge_m,
        )
    return assign_stable_candidate_ids(merged_gdf)


def assign_stable_candidate_ids(objects: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Assign deterministic candidate/object IDs from final field and geometry state."""
    if objects.empty:
        return objects

    output = objects.copy()
    base_ids = [_stable_id_base(row) for _, row in output.iterrows()]
    id_counts: dict[str, int] = {}
    for base_id in sorted(set(base_ids)):
        id_counts[base_id] = base_ids.count(base_id)

    seen: dict[str, int] = {}
    candidate_ids: list[str] = []
    for base_id in base_ids:
        count = seen.get(base_id, 0) + 1
        seen[base_id] = count
        candidate_ids.append(base_id if id_counts[base_id] == 1 else f"{base_id}-{count:02d}")

    output["candidate_id"] = candidate_ids
    output["object_id"] = candidate_ids
    return output


def _field_local_labels(
    data: FloatArray,
    field_ids: np.ndarray,
    config: DynamicExtractionConfig,
) -> np.ndarray:
    labels = np.zeros(data.shape, dtype=np.int32)
    next_label = 1
    for field_id in _unique_field_ids(field_ids):
        field_mask = field_ids == field_id
        positive_support_mask = field_mask & np.isfinite(data) & (data > 0)
        values = data[positive_support_mask]
        if values.size == 0:
            continue
        seed_threshold = float(np.nanpercentile(values, config.anomaly_percentile))
        support_threshold = _support_threshold(values, config)
        seed_mask = positive_support_mask & (data >= seed_threshold)
        support_mask = positive_support_mask & (data >= support_threshold)
        field_labels = np.asarray(label(support_mask, connectivity=config.connectivity))
        for component_label in range(1, int(field_labels.max()) + 1):
            component_mask = field_labels == component_label
            if not np.any(component_mask & seed_mask):
                continue
            if int(np.count_nonzero(component_mask)) < config.min_support_pixels:
                continue
            labels[component_mask] = next_label
            next_label += 1
    return labels


def _support_threshold(values: np.ndarray, config: DynamicExtractionConfig) -> float:
    if config.support_percentile is None:
        return float(np.nanpercentile(values, config.anomaly_percentile))
    support_percentile = min(config.support_percentile, config.anomaly_percentile)
    return float(np.nanpercentile(values, support_percentile))


def _component_geometry(mask: np.ndarray, transform: Affine) -> BaseGeometry | None:
    label_image = mask.astype(np.uint8)
    geometries = [
        shape(geometry)
        for geometry, value in shapes(label_image, mask=mask, transform=transform)
        if int(value) == 1
    ]
    if not geometries:
        return None
    geometry = geometries[0]
    for next_geometry in geometries[1:]:
        geometry = geometry.union(next_geometry)
    return geometry


def _object_row(
    *,
    object_id: str,
    geometry: BaseGeometry,
    field_id: str,
    layers: tuple[FieldAnomalyLayer, ...],
    mask: np.ndarray,
    detection_data: FloatArray,
    config: DynamicExtractionConfig,
    crs: str,
) -> dict[str, Any]:
    metrics = compute_shape_metrics(geometry)
    stats = compute_object_zonal_stats(layers, mask)
    support_pixel_count = int(np.count_nonzero(mask))
    pixel_area_m2 = metrics.area / support_pixel_count if support_pixel_count else 0.0
    annulus_contrast, ring_score = _ring_evidence(detection_data, mask)
    flags = _object_flags(
        metrics.area,
        metrics.equivalent_diameter,
        metrics.circularity,
        metrics.elongation,
        ring_score,
        support_pixel_count,
        config,
    )
    return {
        "object_id": object_id,
        "field_id": field_id,
        "geometry": geometry,
        "area_m2": metrics.area,
        "perimeter_m": metrics.perimeter,
        "equivalent_diameter_m": metrics.equivalent_diameter,
        "circularity": metrics.circularity,
        "elongation": metrics.elongation,
        "annulus_contrast": annulus_contrast,
        "ringness_score": ring_score,
        "support_pixel_count": support_pixel_count,
        "pixel_area_m2": pixel_area_m2,
        "mean_anomaly": stats.mean_anomaly,
        "max_anomaly": stats.max_anomaly,
        "per_feature_mean": stats.per_feature_mean,
        "per_feature_max": stats.per_feature_max,
        "field_residual_max": stats.per_residual_type_max.get("field", float("nan")),
        "peer_residual_max": stats.per_residual_type_max.get("peer", float("nan")),
        "temporal_residual_max": stats.per_residual_type_max.get("temporal", float("nan")),
        "residual_types_present": tuple(sorted(stats.per_residual_type_max)),
        "source_feature_names": stats.source_feature_names,
        "source_layer_ids": stats.source_layer_ids,
        "anomalous_dates": stats.anomalous_dates,
        "source_detection_count": stats.supporting_observation_count,
        "repeated_seasons": _season_count(stats.anomalous_dates),
        "dynamic_object_flags": flags,
        "passes_dynamic_filters": not flags,
        "crs": crs,
    }


def _merge_repeated_detections(
    detections: gpd.GeoDataFrame,
    config: DynamicExtractionConfig,
) -> gpd.GeoDataFrame:
    rows = [row.to_dict() for _, row in detections.iterrows()]
    merged: list[dict[str, Any]] = []
    used: set[int] = set()
    for index, row in enumerate(rows):
        if index in used:
            continue
        group = [row]
        used.add(index)
        geometry = row["geometry"]
        for other_index, other in enumerate(rows[index + 1 :], start=index + 1):
            if other_index in used or other["field_id"] != row["field_id"]:
                continue
            if geometry.buffer(config.merge_distance_m).intersects(other["geometry"]):
                group.append(other)
                used.add(other_index)
                geometry = geometry.union(other["geometry"])
        merged.append(
            _merged_row(
                group,
                object_id=f"dynamic-object-{len(merged) + 1:06d}",
                config=config,
            )
        )
    return gpd.GeoDataFrame(merged, geometry="geometry", crs=detections.crs)


def _merged_row(
    group: list[dict[str, Any]], *, object_id: str, config: DynamicExtractionConfig
) -> dict[str, Any]:
    geometry = group[0]["geometry"]
    for row in group[1:]:
        geometry = geometry.union(row["geometry"])
    metrics = compute_shape_metrics(geometry)
    dates = sorted({date for row in group for date in row["anomalous_dates"]})
    pixel_area_m2 = float(group[0].get("pixel_area_m2") or 0.0)
    support_pixel_count = round(metrics.area / pixel_area_m2) if pixel_area_m2 > 0 else 0
    ring_score = float(np.nanmax([row.get("ringness_score", 0.0) for row in group]))
    annulus_contrast = float(np.nanmax([row.get("annulus_contrast", 0.0) for row in group]))
    flags = _object_flags(
        metrics.area,
        metrics.equivalent_diameter,
        metrics.circularity,
        metrics.elongation,
        ring_score,
        support_pixel_count,
        config,
    )
    per_feature_mean = _merge_feature_stats(group, "per_feature_mean")
    per_feature_max = _merge_feature_stats(group, "per_feature_max", use_max=True)
    residual_types_present = tuple(
        sorted({residual_type for row in group for residual_type in row["residual_types_present"]})
    )
    return {
        **group[0],
        "object_id": object_id,
        "geometry": geometry,
        "area_m2": metrics.area,
        "perimeter_m": metrics.perimeter,
        "equivalent_diameter_m": metrics.equivalent_diameter,
        "circularity": metrics.circularity,
        "elongation": metrics.elongation,
        "annulus_contrast": annulus_contrast,
        "ringness_score": ring_score,
        "support_pixel_count": support_pixel_count,
        "pixel_area_m2": pixel_area_m2,
        "mean_anomaly": float(np.nanmean([row["mean_anomaly"] for row in group])),
        "max_anomaly": float(np.nanmax([row["max_anomaly"] for row in group])),
        "per_feature_mean": per_feature_mean,
        "per_feature_max": per_feature_max,
        "field_residual_max": _merge_numeric_max(group, "field_residual_max"),
        "peer_residual_max": _merge_numeric_max(group, "peer_residual_max"),
        "temporal_residual_max": _merge_numeric_max(group, "temporal_residual_max"),
        "residual_types_present": residual_types_present,
        "source_feature_names": tuple(
            sorted({name for row in group for name in row["source_feature_names"]})
        ),
        "source_layer_ids": tuple(
            sorted({layer_id for row in group for layer_id in row["source_layer_ids"]})
        ),
        "anomalous_dates": tuple(dates),
        "source_detection_count": sum(int(row.get("source_detection_count") or 0) for row in group),
        "repeated_seasons": _season_count(tuple(dates)),
        "dynamic_object_flags": flags,
        "passes_dynamic_filters": not flags,
    }


def _stable_id_base(row: Any) -> str:
    geometry = row.geometry if hasattr(row, "geometry") else row["geometry"]
    centroid = geometry.centroid
    centroid_label = (
        f"x{_round_to_nearest(centroid.x, 10):.0f}y{_round_to_nearest(centroid.y, 10):.0f}"
    )
    field_id = _safe_id_part(str(_row_value(row, "field_id") or "unknownfield"))
    geometry_hash = hashlib.sha256(geometry.normalize().wkb).hexdigest()[:8]
    return f"dyn-{field_id}-{centroid_label}-{geometry_hash}"


def _round_to_nearest(value: float, step: int) -> float:
    return round(value / step) * step


def _safe_id_part(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-").lower()
    return safe or "unknownfield"


def _row_value(row: Any, key: str) -> object | None:
    if hasattr(row, "index") and key in row.index:
        return row[key]
    if isinstance(row, dict):
        return row.get(key)
    return None


def _merge_feature_stats(
    group: list[dict[str, Any]],
    field_name: str,
    *,
    use_max: bool = False,
) -> dict[str, float]:
    feature_names = {name for row in group for name in row[field_name]}
    stats: dict[str, float] = {}
    for feature_name in feature_names:
        values = [row[field_name][feature_name] for row in group if feature_name in row[field_name]]
        stats[feature_name] = float(np.nanmax(values) if use_max else np.nanmean(values))
    return stats


def _merge_numeric_max(group: list[dict[str, Any]], field_name: str) -> float:
    values = [float(row[field_name]) for row in group if np.isfinite(float(row[field_name]))]
    return float(np.nanmax(values)) if values else float("nan")


def _object_flags(
    area_m2: float,
    diameter_m: float,
    circularity: float,
    elongation: float,
    ringness: float,
    support_pixel_count: int,
    config: DynamicExtractionConfig,
) -> list[str]:
    flags: list[str] = []
    if (
        area_m2 < config.min_area_m2
        or diameter_m < config.min_diameter_m
        or support_pixel_count < config.min_support_pixels
    ):
        flags.append("too_small")
    if area_m2 > config.max_area_m2 or diameter_m > config.max_diameter_m:
        flags.append("too_large")
    if elongation > config.max_elongation:
        flags.append("elongated")
    if (
        config.broad_patch_min_area_m2 is not None
        and area_m2 >= config.broad_patch_min_area_m2
        and circularity <= config.broad_patch_max_circularity
        and ringness <= config.broad_patch_max_ringness
    ):
        flags.append("broad_patch")
    return flags


def _ring_evidence(data: FloatArray, mask: np.ndarray) -> tuple[float, float]:
    annulus = _nanmean(data[mask])
    center_mask = _center_mask(mask)
    background_mask = binary_dilation(mask, iterations=3) & ~mask
    center = _nanmean(data[center_mask])
    background = _nanmean(data[background_mask])
    contrast = annulus - background
    return float(contrast), ringness_score(annulus, center, background)


def _center_mask(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return np.zeros(mask.shape, dtype=bool)
    row_center = float(np.mean(rows))
    col_center = float(np.mean(cols))
    radius = max(float(np.sqrt(rows.size / np.pi)) * 0.45, 1.0)
    grid_rows, grid_cols = np.ogrid[: mask.shape[0], : mask.shape[1]]
    center_disk = (grid_rows - row_center) ** 2 + (grid_cols - col_center) ** 2 <= radius**2
    center = center_disk & ~mask
    return center if np.any(center) else center_disk & mask


def _nanmean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    return float(np.nanmean(finite))


def _majority_field_id(values: np.ndarray) -> str:
    labels, counts = np.unique(values.astype(str), return_counts=True)
    return str(labels[int(np.argmax(counts))])


def _season_count(dates: tuple[str, ...]) -> int:
    return len({date[:4] for date in dates if date})


def _unique_field_ids(field_ids: np.ndarray) -> tuple[object, ...]:
    ids: dict[str, object] = {}
    for raw_id in field_ids.ravel():
        if _is_missing_field_id(raw_id):
            continue
        ids.setdefault(
            _field_id_sort_key(raw_id), raw_id.item() if hasattr(raw_id, "item") else raw_id
        )
    return tuple(ids[key] for key in sorted(ids))


def _is_missing_field_id(field_id: object) -> bool:
    if field_id is None:
        return True
    if isinstance(field_id, str):
        return field_id == ""
    if isinstance(field_id, int):
        return field_id == 0
    if isinstance(field_id, np.integer):
        return bool(field_id == 0)
    if isinstance(field_id, float | np.floating):
        return bool(np.isnan(field_id) or field_id == 0.0)
    return False


def _field_id_sort_key(field_id: object) -> str:
    return f"{type(field_id).__name__}:{field_id}"


def _require_common_shapes(
    anomaly_layers: tuple[FieldAnomalyLayer, ...],
    field_ids: np.ndarray,
) -> None:
    for layer in anomaly_layers:
        if layer.data.shape != field_ids.shape:
            raise ValueError("Anomaly layers and field_ids must have matching shapes.")


def _empty_objects(crs: str) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "object_id": [],
            "field_id": [],
            "geometry": [],
        },
        geometry="geometry",
        crs=crs,
    )
