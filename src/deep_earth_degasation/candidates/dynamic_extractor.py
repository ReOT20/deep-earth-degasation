from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import geopandas as gpd
import numpy as np
from affine import Affine
from rasterio.features import shapes
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry
from skimage.measure import label, regionprops

from deep_earth_degasation.anomaly.field_normalization import FieldAnomalyLayer
from deep_earth_degasation.candidates.object_features import compute_object_zonal_stats
from deep_earth_degasation.context.fields import assign_field_context
from deep_earth_degasation.io.raster_stack import FloatArray
from deep_earth_degasation.morphology.shape import compute_shape_metrics


@dataclass(frozen=True)
class DynamicExtractionConfig:
    anomaly_percentile: float = 95.0
    min_area_m2: float = 1_000.0
    max_area_m2: float = 800_000.0
    min_diameter_m: float = 40.0
    max_diameter_m: float = 1_000.0
    max_elongation: float = 4.0
    merge_distance_m: float = 30.0
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
    extraction_config = config or DynamicExtractionConfig()
    if not anomaly_layers:
        return _empty_objects(crs)
    _require_common_shapes(anomaly_layers, field_ids)

    detections: list[dict[str, Any]] = []
    for layer in anomaly_layers:
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
                    layers=(layer,),
                    mask=component_mask,
                    config=extraction_config,
                    crs=crs,
                )
            )

    if not detections:
        return _empty_objects(crs)

    detections_gdf = gpd.GeoDataFrame(detections, geometry="geometry", crs=crs)
    merged_gdf = _merge_repeated_detections(detections_gdf, extraction_config)
    if fields is not None and not merged_gdf.empty:
        merged_gdf = assign_field_context(
            merged_gdf,
            fields,
            field_id_column=field_id_column,
        )
    return merged_gdf


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
        threshold = float(np.nanpercentile(values, config.anomaly_percentile))
        anomaly_mask = positive_support_mask & (data >= threshold)
        field_labels = label(anomaly_mask, connectivity=config.connectivity)
        for component_label in range(1, int(field_labels.max()) + 1):
            labels[field_labels == component_label] = next_label
            next_label += 1
    return labels


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
    config: DynamicExtractionConfig,
    crs: str,
) -> dict[str, Any]:
    metrics = compute_shape_metrics(geometry)
    stats = compute_object_zonal_stats(layers, mask)
    flags = _object_flags(metrics.area, metrics.equivalent_diameter, metrics.elongation, config)
    return {
        "object_id": object_id,
        "field_id": field_id,
        "geometry": geometry,
        "area_m2": metrics.area,
        "perimeter_m": metrics.perimeter,
        "equivalent_diameter_m": metrics.equivalent_diameter,
        "circularity": metrics.circularity,
        "elongation": metrics.elongation,
        "mean_anomaly": stats.mean_anomaly,
        "max_anomaly": stats.max_anomaly,
        "per_feature_mean": stats.per_feature_mean,
        "per_feature_max": stats.per_feature_max,
        "source_feature_names": stats.source_feature_names,
        "source_layer_ids": stats.source_layer_ids,
        "anomalous_dates": stats.anomalous_dates,
        "source_detection_count": 1,
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
    flags = _object_flags(metrics.area, metrics.equivalent_diameter, metrics.elongation, config)
    per_feature_mean = _merge_feature_stats(group, "per_feature_mean")
    per_feature_max = _merge_feature_stats(group, "per_feature_max", use_max=True)
    return {
        **group[0],
        "object_id": object_id,
        "geometry": geometry,
        "area_m2": metrics.area,
        "perimeter_m": metrics.perimeter,
        "equivalent_diameter_m": metrics.equivalent_diameter,
        "circularity": metrics.circularity,
        "elongation": metrics.elongation,
        "mean_anomaly": float(np.nanmean([row["mean_anomaly"] for row in group])),
        "max_anomaly": float(np.nanmax([row["max_anomaly"] for row in group])),
        "per_feature_mean": per_feature_mean,
        "per_feature_max": per_feature_max,
        "source_feature_names": tuple(
            sorted({name for row in group for name in row["source_feature_names"]})
        ),
        "source_layer_ids": tuple(
            sorted({layer_id for row in group for layer_id in row["source_layer_ids"]})
        ),
        "anomalous_dates": tuple(dates),
        "source_detection_count": len(group),
        "repeated_seasons": _season_count(tuple(dates)),
        "dynamic_object_flags": flags,
        "passes_dynamic_filters": not flags,
    }


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


def _object_flags(
    area_m2: float,
    diameter_m: float,
    elongation: float,
    config: DynamicExtractionConfig,
) -> list[str]:
    flags: list[str] = []
    if area_m2 < config.min_area_m2 or diameter_m < config.min_diameter_m:
        flags.append("too_small")
    if area_m2 > config.max_area_m2 or diameter_m > config.max_diameter_m:
        flags.append("too_large")
    if elongation > config.max_elongation:
        flags.append("elongated")
    return flags


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
    if isinstance(field_id, float | np.floating):
        return bool(np.isnan(field_id))
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
