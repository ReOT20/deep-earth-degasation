from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
from rasterio.features import rasterize
from shapely.geometry.base import BaseGeometry

from deep_earth_degasation.anomaly.composite import CompositeAnomalyMap, composite_anomaly_map
from deep_earth_degasation.anomaly.field_normalization import (
    FieldAnomalyLayer,
    field_normalized_anomaly,
)
from deep_earth_degasation.anomaly.residuals import (
    peer_residual_layers,
    residual_layers_metadata,
    temporal_residual_layers,
)
from deep_earth_degasation.candidates.dynamic_extractor import (
    DynamicExtractionConfig,
    extract_dynamic_objects_from_composite,
)
from deep_earth_degasation.config import MVPConfig, resolved_config_dict
from deep_earth_degasation.context.false_positive import (
    FalsePositiveContext,
    FalsePositiveFilterConfig,
    apply_false_positive_filters,
)
from deep_earth_degasation.context.landcover import assign_landcover_context
from deep_earth_degasation.features.sar import (
    sentinel1_event_response_features,
    sentinel1_features_from_stack,
)
from deep_earth_degasation.features.spectral import sentinel2_features_from_stack
from deep_earth_degasation.features.thermal import landsat_thermal_features_from_stack
from deep_earth_degasation.features.types import DynamicFeatureLayer, DynamicFeatureResult
from deep_earth_degasation.features.weather import (
    WeatherContext,
    load_weather_context_manifest,
    post_rain_drying_features,
)
from deep_earth_degasation.io.candidates import (
    candidate_object_score_rows,
    write_candidate_object_scores_csv,
    write_candidate_object_time_series,
    write_candidate_objects_geojson,
)
from deep_earth_degasation.io.labeling import write_labeling_table
from deep_earth_degasation.io.raster_stack import RasterLayer, RasterStack, load_raster_stack
from deep_earth_degasation.io.raster_stack import write_run_manifest as write_raster_run_manifest
from deep_earth_degasation.io.review_exports import (
    write_object_feature_table,
    write_rank_explanations,
)
from deep_earth_degasation.io.vector import VectorDataError, VectorLayer, load_vector_layer
from deep_earth_degasation.learning.dataset import write_learning_dataset
from deep_earth_degasation.pipeline.manifest import PreparedStackManifest
from deep_earth_degasation.reports.passport import write_candidate_passport
from deep_earth_degasation.reports.quicklook import write_quicklook_png
from deep_earth_degasation.scoring import score_candidate_objects
from deep_earth_degasation.validation.summary import (
    build_validation_summary,
    write_validation_summary,
)

GUARDRAIL_MESSAGE = (
    "Dynamic MVP artifacts are ranked candidate surface anomalies for expert and field "
    "review; they are not direct H2 detections or proof of active degassing."
)


@dataclass(frozen=True)
class DynamicMVPArtifactPaths:
    output_dir: Path
    candidates_geojson: Path
    candidate_scores_csv: Path
    object_feature_table_csv: Path
    rank_explanations_csv: Path
    labeling_table_csv: Path
    learning_dataset_csv: Path
    validation_summary_json: Path
    passports_dir: Path
    time_series_dir: Path
    quicklooks_dir: Path
    run_manifest_json: Path
    resolved_config_json: Path
    anomaly_map_npy: Path
    anomaly_map_metadata_json: Path
    residual_cube_npz: Path
    residual_metadata_json: Path


def run_dynamic_mvp(
    *,
    config: MVPConfig,
    manifest: PreparedStackManifest,
    output_dir: Path,
) -> DynamicMVPArtifactPaths:
    """Run the prepared-data dynamic MVP artifact pipeline."""
    output_dir.mkdir(parents=True, exist_ok=True)
    _validate_runtime_config(config)
    stack = load_raster_stack(manifest)
    reference_layer = _reference_layer(stack)
    vectors = _load_vectors(manifest, config)
    fields = _required_vector(vectors, "fields").data
    field_id_column = manifest.vectors["fields"].id_field or "stable_id"
    field_ids = _rasterize_fields(fields, reference_layer, field_id_column=field_id_column)
    features, feature_flags = _dynamic_features(stack, reference_layer, config, manifest)
    feature_flags = (*feature_flags, *_configured_missing_context_flags(config, vectors, manifest))
    anomaly_layers, anomaly_flags = _field_anomaly_layers(
        features,
        field_ids,
        config=config,
        reference_layer=reference_layer,
    )
    residual_layers, residual_flags = _hierarchical_residual_layers(
        features,
        anomaly_layers,
        field_ids,
        fields,
        config=config,
    )
    composite = composite_anomaly_map(anomaly_layers, config.anomaly_components)
    objects = extract_dynamic_objects_from_composite(
        composite,
        residual_layers,
        field_ids,
        transform=reference_layer.transform,
        crs=manifest.crs,
        config=_dynamic_extraction_config(config),
        fields=fields,
        field_id_column=field_id_column,
    )
    if not objects.empty:
        objects = apply_false_positive_filters(
            objects,
            _false_positive_context(vectors),
            _false_positive_filter_config(config),
        )
        objects = _append_landcover_context(objects, vectors, config)
        objects = _append_missing_flags(objects, (*feature_flags, *anomaly_flags, *residual_flags))
        objects = _append_dynamic_detector_flags(objects, config)
        objects = score_candidate_objects(
            objects,
            config.scoring,
            min_repeated_seasons=(
                config.dynamic_detector.min_repeated_seasons
                if config.dynamic_detector is not None
                else 2
            ),
        )

    paths = DynamicMVPArtifactPaths(
        output_dir=output_dir,
        candidates_geojson=output_dir / "candidates.geojson",
        candidate_scores_csv=output_dir / "candidate_scores.csv",
        object_feature_table_csv=output_dir / "object_feature_table.csv",
        rank_explanations_csv=output_dir / "rank_explanations.csv",
        labeling_table_csv=output_dir / "labeling_table.csv",
        learning_dataset_csv=output_dir / "learning_dataset.csv",
        validation_summary_json=output_dir / "validation_summary.json",
        passports_dir=output_dir / "passports",
        time_series_dir=output_dir / "time_series",
        quicklooks_dir=output_dir / "quicklooks",
        run_manifest_json=output_dir / "run_manifest.json",
        resolved_config_json=output_dir / "resolved_config.json",
        anomaly_map_npy=output_dir / "anomaly_maps" / "composite_anomaly.npy",
        anomaly_map_metadata_json=output_dir / "anomaly_maps" / "composite_anomaly.json",
        residual_cube_npz=output_dir / "anomaly_maps" / "hierarchical_residuals.npz",
        residual_metadata_json=output_dir / "anomaly_maps" / "hierarchical_residuals.json",
    )
    _write_artifacts(
        paths,
        objects,
        stack,
        config,
        composite,
        residual_layers=residual_layers,
        residual_flags=residual_flags,
        known_sites=_vector_data(vectors, "known_sites"),
    )
    return paths


def _reference_layer(stack: RasterStack) -> RasterLayer:
    for layer in stack.layers:
        if np.isclose(layer.spec.resolution_m, stack.manifest.resolution_m):
            return layer
    return stack.layers[0]


def _load_vectors(
    manifest: PreparedStackManifest,
    config: MVPConfig,
) -> dict[str, VectorLayer | None]:
    allow_reprojection = (
        config.prepared_data.allow_reprojection if config.prepared_data is not None else False
    )
    vectors: dict[str, VectorLayer | None] = {}
    for name, spec in manifest.vectors.items():
        if not spec.path.exists():
            if spec.required:
                load_vector_layer(
                    spec.path,
                    name=name,
                    role=spec.role,
                    id_field=spec.id_field,
                    target_crs=manifest.crs,
                    allow_reprojection=allow_reprojection,
                )
            vectors[name] = None
            continue
        try:
            vectors[name] = load_vector_layer(
                spec.path,
                name=name,
                role=spec.role,
                id_field=spec.id_field,
                target_crs=manifest.crs,
                allow_reprojection=allow_reprojection,
            )
        except VectorDataError as exc:
            if not spec.required and "contains no features" in str(exc):
                vectors[name] = None
                continue
            raise
    return vectors


def _required_vector(vectors: dict[str, VectorLayer | None], name: str) -> VectorLayer:
    layer = vectors.get(name)
    if layer is None:
        raise ValueError(f"Required vector layer {name!r} is missing.")
    return layer


def _rasterize_fields(
    fields: gpd.GeoDataFrame,
    reference_layer: RasterLayer,
    *,
    field_id_column: str,
) -> np.ndarray:
    shapes = [
        (geometry, index)
        for index, geometry in enumerate(fields.geometry, start=1)
        if geometry is not None and not geometry.is_empty
    ]
    labels = rasterize(
        shapes,
        out_shape=reference_layer.shape,
        transform=reference_layer.transform,
        fill=0,
        all_touched=True,
        dtype="int32",
    )
    if not np.any(labels):
        raise ValueError("Fields vector does not overlap the prepared raster grid.")
    return labels


def _dynamic_features(
    stack: RasterStack,
    reference_layer: RasterLayer,
    config: MVPConfig,
    manifest: PreparedStackManifest,
) -> tuple[tuple[DynamicFeatureLayer, ...], tuple[str, ...]]:
    features: list[DynamicFeatureLayer] = []
    flags: list[str] = []
    sentinel2_result = sentinel2_features_from_stack(stack)
    sentinel1_result = sentinel1_features_from_stack(stack)
    for result in (
        sentinel2_result,
        sentinel1_result,
        landsat_thermal_features_from_stack(stack),
    ):
        features.extend(_matching_grid_features(result, reference_layer, flags, config))
        flags.extend(result.missing_data_flags)

    weather_context = _load_weather_context(manifest)
    event_results = _event_conditioned_feature_results(
        tuple(features),
        weather_context,
        config,
    )
    for result in event_results:
        features.extend(_matching_grid_features(result, reference_layer, flags, config))
        flags.extend(result.missing_data_flags)
    flags.extend(_sar_reliability_flags(tuple(features), config))
    return tuple(features), tuple(sorted(set(flags)))


def _load_weather_context(manifest: PreparedStackManifest) -> WeatherContext | None:
    if manifest.weather_context is None:
        return None
    return load_weather_context_manifest(manifest.weather_context.path)


def _event_conditioned_feature_results(
    features: tuple[DynamicFeatureLayer, ...],
    weather_context: WeatherContext | None,
    config: MVPConfig,
) -> tuple[DynamicFeatureResult, ...]:
    results: list[DynamicFeatureResult] = []
    if _post_rain_context_configured(config):
        results.append(
            post_rain_drying_features(
                _moisture_features(features),
                weather_context,
            )
        )
    if _sar_event_response_configured(config):
        results.append(
            sentinel1_event_response_features(
                _sar_features(features),
                weather_context,
            )
        )
    return tuple(results)


def _moisture_features(
    features: tuple[DynamicFeatureLayer, ...],
) -> tuple[DynamicFeatureLayer, ...]:
    return tuple(feature for feature in features if feature.name in {"NDMI", "NDWI", "MSI"})


def _sar_features(
    features: tuple[DynamicFeatureLayer, ...],
) -> tuple[DynamicFeatureLayer, ...]:
    return tuple(feature for feature in features if feature.sensor.lower() == "sentinel1")


def _sar_reliability_flags(
    features: tuple[DynamicFeatureLayer, ...],
    config: MVPConfig,
) -> tuple[str, ...]:
    if not _sar_event_response_configured(config) and "sar" not in config.anomaly_components:
        return ()
    if not any(feature.sensor.lower() == "sentinel1" for feature in features):
        return ()
    vegetation_names = {"NDVI", "red_edge", "NDRE"}
    if any(feature.name in vegetation_names for feature in features):
        return ()
    return ("missing_crop_density_context_for_sar",)


def _matching_grid_features(
    result: DynamicFeatureResult,
    reference_layer: RasterLayer,
    flags: list[str],
    config: MVPConfig,
) -> list[DynamicFeatureLayer]:
    features: list[DynamicFeatureLayer] = []
    for feature in result.features:
        if feature.data.shape != reference_layer.shape:
            flags.append(f"skipped_grid_mismatch_{feature.name}_{feature.date or 'undated'}")
            continue
        if not _feature_in_configured_time_window(feature, config):
            flags.append(f"skipped_out_of_window_{feature.name}_{feature.date or 'undated'}")
            continue
        features.append(feature)
    return features


def _feature_in_configured_time_window(feature: DynamicFeatureLayer, config: MVPConfig) -> bool:
    if feature.date is None:
        return True
    month = int(feature.date[5:7])
    name = feature.name.lower()
    if any(token in name for token in ("ndvi", "ndre", "red_edge", "red-edge", "vegetation")):
        return month in config.time.vegetation_months
    if any(token in name for token in ("bsi", "brightness", "bare_soil", "bare-soil")):
        return month in config.time.bare_soil_months
    return True


def _field_anomaly_layers(
    features: tuple[DynamicFeatureLayer, ...],
    field_ids: np.ndarray,
    *,
    config: MVPConfig,
    reference_layer: RasterLayer,
) -> tuple[tuple[FieldAnomalyLayer, ...], tuple[str, ...]]:
    layers: list[FieldAnomalyLayer] = []
    flags: list[str] = []
    for feature in features:
        component = _component_for_feature(feature, config)
        if component is None:
            flags.append(f"missing_anomaly_component_for_{feature.name}")
            continue
        anomaly = field_normalized_anomaly(
            feature,
            field_ids,
            component=component,
            min_valid_pixels=(
                config.normalization.min_valid_pixels_per_field_date
                if config.normalization is not None
                and config.normalization.min_valid_pixels_per_field_date is not None
                else 1
            ),
            mad_epsilon=(
                config.normalization.mad_epsilon
                if config.normalization is not None and config.normalization.mad_epsilon is not None
                else 1.0e-6
            ),
        )
        if anomaly.data.shape != reference_layer.shape:
            flags.append(f"skipped_anomaly_grid_mismatch_{feature.name}")
            continue
        layers.append(anomaly)
        flags.extend(anomaly.missing_data_flags)
    if not layers:
        raise ValueError("No field-normalized anomaly layers could be produced.")
    return tuple(layers), tuple(sorted(set(flags)))


def _hierarchical_residual_layers(
    features: tuple[DynamicFeatureLayer, ...],
    field_layers: tuple[FieldAnomalyLayer, ...],
    field_ids: np.ndarray,
    fields: gpd.GeoDataFrame,
    *,
    config: MVPConfig,
) -> tuple[tuple[FieldAnomalyLayer, ...], tuple[str, ...]]:
    min_valid_pixels = (
        config.normalization.min_valid_pixels_per_field_date
        if config.normalization is not None
        and config.normalization.min_valid_pixels_per_field_date is not None
        else 1
    )
    mad_epsilon = (
        config.normalization.mad_epsilon
        if config.normalization is not None and config.normalization.mad_epsilon is not None
        else 1.0e-6
    )
    peer_result = peer_residual_layers(
        features,
        field_ids,
        fields,
        component_for_feature=lambda feature: _component_for_feature(feature, config),
        min_valid_pixels=min_valid_pixels,
        mad_epsilon=mad_epsilon,
    )
    temporal_result = temporal_residual_layers(
        features,
        component_for_feature=lambda feature: _component_for_feature(feature, config),
        mad_epsilon=mad_epsilon,
    )
    residual_layers = (*field_layers, *peer_result.layers, *temporal_result.layers)
    flags = tuple(
        sorted(set(peer_result.missing_data_flags) | set(temporal_result.missing_data_flags))
    )
    return residual_layers, flags


def _component_for_feature(feature: DynamicFeatureLayer, config: MVPConfig) -> str | None:
    for component_name, component in config.anomaly_components.items():
        if feature.name in component.inputs:
            return component_name
        if component_name == "post_rain_drying" and feature.name.startswith("post_rain_drying"):
            return component_name
        if component_name == "sar" and feature.name.startswith("sar_event_response"):
            return component_name
    return None


def _dynamic_extraction_config(config: MVPConfig) -> DynamicExtractionConfig:
    dynamic = config.dynamic_detector
    constraints = config.object_constraints
    return DynamicExtractionConfig(
        anomaly_percentile=dynamic.anomaly_percentile if dynamic is not None else 95.0,
        support_percentile=dynamic.support_percentile if dynamic is not None else None,
        min_support_pixels=dynamic.min_support_pixels if dynamic is not None else 1,
        min_area_m2=constraints.min_area_ha * 10_000.0,
        max_area_m2=constraints.max_area_ha * 10_000.0,
        min_diameter_m=constraints.min_diameter_m,
        max_diameter_m=constraints.max_diameter_m,
        merge_distance_m=(
            dynamic.merge_distance_m
            if dynamic is not None and dynamic.merge_distance_m is not None
            else 30.0
        ),
        merge_across_dates=(
            dynamic.merge_across_dates
            if dynamic is not None and dynamic.merge_across_dates is not None
            else True
        ),
        min_distance_to_field_edge_m=constraints.min_distance_to_field_edge_m,
    )


def _false_positive_filter_config(config: MVPConfig) -> FalsePositiveFilterConfig:
    filters = config.false_positive_filters
    field_edge_buffer_m = (
        filters.field_edge_buffer_m
        if filters is not None and filters.field_edge_buffer_m is not None
        else config.object_constraints.min_distance_to_field_edge_m
    )
    if filters is None:
        return FalsePositiveFilterConfig(
            field_edge_buffer_m=field_edge_buffer_m,
            penalties=_false_positive_penalties(config),
        )
    return FalsePositiveFilterConfig(
        flag_roads=filters.flag_roads,
        flag_water=filters.flag_water,
        flag_built_up=filters.flag_built_up
        if filters.flag_built_up is not None
        else bool(filters.flag_builtup if filters.flag_builtup is not None else True),
        flag_excluded_zones=filters.flag_excluded_zones,
        flag_quarries=filters.flag_quarries,
        flag_woody_patches=filters.flag_woody_patches,
        flag_field_edges=filters.flag_field_edges,
        flag_linear_objects=filters.flag_linear_objects,
        flag_cloud_shadows=filters.flag_cloud_shadows,
        flag_harvest_patterns=filters.flag_harvest_patterns,
        flag_irrigation=bool(
            filters.flag_irrigation if filters.flag_irrigation is not None else True
        ),
        road_buffer_m=filters.road_buffer_m or 20.0,
        water_buffer_m=filters.water_buffer_m or 20.0,
        builtup_buffer_m=filters.builtup_buffer_m or 50.0,
        woody_patch_buffer_m=filters.woody_patch_buffer_m or 20.0,
        field_edge_buffer_m=field_edge_buffer_m,
        max_elongation_without_penalty=filters.max_elongation_without_penalty or 4.0,
        penalties=_false_positive_penalties(config),
    )


def _false_positive_penalties(config: MVPConfig) -> dict[str, float]:
    default_penalties = FalsePositiveFilterConfig().penalties
    filters = config.false_positive_filters
    if filters is not None and not filters.use_penalties:
        return {key: 0.0 for key in default_penalties}

    penalties = dict(default_penalties)
    configured = config.scoring.penalties
    if configured is None:
        return penalties
    for key, value in configured.model_dump().items():
        if value is not None:
            penalties[key] = float(value)
    return penalties


def _false_positive_context(vectors: dict[str, VectorLayer | None]) -> FalsePositiveContext:
    return FalsePositiveContext(
        roads=_vector_data(vectors, "roads"),
        water=_vector_data(vectors, "water"),
        built_up=_vector_data(vectors, "built_up"),
        excluded_zones=_vector_data(vectors, "excluded_zones"),
        quarries=_vector_data(vectors, "quarries"),
        woody_patches=_vector_data(vectors, "woody_patches"),
        cloud_shadows=_vector_data(vectors, "cloud_shadows"),
        harvest_patterns=_vector_data(vectors, "harvest_patterns"),
        irrigation=_vector_data(vectors, "irrigation"),
    )


def _vector_data(vectors: dict[str, VectorLayer | None], name: str) -> gpd.GeoDataFrame | None:
    layer = vectors.get(name)
    return None if layer is None else layer.data


def _validate_runtime_config(config: MVPConfig) -> None:
    dynamic = config.dynamic_detector
    if dynamic is not None and dynamic.connected_components is False:
        raise ValueError(
            "dynamic_detector.connected_components=false is not supported by run-mvp; "
            "set it to true or defer the field."
        )


def _configured_missing_context_flags(
    config: MVPConfig,
    vectors: dict[str, VectorLayer | None],
    manifest: PreparedStackManifest,
) -> tuple[str, ...]:
    flags: set[str] = set()
    if _post_rain_context_configured(config) and manifest.weather_context is None:
        flags.add("missing_weather_context")
    if "geology_context" in config.anomaly_components and _vector_data(vectors, "geology") is None:
        flags.add("missing_geology_context")
    return tuple(sorted(flags))


def _post_rain_context_configured(config: MVPConfig) -> bool:
    if "post_rain_drying" in config.anomaly_components:
        return True
    if (
        config.scoring.cropland.post_rain_weight is not None
        and config.scoring.cropland.post_rain_weight > 0
    ):
        return True
    return bool(
        config.features is not None
        and config.features.weather is not None
        and config.features.weather.features
    )


def _sar_event_response_configured(config: MVPConfig) -> bool:
    sar_component = config.anomaly_components.get("sar")
    if sar_component is not None and any(
        "event_response" in input_name for input_name in sar_component.inputs
    ):
        return True
    return bool(
        config.features is not None
        and config.features.sentinel1 is not None
        and any("event_response" in name for name in config.features.sentinel1.features)
    )


def _append_landcover_context(
    objects: gpd.GeoDataFrame,
    vectors: dict[str, VectorLayer | None],
    config: MVPConfig,
) -> gpd.GeoDataFrame:
    landcover = _vector_data(vectors, "landcover")
    if landcover is not None:
        output = assign_landcover_context(
            objects,
            landcover,
            classes=config.land_cover.classes,
            mixed_candidate_min_classes=config.land_cover.mixed_candidate_min_classes,
        )
        return _append_row_context_flags(output, "landcover_context_flags")

    primary_branch = (config.land_cover.primary_branch or "").lower()
    if primary_branch == "cropland":
        output = objects.copy()
        output["landcover_branch"] = "cropland"
        output["dominant_landcover_branch"] = "cropland"
        output["landcover_proportions"] = [{"cropland": 1.0} for _ in range(len(output))]
        output["landcover_context_flags"] = [
            ["landcover_context_assumed_cropland_fields"] for _ in range(len(output))
        ]
        return _append_row_context_flags(output, "landcover_context_flags")

    output = objects.copy()
    output["landcover_branch"] = "unknown"
    output["dominant_landcover_branch"] = "unknown"
    output["landcover_proportions"] = [{} for _ in range(len(output))]
    output["landcover_context_flags"] = [["missing_landcover_context"] for _ in range(len(output))]
    return _append_row_context_flags(output, "landcover_context_flags")


def _append_row_context_flags(gdf: gpd.GeoDataFrame, column: str) -> gpd.GeoDataFrame:
    output = gdf.copy()
    missing_data_flags: list[tuple[str, ...]] = []
    for _, row in output.iterrows():
        existing = (
            _list_value(row["missing_data_flags"]) if "missing_data_flags" in row.index else ()
        )
        context_flags = _list_value(row[column]) if column in row.index else ()
        missing_data_flags.append(tuple(sorted(set(existing) | set(context_flags))))
    output["missing_data_flags"] = missing_data_flags
    return output


def _append_missing_flags(objects: gpd.GeoDataFrame, flags: tuple[str, ...]) -> gpd.GeoDataFrame:
    if not flags:
        return objects
    output = objects.copy()
    sorted_flags = tuple(sorted(set(flags)))
    existing_flags = (
        output["missing_data_flags"]
        if "missing_data_flags" in output
        else [() for _ in range(len(output))]
    )
    output["missing_data_flags"] = [
        tuple(sorted(set(_list_value(value)) | set(sorted_flags))) for value in existing_flags
    ]
    return output


def _append_dynamic_detector_flags(
    objects: gpd.GeoDataFrame,
    config: MVPConfig,
) -> gpd.GeoDataFrame:
    dynamic = config.dynamic_detector
    if dynamic is None:
        return objects

    output = objects.copy()
    missing_data_flags: list[tuple[str, ...]] = []
    for _, row in output.iterrows():
        existing_flags = (
            _list_value(row["missing_data_flags"]) if "missing_data_flags" in row.index else ()
        )
        missing_data_flags.append(
            tuple(
                sorted(
                    set(existing_flags)
                    | _dynamic_detector_missing_flags(
                        row, dynamic.min_valid_observations_per_season
                    )
                )
            )
        )
    output["missing_data_flags"] = missing_data_flags
    return output


def _dynamic_detector_missing_flags(row: object, min_valid_observations: int) -> set[str]:
    source_count = _numeric_row_value(row, "source_detection_count")
    if source_count is None or source_count >= min_valid_observations:
        return set()
    return {"below_min_valid_observations_per_season"}


def _write_artifacts(
    paths: DynamicMVPArtifactPaths,
    objects: gpd.GeoDataFrame,
    stack: RasterStack,
    config: MVPConfig,
    composite: CompositeAnomalyMap,
    *,
    residual_layers: tuple[FieldAnomalyLayer, ...],
    residual_flags: tuple[str, ...],
    known_sites: gpd.GeoDataFrame | None,
) -> None:
    paths.anomaly_map_npy.parent.mkdir(parents=True, exist_ok=True)
    _write_residual_products(paths, residual_layers, residual_flags)
    output_config = config.outputs
    review_limit = max(0, output_config.candidates_top_n)
    passports_dir = paths.passports_dir if output_config.generate_passports else None
    score_rows = candidate_object_score_rows(
        objects,
        passports_dir=passports_dir,
        passport_path_limit=review_limit,
    )
    review_score_rows = candidate_object_score_rows(
        objects,
        passports_dir=passports_dir,
        limit=review_limit,
        passport_path_limit=review_limit,
    )

    if output_config.export_geojson:
        write_candidate_objects_geojson(
            objects,
            paths.candidates_geojson,
            passports_dir=passports_dir,
            passport_path_limit=review_limit,
        )
    else:
        _unlink_if_exists(paths.candidates_geojson)

    if output_config.export_csv:
        write_candidate_object_scores_csv(
            objects,
            paths.candidate_scores_csv,
            passports_dir=passports_dir,
            passport_path_limit=review_limit,
        )
        write_object_feature_table(score_rows, paths.object_feature_table_csv)
        write_rank_explanations(score_rows, paths.rank_explanations_csv)
    else:
        _unlink_if_exists(paths.candidate_scores_csv)
        _unlink_if_exists(paths.object_feature_table_csv)
        _unlink_if_exists(paths.rank_explanations_csv)

    write_labeling_table(review_score_rows, paths.labeling_table_csv)
    write_learning_dataset(
        score_rows,
        paths.learning_dataset_csv,
        run_id=stack.manifest.run.id,
        feature_snapshot_id=_feature_snapshot_id(stack.manifest),
        aoi_name=config.aoi.name,
        geometry_ref=str(paths.candidates_geojson) if output_config.export_geojson else "",
    )
    validation_config = config.validation
    if output_config.export_validation_summary is not False:
        validation_summary = build_validation_summary(
            score_rows=score_rows,
            candidates=objects,
            known_sites=known_sites,
            top_n=validation_config.top_n if validation_config is not None else 20,
            known_site_recall_top_n=(
                tuple(validation_config.known_site_recall_top_n)
                if validation_config is not None
                else ()
            ),
            expert_precision_top_n=(
                validation_config.expert_precision_top_n if validation_config is not None else None
            ),
            guardrail=GUARDRAIL_MESSAGE,
        )
        write_validation_summary(validation_summary, paths.validation_summary_json)
    else:
        _unlink_if_exists(paths.validation_summary_json)

    if output_config.generate_passports:
        paths.passports_dir.mkdir(parents=True, exist_ok=True)
        _clear_generated_files(paths.passports_dir, "dynamic-*.md")
        for score_row in review_score_rows:
            passport_path = Path(str(score_row["passport_path"]))
            write_candidate_passport(score_row, passport_path)
    elif paths.passports_dir.exists():
        _clear_generated_files(paths.passports_dir, "dynamic-*.md")

    if output_config.export_time_series is not False:
        _clear_generated_files(paths.time_series_dir, "dynamic-*.csv")
        write_candidate_object_time_series(objects, paths.time_series_dir, limit=review_limit)
    elif paths.time_series_dir.exists():
        _clear_generated_files(paths.time_series_dir, "dynamic-*.csv")

    if output_config.generate_quicklooks is True:
        _clear_generated_files(paths.quicklooks_dir, "dynamic-*.png")
        _write_quicklooks(
            objects,
            composite,
            paths.quicklooks_dir,
            transform=_reference_layer(stack).transform,
            limit=review_limit,
        )
    elif paths.quicklooks_dir.exists():
        _clear_generated_files(paths.quicklooks_dir, "dynamic-*.png")

    write_raster_run_manifest(stack, paths.run_manifest_json)
    if output_config.export_resolved_config:
        paths.resolved_config_json.write_text(
            json.dumps(resolved_config_dict(config), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    else:
        _unlink_if_exists(paths.resolved_config_json)
    np.save(paths.anomaly_map_npy, composite.data)
    paths.anomaly_map_metadata_json.write_text(
        json.dumps(
            {
                "component_weights": composite.component_weights,
                "missing_data_flags": list(composite.missing_data_flags),
                "source_feature_names": list(composite.source_feature_names),
                "source_layer_ids": list(composite.source_layer_ids),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_residual_products(
    paths: DynamicMVPArtifactPaths,
    residual_layers: tuple[FieldAnomalyLayer, ...],
    residual_flags: tuple[str, ...],
) -> None:
    arrays: dict[str, Any] = {
        f"{index:03d}_{_safe_layer_name(layer.name)}_{layer.residual_type}": layer.data
        for index, layer in enumerate(residual_layers, start=1)
    }
    np.savez_compressed(paths.residual_cube_npz, **arrays)
    paths.residual_metadata_json.write_text(
        json.dumps(
            residual_layers_metadata(residual_layers, missing_data_flags=residual_flags),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _safe_layer_name(name: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in name)


def _clear_generated_files(directory: Path, pattern: str) -> None:
    if not directory.exists():
        return
    for path in directory.glob(pattern):
        if path.is_file():
            path.unlink()


def _unlink_if_exists(path: Path) -> None:
    if path.exists() and path.is_file():
        path.unlink()


def _write_quicklooks(
    objects: gpd.GeoDataFrame,
    composite: CompositeAnomalyMap,
    output_dir: Path,
    *,
    transform: Any,
    limit: int,
) -> None:
    for score_row in candidate_object_score_rows(objects, limit=limit):
        candidate_id = str(score_row["candidate_id"])
        rank = int(score_row["rank"])
        geometry = _candidate_geometry_by_id(objects, candidate_id)
        write_quicklook_png(
            composite.data,
            output_dir / f"dynamic-object-{rank:06d}.png",
            candidate_geometry=geometry,
            transform=transform,
            title=candidate_id,
        )


def _candidate_geometry_by_id(
    objects: gpd.GeoDataFrame,
    candidate_id: str,
) -> BaseGeometry | None:
    for _, row in objects.iterrows():
        row_id = row["candidate_id"] if "candidate_id" in row.index else row.get("object_id", "")
        geometry = row.geometry
        if str(row_id) == candidate_id and isinstance(geometry, BaseGeometry):
            return geometry
    return None


def _feature_snapshot_id(manifest: PreparedStackManifest) -> str:
    digest = hashlib.sha256(manifest.path.read_bytes()).hexdigest()[:16]
    return f"prepared_manifest:{manifest.run.id}:{digest}"


def _read_score_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _list_value(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list | tuple | set):
        return tuple(str(item) for item in value)
    return (str(value),)


def _numeric_row_value(row: Any, key: str) -> float | None:
    if hasattr(row, "index") and key in row.index:
        value = row[key]
        if isinstance(value, int | float) and not isinstance(value, bool):
            return float(value)
    return None
