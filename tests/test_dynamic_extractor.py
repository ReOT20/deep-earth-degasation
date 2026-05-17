from __future__ import annotations

import geopandas as gpd
import numpy as np
from affine import Affine
from shapely.geometry import box

from deep_earth_degasation.anomaly.composite import CompositeAnomalyMap
from deep_earth_degasation.anomaly.field_normalization import FieldAnomalyLayer
from deep_earth_degasation.candidates.dynamic_extractor import (
    DynamicExtractionConfig,
    assign_stable_candidate_ids,
    extract_dynamic_objects,
    extract_dynamic_objects_from_composite,
)

CRS = "EPSG:32637"
TRANSFORM = Affine.translation(0, 100) * Affine.scale(10, -10)


def test_circular_anomaly_extracts_one_dynamic_object_with_stats() -> None:
    data = _blank()
    _draw_disk(data, center=(5, 5), radius=2, value=10.0)
    fields = np.ones(data.shape, dtype=int)

    objects = extract_dynamic_objects(
        (_layer("NDMI", data, date="2024-05-01"),),
        fields,
        transform=TRANSFORM,
        crs=CRS,
        config=_config(),
    )

    assert len(objects) == 1
    row = objects.iloc[0]
    assert row["field_id"] == "1"
    assert bool(row["passes_dynamic_filters"]) is True
    assert row["source_feature_names"] == ("NDMI",)
    assert row["per_feature_max"]["NDMI"] == 10.0
    assert row["max_anomaly"] == 10.0
    assert row["support_pixel_count"] == 13
    assert row["circularity"] > 0.35
    assert row["elongation"] < 1.5


def test_support_threshold_grows_candidate_around_seed_pixel() -> None:
    data = _blank()
    data[4:7, 4:7] = 4.0
    data[5, 5] = 10.0
    fields = np.ones(data.shape, dtype=int)

    objects = extract_dynamic_objects(
        (_layer("NDMI", data),),
        fields,
        transform=TRANSFORM,
        crs=CRS,
        config=DynamicExtractionConfig(
            anomaly_percentile=99.0,
            support_percentile=80.0,
            min_support_pixels=5,
            min_area_m2=100.0,
            min_diameter_m=5.0,
            max_area_m2=50_000.0,
            max_diameter_m=300.0,
            merge_distance_m=20.0,
        ),
    )

    assert len(objects) == 1
    assert objects["support_pixel_count"].iloc[0] == 9
    assert objects.geometry.iloc[0].area == 900.0


def test_min_support_pixels_filters_single_pixel_detections() -> None:
    data = _blank()
    data[5, 5] = 10.0
    fields = np.ones(data.shape, dtype=int)

    objects = extract_dynamic_objects(
        (_layer("NDMI", data),),
        fields,
        transform=TRANSFORM,
        crs=CRS,
        config=DynamicExtractionConfig(
            anomaly_percentile=99.0,
            min_support_pixels=2,
            min_area_m2=1.0,
            min_diameter_m=1.0,
            max_area_m2=50_000.0,
            max_diameter_m=300.0,
            merge_distance_m=20.0,
        ),
    )

    assert objects.empty


def test_composite_blob_becomes_one_candidate_with_per_feature_stats() -> None:
    moisture = _blank()
    vegetation = _blank()
    _draw_disk(moisture, center=(5, 5), radius=2, value=10.0)
    _draw_disk(vegetation, center=(5, 5), radius=2, value=8.0)
    composite = _composite((moisture + vegetation) / 2.0)
    fields = np.ones(moisture.shape, dtype=int)

    objects = extract_dynamic_objects_from_composite(
        composite,
        (
            _layer("NDMI", moisture, component="moisture", layer_id="ndmi"),
            _layer("NDVI", vegetation, component="vegetation", layer_id="ndvi"),
        ),
        fields,
        transform=TRANSFORM,
        crs=CRS,
        config=_config(min_area_m2=100.0, min_diameter_m=5.0),
    )

    assert len(objects) == 1
    row = objects.iloc[0]
    assert row["source_feature_names"] == ("NDMI", "NDVI")
    assert row["source_layer_ids"] == ("ndmi", "ndvi")
    assert row["per_feature_max"]["NDMI"] == 10.0
    assert row["per_feature_max"]["NDVI"] == 8.0


def test_composite_stats_aggregate_repeated_feature_dates_without_overwrite() -> None:
    early_ndmi = _blank()
    later_ndmi = _blank()
    vegetation = _blank()
    _draw_disk(early_ndmi, center=(5, 5), radius=2, value=10.0)
    _draw_disk(vegetation, center=(5, 5), radius=2, value=8.0)
    composite = _composite((early_ndmi + vegetation) / 2.0)
    fields = np.ones(early_ndmi.shape, dtype=int)

    objects = extract_dynamic_objects_from_composite(
        composite,
        (
            _layer(
                "NDMI", early_ndmi, component="moisture", date="2024-05-15", layer_id="ndmi_2024"
            ),
            _layer(
                "NDMI", later_ndmi, component="moisture", date="2025-05-15", layer_id="ndmi_2025"
            ),
            _layer(
                "NDVI", vegetation, component="vegetation", date="2024-05-15", layer_id="ndvi_2024"
            ),
        ),
        fields,
        transform=TRANSFORM,
        crs=CRS,
        config=_config(min_area_m2=100.0, min_diameter_m=5.0),
    )

    assert len(objects) == 1
    row = objects.iloc[0]
    assert row["per_feature_max"]["NDMI"] == 10.0
    assert row["per_feature_max"]["NDVI"] == 8.0
    assert row["anomalous_dates"] == ("2024-05-15",)
    assert row["source_detection_count"] == 2
    assert row["repeated_seasons"] == 1


def test_single_feature_noise_does_not_flood_composite_inventory() -> None:
    moisture = _blank(width=16)
    vegetation = _blank(width=16)
    moisture[5, 3] = 12.0
    vegetation[5, 12] = 12.0
    composite_data = _blank(width=16)
    _draw_disk(composite_data, center=(5, 8), radius=1, value=8.0)
    fields = np.ones(moisture.shape, dtype=int)

    per_feature_objects = extract_dynamic_objects(
        (
            _layer("NDMI", moisture, component="moisture", layer_id="ndmi"),
            _layer("NDVI", vegetation, component="vegetation", layer_id="ndvi"),
        ),
        fields,
        transform=TRANSFORM,
        crs=CRS,
        config=_config(min_area_m2=50.0, min_diameter_m=5.0),
    )
    composite_objects = extract_dynamic_objects_from_composite(
        _composite(composite_data),
        (
            _layer("NDMI", moisture, component="moisture", layer_id="ndmi"),
            _layer("NDVI", vegetation, component="vegetation", layer_id="ndvi"),
        ),
        fields,
        transform=TRANSFORM,
        crs=CRS,
        config=_config(min_area_m2=50.0, min_diameter_m=5.0),
    )

    assert len(per_feature_objects) == 2
    assert len(composite_objects) == 1
    assert composite_objects["source_feature_names"].iloc[0] == ("NDMI", "NDVI")


def test_composite_nearby_blobs_remain_separate_without_repeated_detection_merge() -> None:
    data = _blank(width=14)
    _draw_disk(data, center=(5, 4), radius=1, value=10.0)
    _draw_disk(data, center=(5, 8), radius=1, value=10.0)
    fields = np.ones(data.shape, dtype=int)

    objects = extract_dynamic_objects_from_composite(
        _composite(data),
        (_layer("NDMI", data),),
        fields,
        transform=TRANSFORM,
        crs=CRS,
        config=DynamicExtractionConfig(
            anomaly_percentile=90.0,
            min_area_m2=50.0,
            min_diameter_m=5.0,
            max_area_m2=50_000.0,
            max_diameter_m=300.0,
            merge_distance_m=100.0,
            merge_across_dates=True,
        ),
    )

    assert len(objects) == 2


def test_partial_ring_has_annulus_evidence() -> None:
    data = _blank(height=15, width=15)
    rows, cols = np.ogrid[:15, :15]
    distance = np.sqrt((rows - 7) ** 2 + (cols - 7) ** 2)
    data[(distance >= 3) & (distance <= 4)] = 10.0
    fields = np.ones(data.shape, dtype=int)

    objects = extract_dynamic_objects_from_composite(
        _composite(data),
        (_layer("NDMI", data),),
        fields,
        transform=TRANSFORM,
        crs=CRS,
        config=_config(min_area_m2=100.0, min_diameter_m=5.0),
    )

    assert len(objects) == 1
    assert objects["annulus_contrast"].iloc[0] > 0
    assert objects["ringness_score"].iloc[0] > 0


def test_broad_field_patch_is_flagged() -> None:
    data = _blank(height=20, width=20)
    data[5:15, 2:18] = 10.0
    fields = np.ones(data.shape, dtype=int)

    objects = extract_dynamic_objects_from_composite(
        _composite(data),
        (_layer("BSI", data, component="soil_brightness"),),
        fields,
        transform=TRANSFORM,
        crs=CRS,
        config=DynamicExtractionConfig(
            anomaly_percentile=90.0,
            min_area_m2=100.0,
            max_area_m2=50_000.0,
            min_diameter_m=5.0,
            max_diameter_m=300.0,
            max_elongation=4.0,
            broad_patch_min_area_m2=5_000.0,
            merge_distance_m=20.0,
        ),
    )

    assert len(objects) == 1
    assert "broad_patch" in objects["dynamic_object_flags"].iloc[0]


def test_components_do_not_merge_across_field_boundaries() -> None:
    data = _blank(width=12)
    data[4:7, 4:8] = 10.0
    fields = np.array([[1] * 6 + [2] * 6 for _ in range(10)])

    objects = extract_dynamic_objects(
        (_layer("NDMI", data),),
        fields,
        transform=TRANSFORM,
        crs=CRS,
        config=_config(min_area_m2=100.0, min_diameter_m=5.0),
    )

    assert len(objects) == 2
    assert set(objects["field_id"]) == {"1", "2"}


def test_zero_label_background_is_not_extracted_as_dynamic_object() -> None:
    data = _blank()
    data[1:4, 1:4] = 12.0
    data[6:8, 6:8] = 10.0
    field_ids = np.zeros(data.shape, dtype=int)
    field_ids[6:8, 6:8] = 1

    objects = extract_dynamic_objects(
        (_layer("NDMI", data),),
        field_ids,
        transform=TRANSFORM,
        crs=CRS,
        config=_config(min_area_m2=100.0, min_diameter_m=5.0),
    )

    assert len(objects) == 1
    assert objects["field_id"].iloc[0] == "1"
    assert objects.geometry.iloc[0].bounds == (60.0, 20.0, 80.0, 40.0)


def test_field_context_is_assigned_when_field_polygons_are_supplied() -> None:
    data = _blank()
    _draw_disk(data, center=(5, 5), radius=2, value=10.0)
    field_ids = np.ones(data.shape, dtype=int)
    fields = gpd.GeoDataFrame(
        {"field_id": ["field-a"], "geometry": [box(0, 0, 100, 100)]},
        crs=CRS,
    )

    objects = extract_dynamic_objects(
        (_layer("NDMI", data),),
        field_ids,
        transform=TRANSFORM,
        crs=CRS,
        config=_config(),
        fields=fields,
    )

    assert objects["field_id"].iloc[0] == "field-a"
    assert objects["distance_to_field_edge_m"].iloc[0] is not None
    assert objects["field_context_flags"].iloc[0] == []


def test_field_context_honors_custom_field_id_column() -> None:
    data = _blank()
    _draw_disk(data, center=(5, 5), radius=2, value=10.0)
    field_ids = np.ones(data.shape, dtype=int)
    fields = gpd.GeoDataFrame(
        {"plot_code": ["plot-a"], "geometry": [box(0, 0, 100, 100)]},
        crs=CRS,
    )

    objects = extract_dynamic_objects(
        (_layer("NDMI", data),),
        field_ids,
        transform=TRANSFORM,
        crs=CRS,
        config=_config(),
        fields=fields,
        field_id_column="plot_code",
    )

    assert objects["field_id"].iloc[0] == "plot-a"
    assert objects["candidate_id"].iloc[0].startswith("dyn-plot-a-")


def test_stable_ids_are_order_independent() -> None:
    data = _blank(width=14)
    _draw_disk(data, center=(5, 3), radius=1, value=10.0)
    _draw_disk(data, center=(5, 10), radius=1, value=9.0)
    fields = np.ones(data.shape, dtype=int)

    objects = extract_dynamic_objects(
        (_layer("NDMI", data),),
        fields,
        transform=TRANSFORM,
        crs=CRS,
        config=_config(min_area_m2=50.0, min_diameter_m=5.0),
    )
    reordered = assign_stable_candidate_ids(objects.iloc[::-1].reset_index(drop=True))

    assert set(reordered["candidate_id"]) == set(objects["candidate_id"])
    assert all(candidate_id.startswith("dyn-1-") for candidate_id in objects["candidate_id"])


def test_stable_ids_change_for_material_geometry_change() -> None:
    first = _blank()
    second = _blank()
    _draw_disk(first, center=(5, 5), radius=1, value=10.0)
    _draw_disk(second, center=(5, 6), radius=1, value=10.0)
    fields = np.ones(first.shape, dtype=int)

    first_objects = extract_dynamic_objects(
        (_layer("NDMI", first),),
        fields,
        transform=TRANSFORM,
        crs=CRS,
        config=_config(min_area_m2=50.0, min_diameter_m=5.0),
    )
    second_objects = extract_dynamic_objects(
        (_layer("NDMI", second),),
        fields,
        transform=TRANSFORM,
        crs=CRS,
        config=_config(min_area_m2=50.0, min_diameter_m=5.0),
    )

    assert first_objects["candidate_id"].iloc[0] != second_objects["candidate_id"].iloc[0]


def test_tiny_and_huge_components_are_flagged() -> None:
    tiny = _blank()
    tiny[5, 5] = 10.0
    huge = np.full((10, 10), 10.0, dtype=float)
    fields = np.ones(tiny.shape, dtype=int)

    tiny_objects = extract_dynamic_objects(
        (_layer("NDMI", tiny),),
        fields,
        transform=TRANSFORM,
        crs=CRS,
        config=_config(min_area_m2=200.0, min_diameter_m=20.0),
    )
    huge_objects = extract_dynamic_objects(
        (_layer("NDMI", huge),),
        fields,
        transform=TRANSFORM,
        crs=CRS,
        config=_config(max_area_m2=500.0, max_diameter_m=30.0),
    )

    assert tiny_objects["dynamic_object_flags"].iloc[0] == ["too_small"]
    assert bool(tiny_objects["passes_dynamic_filters"].iloc[0]) is False
    assert "too_large" in huge_objects["dynamic_object_flags"].iloc[0]


def test_elongated_stripe_is_flagged() -> None:
    data = _blank(width=14)
    data[5, 2:12] = 10.0
    fields = np.ones(data.shape, dtype=int)

    objects = extract_dynamic_objects(
        (_layer("VV_VH_ratio", data, component="sar"),),
        fields,
        transform=TRANSFORM,
        crs=CRS,
        config=_config(min_area_m2=50.0, min_diameter_m=5.0, max_elongation=4.0),
    )

    assert len(objects) == 1
    assert "elongated" in objects["dynamic_object_flags"].iloc[0]
    assert bool(objects["passes_dynamic_filters"].iloc[0]) is False


def test_repeated_overlapping_detections_merge_with_persistence_metrics() -> None:
    first = _blank()
    second = _blank()
    _draw_disk(first, center=(5, 5), radius=2, value=10.0)
    _draw_disk(second, center=(5, 6), radius=2, value=9.0)
    fields = np.ones(first.shape, dtype=int)

    objects = extract_dynamic_objects(
        (
            _layer("NDMI", first, date="2024-05-01", layer_id="ndmi_1"),
            _layer("NDMI", second, date="2025-05-01", layer_id="ndmi_2"),
        ),
        fields,
        transform=TRANSFORM,
        crs=CRS,
        config=_config(),
    )

    assert len(objects) == 1
    row = objects.iloc[0]
    assert row["source_detection_count"] == 2
    assert row["anomalous_dates"] == ("2024-05-01", "2025-05-01")
    assert row["repeated_seasons"] == 2
    assert row["source_layer_ids"] == ("ndmi_1", "ndmi_2")


def test_merged_repeated_detections_recompute_filters_when_small_parts_become_valid() -> None:
    first = _blank()
    second = _blank()
    first[5, 5] = 10.0
    second[5, 6] = 9.0
    fields = np.ones(first.shape, dtype=int)

    objects = extract_dynamic_objects(
        (
            _layer("NDMI", first, date="2024-05-01", layer_id="ndmi_1"),
            _layer("NDMI", second, date="2025-05-01", layer_id="ndmi_2"),
        ),
        fields,
        transform=TRANSFORM,
        crs=CRS,
        config=_config(min_area_m2=150.0, min_diameter_m=5.0),
    )

    assert len(objects) == 1
    assert objects["source_detection_count"].iloc[0] == 2
    assert objects["dynamic_object_flags"].iloc[0] == []
    assert bool(objects["passes_dynamic_filters"].iloc[0]) is True


def test_merged_repeated_detections_recompute_filters_when_union_becomes_too_large() -> None:
    first = _blank()
    second = _blank()
    _draw_disk(first, center=(5, 4), radius=1, value=10.0)
    _draw_disk(second, center=(5, 6), radius=1, value=9.0)
    fields = np.ones(first.shape, dtype=int)

    objects = extract_dynamic_objects(
        (
            _layer("NDMI", first, date="2024-05-01", layer_id="ndmi_1"),
            _layer("NDMI", second, date="2025-05-01", layer_id="ndmi_2"),
        ),
        fields,
        transform=TRANSFORM,
        crs=CRS,
        config=_config(min_area_m2=50.0, min_diameter_m=5.0, max_area_m2=600.0),
    )

    assert len(objects) == 1
    assert objects["source_detection_count"].iloc[0] == 2
    assert "too_large" in objects["dynamic_object_flags"].iloc[0]
    assert bool(objects["passes_dynamic_filters"].iloc[0]) is False


def _blank(height: int = 10, width: int = 10) -> np.ndarray:
    return np.zeros((height, width), dtype=float)


def _draw_disk(data: np.ndarray, *, center: tuple[int, int], radius: int, value: float) -> None:
    row_center, col_center = center
    rows, cols = np.ogrid[: data.shape[0], : data.shape[1]]
    mask = (rows - row_center) ** 2 + (cols - col_center) ** 2 <= radius**2
    data[mask] = value


def _layer(
    feature_name: str,
    data: np.ndarray,
    *,
    component: str = "moisture",
    date: str = "2024-05-01",
    layer_id: str | None = None,
) -> FieldAnomalyLayer:
    return FieldAnomalyLayer(
        name=f"{feature_name}_field_anomaly",
        component=component,
        data=data,
        date=date,
        source_feature_names=(feature_name,),
        source_layer_ids=(layer_id or f"{feature_name}_layer",),
        evidence_direction="higher_values_indicate_stronger_local_anomaly_support",
    )


def _composite(data: np.ndarray) -> CompositeAnomalyMap:
    return CompositeAnomalyMap(
        data=data,
        component_maps={"composite": data},
        component_weights={"composite": 1.0},
        source_feature_names=("NDMI", "NDVI"),
        source_layer_ids=("ndmi", "ndvi"),
    )


def _config(
    *,
    min_area_m2: float = 100.0,
    max_area_m2: float = 50_000.0,
    min_diameter_m: float = 5.0,
    max_diameter_m: float = 300.0,
    max_elongation: float = 4.0,
) -> DynamicExtractionConfig:
    return DynamicExtractionConfig(
        anomaly_percentile=90.0,
        min_area_m2=min_area_m2,
        max_area_m2=max_area_m2,
        min_diameter_m=min_diameter_m,
        max_diameter_m=max_diameter_m,
        max_elongation=max_elongation,
        merge_distance_m=20.0,
    )
