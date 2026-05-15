from __future__ import annotations

import csv
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import yaml
from rasterio.transform import from_origin
from shapely.geometry import LineString, box
from typer.testing import CliRunner

from deep_earth_degasation.cli import app

CRS = "EPSG:32637"
TRANSFORM = from_origin(0, 100, 10, 10)
runner = CliRunner()


def test_run_mvp_cli_writes_dynamic_artifact_set(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    manifest_path = _write_prepared_data(tmp_path)
    output_dir = tmp_path / "outputs"

    result = runner.invoke(
        app,
        [
            "run-mvp",
            "--config",
            str(config_path),
            "--data-manifest",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "ranked candidate surface anomalies" in result.output
    assert "not direct H2 detections" in result.output
    assert "validation_summary_json=" in result.output

    candidates_geojson = output_dir / "candidates.geojson"
    candidate_scores_csv = output_dir / "candidate_scores.csv"
    labeling_table_csv = output_dir / "labeling_table.csv"
    validation_summary_json = output_dir / "validation_summary.json"
    passports_dir = output_dir / "passports"
    time_series_dir = output_dir / "time_series"
    run_manifest = output_dir / "run_manifest.json"
    resolved_config = output_dir / "resolved_config.json"
    anomaly_map = output_dir / "anomaly_maps" / "composite_anomaly.npy"

    for path in (
        candidates_geojson,
        candidate_scores_csv,
        labeling_table_csv,
        validation_summary_json,
        run_manifest,
        resolved_config,
        anomaly_map,
    ):
        assert path.exists()
    assert passports_dir.is_dir()
    assert time_series_dir.is_dir()

    with candidate_scores_csv.open(newline="", encoding="utf-8") as file:
        score_rows = list(csv.DictReader(file))
    assert score_rows
    assert score_rows[0]["object_score"] != ""
    assert score_rows[0]["dynamic_score"] != ""
    assert score_rows[0]["field_id"] == "plot-1"
    assert score_rows[0]["moisture_anomaly"] != ""
    assert score_rows[0]["false_positive_flags"] != ""
    assert score_rows[0]["missing_data_flags"] != ""
    assert score_rows[0]["passport_path"].endswith(".md")

    validation_summary = json.loads(validation_summary_json.read_text(encoding="utf-8"))
    assert "not direct H2 detections" in validation_summary["guardrail"]
    assert validation_summary["candidate_count"] == len(score_rows)
    assert validation_summary["unlabeled_background_treated_as_negative"] is False
    assert "missing_known_sites" in validation_summary["missing_data_flags"]
    assert "missing_expert_labels" in validation_summary["missing_data_flags"]

    with labeling_table_csv.open(newline="", encoding="utf-8") as file:
        labeling_rows = list(csv.DictReader(file))
    assert labeling_rows[0]["candidate_id"] == score_rows[0]["candidate_id"]
    assert labeling_rows[0]["source_feature_names"] != ""

    passport_path = Path(score_rows[0]["passport_path"])
    assert passport_path.exists()
    passport = passport_path.read_text(encoding="utf-8")
    assert "not direct H2 detection" in passport
    assert "`dynamic_score`:" in passport

    candidate_data = json.loads(candidates_geojson.read_text(encoding="utf-8"))
    candidate_gdf = gpd.read_file(candidates_geojson)
    assert str(candidate_gdf.crs) == CRS
    assert candidate_gdf.total_bounds[0] >= 0.0
    assert candidate_gdf.total_bounds[2] <= 100.0
    assert candidate_data["features"]
    assert candidate_data["features"][0]["properties"]["passport_path"].endswith(".md")
    assert list(time_series_dir.glob("*.csv"))
    assert np.load(anomaly_map).shape == (10, 10)

    stale_passport = passports_dir / "dynamic-object-999999.md"
    stale_time_series = time_series_dir / "dynamic-object-999999.csv"
    stale_passport.write_text("stale", encoding="utf-8")
    stale_time_series.write_text("stale", encoding="utf-8")

    rerun = runner.invoke(
        app,
        [
            "run-mvp",
            "--config",
            str(config_path),
            "--data-manifest",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert rerun.exit_code == 0, rerun.output
    assert not stale_passport.exists()
    assert not stale_time_series.exists()


def _write_config(tmp_path: Path) -> Path:
    config = {
        "project": {
            "name": "synthetic_dynamic_mvp",
            "version": "0.1",
            "mode": "prepared_data_dynamic_mvp",
        },
        "aoi": {"name": "synthetic", "input_geojson": str(tmp_path / "aoi.geojson")},
        "time": {
            "start_year": 2024,
            "end_year": 2024,
            "vegetation_months": [5],
            "bare_soil_months": [5],
        },
        "land_cover": {
            "primary_branch": "cropland",
            "classes": {"cropland": ["crops"], "forest": ["trees"], "mixed": ["mixed"]},
            "mixed_candidate_min_classes": 2,
        },
        "object_constraints": {
            "min_diameter_m": 5.0,
            "max_diameter_m": 300.0,
            "min_area_ha": 0.005,
            "max_area_ha": 10.0,
            "min_distance_to_field_edge_m": 5.0,
        },
        "prepared_data": {
            "raster_stack_manifest": str(tmp_path / "manifest.yaml"),
            "reject_lonlat_for_metric_operations": True,
            "allow_reprojection": False,
            "allow_resampling": False,
        },
        "normalization": {
            "cropland_method": "field_level_robust_zscore",
            "center": "median",
            "scale": "MAD",
            "min_valid_pixels_per_field_date": 1,
            "mad_epsilon": 1.0e-6,
        },
        "dynamic_detector": {
            "anomaly_percentile": 90.0,
            "min_valid_observations_per_season": 1,
            "min_repeated_seasons": 1,
            "connected_components": True,
            "morphology_filter": True,
            "merge_distance_m": 20.0,
            "merge_across_dates": True,
        },
        "false_positive_filters": {
            "use_penalties": True,
            "flag_roads": True,
            "road_buffer_m": 20.0,
            "flag_water": False,
            "flag_built_up": False,
            "flag_field_edges": False,
            "flag_linear_objects": False,
            "flag_cloud_shadows": False,
            "flag_harvest_patterns": False,
            "flag_irrigation": False,
            "flag_quarries": False,
        },
        "anomaly_components": {
            "moisture": {"inputs": ["NDMI"], "high_score_means": "dry", "weight": 0.4},
            "vegetation_stress": {
                "inputs": ["NDVI"],
                "high_score_means": "stress",
                "weight": 0.3,
            },
            "soil_brightness": {"inputs": ["BSI"], "high_score_means": "bright", "weight": 0.3},
        },
        "scoring": {
            "priority_thresholds": {"A": 0.75, "B": 0.55, "C": 0.35, "D": 0.0},
            "cropland": {
                "moisture_weight": 0.4,
                "vegetation_weight": 0.3,
                "brightness_weight": 0.3,
            },
        },
        "validation": {
            "top_n": 20,
            "known_site_recall_top_n": [1, 20],
            "expert_precision_top_n": 20,
            "export_false_positive_counts": True,
            "do_not_treat_unlabeled_as_negative": True,
        },
        "outputs": {
            "candidates_top_n": 30,
            "export_geojson": True,
            "export_csv": True,
            "generate_passports": True,
            "export_resolved_config": True,
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def _write_prepared_data(tmp_path: Path) -> Path:
    aoi_path = tmp_path / "aoi.geojson"
    fields_path = tmp_path / "fields.geojson"
    roads_path = tmp_path / "roads.geojson"
    _write_vector(aoi_path, {"name": ["synthetic"]}, [box(0, 0, 100, 100)])
    _write_vector(fields_path, {"plot_code": ["plot-1"]}, [box(0, 0, 100, 100)])
    _write_vector(roads_path, {"road_id": ["road-1"]}, [LineString([(45, 0), (45, 100)])])

    ndmi = np.full((10, 10), 0.6, dtype=np.float32)
    ndvi = np.full((10, 10), 0.7, dtype=np.float32)
    bsi = np.full((10, 10), 0.2, dtype=np.float32)
    ndmi[4:7, 4:7] = 0.05
    ndvi[4:7, 4:7] = 0.10
    bsi[4:7, 4:7] = 1.0
    _write_raster(tmp_path / "ndmi.tif", ndmi)
    _write_raster(tmp_path / "ndvi.tif", ndvi)
    _write_raster(tmp_path / "bsi.tif", bsi)

    manifest = {
        "run": {"id": "synthetic_dynamic_cli", "created_by": "pytest"},
        "crs": CRS,
        "resolution_m": 10,
        "aoi": {"path": str(aoi_path), "crs": CRS},
        "vectors": {
            "fields": {
                "path": str(fields_path),
                "crs": CRS,
                "role": "fields",
                "required": True,
                "id_field": "plot_code",
            },
            "roads": {
                "path": str(roads_path),
                "crs": CRS,
                "role": "roads",
                "required": False,
            },
        },
        "raster_layers": [
            _layer("ndmi", "NDMI", "ndmi.tif"),
            _layer("ndvi", "NDVI", "ndvi.tif"),
            _layer("bsi", "BSI", "bsi.tif"),
        ],
        "quality": {
            "reject_mismatched_crs": True,
            "reject_mismatched_grid": True,
            "allow_resampling": False,
        },
    }
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    return manifest_path


def _write_vector(path: Path, properties: dict[str, list[str]], geometries: list[object]) -> None:
    gpd.GeoDataFrame({**properties, "geometry": geometries}, geometry="geometry", crs=CRS).to_file(
        path,
        driver="GeoJSON",
    )


def _write_raster(path: Path, data: np.ndarray) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype="float32",
        crs=CRS,
        transform=TRANSFORM,
        nodata=-9999.0,
    ) as dataset:
        dataset.write(data, 1)


def _layer(layer_id: str, feature_name: str, filename: str) -> dict[str, object]:
    return {
        "id": layer_id,
        "sensor": "sentinel2",
        "feature_name": feature_name,
        "date": "2024-05-15",
        "path": filename,
        "crs": CRS,
        "resolution_m": 10,
        "nodata": -9999,
        "role": "input_feature",
    }
