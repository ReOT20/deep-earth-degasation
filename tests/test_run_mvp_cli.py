from __future__ import annotations

import csv
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pytest
import rasterio
import yaml
from rasterio.transform import from_origin
from shapely.geometry import LineString, box
from typer.testing import CliRunner

from deep_earth_degasation.cli import app
from deep_earth_degasation.config import load_config
from deep_earth_degasation.io.vector import VectorDataError
from deep_earth_degasation.pipeline.manifest import load_prepared_stack_manifest
from deep_earth_degasation.pipeline.run_mvp import _false_positive_filter_config, _load_vectors

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
    assert "learning_dataset_csv=" in result.output

    candidates_geojson = output_dir / "candidates.geojson"
    candidate_scores_csv = output_dir / "candidate_scores.csv"
    labeling_table_csv = output_dir / "labeling_table.csv"
    learning_dataset_csv = output_dir / "learning_dataset.csv"
    validation_summary_json = output_dir / "validation_summary.json"
    passports_dir = output_dir / "passports"
    time_series_dir = output_dir / "time_series"
    run_manifest = output_dir / "run_manifest.json"
    resolved_config = output_dir / "resolved_config.json"
    anomaly_map = output_dir / "anomaly_maps" / "composite_anomaly.npy"
    residual_cube = output_dir / "anomaly_maps" / "hierarchical_residuals.npz"
    residual_metadata_path = output_dir / "anomaly_maps" / "hierarchical_residuals.json"

    for path in (
        candidates_geojson,
        candidate_scores_csv,
        labeling_table_csv,
        learning_dataset_csv,
        validation_summary_json,
        run_manifest,
        resolved_config,
        anomaly_map,
        residual_cube,
        residual_metadata_path,
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
    assert score_rows[0]["field_residual_max"] != ""
    assert "field" in score_rows[0]["residual_types_present"]
    assert score_rows[0]["moisture_anomaly"] != ""
    assert score_rows[0]["false_positive_flags"] != ""
    assert score_rows[0]["missing_data_flags"] != ""
    assert score_rows[0]["landcover_branch"] == "cropland"
    assert "landcover_context_assumed_cropland_fields" in score_rows[0]["missing_data_flags"]
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

    with learning_dataset_csv.open(newline="", encoding="utf-8") as file:
        learning_rows = list(csv.DictReader(file))
    assert learning_rows[0]["candidate_id"] == score_rows[0]["candidate_id"]
    assert learning_rows[0]["model_label"] == "unlabeled"
    assert learning_rows[0]["pu_role"] == "unlabeled_pool"
    assert learning_rows[0]["split_group"] == "field:plot-1"
    assert learning_rows[0]["feature_snapshot_id"].startswith(
        "prepared_manifest:synthetic_dynamic_cli:"
    )
    assert learning_rows[0]["feature_snapshot_id"] != "run_manifest.json"
    assert learning_rows[0]["geometry_ref"] == str(candidates_geojson)

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
    residual_metadata = json.loads(residual_metadata_path.read_text(encoding="utf-8"))
    assert residual_metadata["schema_version"] == "hierarchical_residuals.v1"
    assert "missing_peer_residual_context" in residual_metadata["missing_data_flags"]
    assert "missing_temporal_residual_context" in residual_metadata["missing_data_flags"]
    assert {layer["residual_type"] for layer in residual_metadata["layers"]} == {"field"}
    with np.load(residual_cube) as residuals:
        assert residuals.files

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


def test_run_mvp_penalty_toggle_keeps_flags_without_score_penalty(tmp_path: Path) -> None:
    true_config_path = _write_config(tmp_path / "penalties_true")
    true_manifest_path = _write_prepared_data(tmp_path / "penalties_true")
    false_config_path = _write_config(tmp_path / "penalties_false")
    false_manifest_path = _write_prepared_data(tmp_path / "penalties_false")
    _update_config(
        true_config_path,
        lambda config: config["scoring"].update({"penalties": {"road": 0.6}}),
    )
    _update_config(
        false_config_path,
        lambda config: (
            config["scoring"].update({"penalties": {"road": 0.6}}),
            config["false_positive_filters"].update({"use_penalties": False}),
        ),
    )

    true_output = tmp_path / "true_outputs"
    false_output = tmp_path / "false_outputs"
    true_result = runner.invoke(
        app,
        [
            "run-mvp",
            "--config",
            str(true_config_path),
            "--data-manifest",
            str(true_manifest_path),
            "--output-dir",
            str(true_output),
        ],
    )
    false_result = runner.invoke(
        app,
        [
            "run-mvp",
            "--config",
            str(false_config_path),
            "--data-manifest",
            str(false_manifest_path),
            "--output-dir",
            str(false_output),
        ],
    )

    assert true_result.exit_code == 0, true_result.output
    assert false_result.exit_code == 0, false_result.output
    true_row = _read_csv_rows(true_output / "candidate_scores.csv")[0]
    false_row = _read_csv_rows(false_output / "candidate_scores.csv")[0]
    assert "road_risk" in false_row["false_positive_flags"]
    assert float(true_row["false_positive_penalty"]) == 0.6
    assert float(false_row["false_positive_penalty"]) == 0.0
    assert float(false_row["object_score"]) > float(true_row["object_score"])


def test_optional_empty_vector_files_are_treated_as_absent(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    manifest_path = _write_prepared_data(tmp_path)
    empty_context = tmp_path / "empty_quarries.geojson"
    _write_empty_geojson(empty_context)
    _update_manifest(
        manifest_path,
        lambda manifest: manifest["vectors"].update(
            {
                "quarries": {
                    "path": str(empty_context),
                    "crs": CRS,
                    "role": "quarries",
                    "required": False,
                }
            }
        ),
    )

    vectors = _load_vectors(load_prepared_stack_manifest(manifest_path), load_config(config_path))

    assert vectors["quarries"] is None


def test_run_mvp_assigns_landcover_from_manifest_layer(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    manifest_path = _write_prepared_data(tmp_path)
    landcover_path = tmp_path / "landcover.geojson"
    _write_vector(landcover_path, {"landcover_class": ["trees"]}, [box(0, 0, 100, 100)])
    _update_manifest(
        manifest_path,
        lambda manifest: manifest["vectors"].update(
            {
                "landcover": {
                    "path": str(landcover_path),
                    "crs": CRS,
                    "role": "landcover",
                    "required": False,
                }
            }
        ),
    )
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
    row = _read_csv_rows(output_dir / "candidate_scores.csv")[0]
    assert row["landcover_branch"] == "forest"
    assert "landcover_context_assumed_cropland_fields" not in row["missing_data_flags"]
    assert "missing_forest_branch_scoring" in row["missing_data_flags"]


def test_run_mvp_reports_peer_and_temporal_residuals_when_context_exists(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    manifest_path = _write_prepared_data(tmp_path)
    fields_path = tmp_path / "fields.geojson"
    _write_vector(
        fields_path,
        {
            "plot_code": ["plot-west", "plot-east"],
            "crop_type": ["wheat", "wheat"],
            "phenology_stage": ["early", "early"],
        },
        [box(0, 0, 50, 100), box(50, 0, 100, 100)],
    )
    late_ndmi = np.full((10, 10), 0.6, dtype=np.float32)
    late_ndmi[4:7, 4:7] = 0.02
    _write_raster(tmp_path / "ndmi_late.tif", late_ndmi)
    _update_manifest(
        manifest_path,
        lambda manifest: manifest["raster_layers"].append(
            _layer("ndmi_late", "NDMI", "ndmi_late.tif", date="2024-06-15")
        ),
    )
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
    row = _read_csv_rows(output_dir / "candidate_scores.csv")[0]
    assert "peer" in row["residual_types_present"]
    assert "temporal" in row["residual_types_present"]
    assert row["peer_residual_max"] != ""
    assert row["temporal_residual_max"] != ""
    metadata = json.loads(
        (output_dir / "anomaly_maps" / "hierarchical_residuals.json").read_text(encoding="utf-8")
    )
    assert {"field", "peer", "temporal"}.issubset(
        {layer["residual_type"] for layer in metadata["layers"]}
    )


def test_required_empty_vector_files_still_fail(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    manifest_path = _write_prepared_data(tmp_path)
    empty_fields = tmp_path / "empty_fields.geojson"
    _write_empty_geojson(empty_fields)
    _update_manifest(
        manifest_path,
        lambda manifest: manifest["vectors"]["fields"].update({"path": str(empty_fields)}),
    )

    with pytest.raises(VectorDataError, match="fields contains no features"):
        _load_vectors(load_prepared_stack_manifest(manifest_path), load_config(config_path))


def test_run_mvp_output_toggles_control_artifacts(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    manifest_path = _write_prepared_data(tmp_path)
    _update_config(
        config_path,
        lambda config: config["outputs"].update(
            {
                "export_geojson": False,
                "export_csv": False,
                "generate_passports": False,
                "generate_quicklooks": False,
                "export_time_series": False,
                "export_validation_summary": False,
                "export_resolved_config": False,
            }
        ),
    )
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
    assert not (output_dir / "candidates.geojson").exists()
    assert not (output_dir / "candidate_scores.csv").exists()
    assert not (output_dir / "validation_summary.json").exists()
    assert not (output_dir / "resolved_config.json").exists()
    assert not list((output_dir / "passports").glob("*.md"))
    assert not list((output_dir / "time_series").glob("*.csv"))
    assert not list((output_dir / "quicklooks").glob("*.png"))
    assert (output_dir / "labeling_table.csv").exists()
    assert (output_dir / "learning_dataset.csv").exists()


def test_candidates_top_n_limits_review_exports(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    manifest_path = _write_prepared_data(tmp_path, extra_blobs=True)
    _update_config(
        config_path,
        lambda config: (
            config["outputs"].update(
                {
                    "candidates_top_n": 2,
                    "generate_quicklooks": True,
                    "export_time_series": True,
                }
            ),
            config["dynamic_detector"].update({"merge_across_dates": False}),
        ),
    )
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
    score_rows = _read_csv_rows(output_dir / "candidate_scores.csv")
    assert len(score_rows) > 2
    top_passport_paths = [Path(row["passport_path"]) for row in score_rows[:2]]
    assert all(path.exists() for path in top_passport_paths)
    assert all(row["passport_path"] == "" for row in score_rows[2:])
    candidate_data = json.loads((output_dir / "candidates.geojson").read_text(encoding="utf-8"))
    features = candidate_data["features"]
    assert len(features) == len(score_rows)
    feature_passport_paths = [feature["properties"]["passport_path"] for feature in features]
    assert all(Path(path).exists() for path in feature_passport_paths[:2])
    assert feature_passport_paths[2:] == [""] * (len(feature_passport_paths) - 2)
    assert len(_read_csv_rows(output_dir / "labeling_table.csv")) == 2
    assert len(list((output_dir / "passports").glob("*.md"))) == 2
    assert len(list((output_dir / "time_series").glob("*.csv"))) == 2
    assert len(list((output_dir / "quicklooks").glob("*.png"))) == 2


def test_field_edge_buffer_falls_back_to_object_constraint(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    _update_config(
        config_path,
        lambda config: (
            config["object_constraints"].update({"min_distance_to_field_edge_m": 13.0}),
            config["false_positive_filters"].pop("field_edge_buffer_m", None),
        ),
    )

    filter_config = _false_positive_filter_config(load_config(config_path))

    assert filter_config.field_edge_buffer_m == 13.0


def test_connected_components_false_is_rejected(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    manifest_path = _write_prepared_data(tmp_path)
    _update_config(
        config_path,
        lambda config: config["dynamic_detector"].update({"connected_components": False}),
    )

    result = runner.invoke(
        app,
        [
            "run-mvp",
            "--config",
            str(config_path),
            "--data-manifest",
            str(manifest_path),
            "--output-dir",
            str(tmp_path / "outputs"),
        ],
    )

    assert result.exit_code != 0
    assert isinstance(result.exception, ValueError)
    assert "connected_components=false is not supported" in str(result.exception)


def test_configured_weather_context_without_weather_data_is_flagged(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    manifest_path = _write_prepared_data(tmp_path)
    _update_config(
        config_path,
        lambda config: config["scoring"]["cropland"].update({"post_rain_weight": 0.2}),
    )
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
    row = _read_csv_rows(output_dir / "candidate_scores.csv")[0]
    assert "missing_weather_context" in row["missing_data_flags"]


def test_run_mvp_integrates_event_conditioned_features(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    manifest_path = _write_prepared_data(tmp_path)
    ndmi_before = np.full((10, 10), 0.8, dtype=np.float32)
    ndre = np.full((10, 10), 0.7, dtype=np.float32)
    vv_before = np.full((10, 10), 2.0, dtype=np.float32)
    vv_after = np.full((10, 10), 2.0, dtype=np.float32)
    ndre[4:7, 4:7] = 0.1
    vv_after[4:7, 4:7] = 5.0
    _write_raster(tmp_path / "ndmi_before.tif", ndmi_before)
    _write_raster(tmp_path / "ndre.tif", ndre)
    _write_raster(tmp_path / "vv_before.tif", vv_before)
    _write_raster(tmp_path / "vv_after.tif", vv_after)
    weather_path = tmp_path / "weather.yaml"
    weather_path.write_text(
        """
events:
  - date: "2024-05-10"
    rainfall_mm: 12.0
    source_id: rain_1
""",
        encoding="utf-8",
    )
    _update_config(
        config_path,
        lambda config: (
            config.setdefault("features", {}).update(
                {
                    "sentinel2": {
                        "indices": ["NDVI", "NDMI", "BSI"],
                        "optional_indices": ["NDRE"],
                        "cloud_threshold": 30,
                    },
                    "sentinel1": {"features": ["VV", "sar_event_response"]},
                    "landsat": {"features": ["LST"]},
                    "weather": {"features": ["post_rain_drying"]},
                }
            ),
            config["anomaly_components"]["vegetation_stress"].update({"inputs": ["NDRE"]}),
            config["anomaly_components"].update(
                {
                    "post_rain_drying": {
                        "inputs": ["post_rain_drying"],
                        "high_score_means": "supporting_post_rain_drying",
                        "weight": 0.2,
                    },
                    "sar": {
                        "inputs": ["VV", "sar_event_response"],
                        "high_score_means": "supporting_sar_event_response",
                        "weight": 0.2,
                    },
                }
            ),
            config["scoring"]["cropland"].update({"post_rain_weight": 0.1, "sar_weight": 0.1}),
        ),
    )
    _update_manifest(
        manifest_path,
        lambda manifest: (
            manifest.update({"weather_context": {"path": str(weather_path)}}),
            manifest["raster_layers"].extend(
                [
                    _layer("ndmi_before", "NDMI", "ndmi_before.tif", date="2024-05-01"),
                    _layer("ndre", "NDRE", "ndre.tif", date="2024-05-15"),
                    {
                        **_layer("vv_before", "VV", "vv_before.tif", date="2024-05-01"),
                        "sensor": "sentinel1",
                    },
                    {
                        **_layer("vv_after", "VV", "vv_after.tif", date="2024-05-15"),
                        "sensor": "sentinel1",
                    },
                ]
            ),
        ),
    )
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
    row = _read_csv_rows(output_dir / "candidate_scores.csv")[0]
    assert row["vegetation_stress"] != ""
    assert row["post_rain_drying"] != ""
    assert row["sar_anomaly"] != ""
    assert "NDRE" in row["source_feature_names"]
    assert "post_rain_drying_NDMI" in row["source_feature_names"]
    assert "sar_event_response_VV" in row["source_feature_names"]
    assert "missing_weather_context" not in row["missing_data_flags"]
    assert "missing_crop_density_context_for_sar" not in row["missing_data_flags"]


def test_merge_across_dates_toggle_does_not_overmerge_composite_components(
    tmp_path: Path,
) -> None:
    merged_config_path = _write_config(tmp_path / "merged")
    merged_manifest_path = _write_prepared_data(tmp_path / "merged", extra_blobs=True)
    unmerged_config_path = _write_config(tmp_path / "unmerged")
    unmerged_manifest_path = _write_prepared_data(tmp_path / "unmerged", extra_blobs=True)
    _update_config(
        unmerged_config_path,
        lambda config: config["dynamic_detector"].update({"merge_across_dates": False}),
    )

    merged_result = runner.invoke(
        app,
        [
            "run-mvp",
            "--config",
            str(merged_config_path),
            "--data-manifest",
            str(merged_manifest_path),
            "--output-dir",
            str(tmp_path / "merged_outputs"),
        ],
    )
    unmerged_result = runner.invoke(
        app,
        [
            "run-mvp",
            "--config",
            str(unmerged_config_path),
            "--data-manifest",
            str(unmerged_manifest_path),
            "--output-dir",
            str(tmp_path / "unmerged_outputs"),
        ],
    )

    assert merged_result.exit_code == 0, merged_result.output
    assert unmerged_result.exit_code == 0, unmerged_result.output
    merged_count = len(_read_csv_rows(tmp_path / "merged_outputs" / "candidate_scores.csv"))
    unmerged_count = len(_read_csv_rows(tmp_path / "unmerged_outputs" / "candidate_scores.csv"))
    assert merged_count == unmerged_count
    assert merged_count > 1


def test_time_windows_exclude_out_of_window_components(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    manifest_path = _write_prepared_data(tmp_path)
    ndre = np.full((10, 10), 0.7, dtype=np.float32)
    ndre[4:7, 4:7] = 0.1
    _write_raster(tmp_path / "ndre.tif", ndre)
    _update_config(
        config_path,
        lambda config: (
            config["time"].update({"vegetation_months": [6], "bare_soil_months": [5]}),
            config["anomaly_components"]["vegetation_stress"].update({"inputs": ["NDVI", "NDRE"]}),
        ),
    )
    _update_manifest(
        manifest_path,
        lambda manifest: manifest["raster_layers"].append(_layer("ndre", "NDRE", "ndre.tif")),
    )
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
    row = _read_csv_rows(output_dir / "candidate_scores.csv")[0]
    assert row["vegetation_stress"] == ""
    assert "skipped_out_of_window_NDVI_2024-05-15" in row["missing_data_flags"]
    assert "skipped_out_of_window_NDRE_2024-05-15" in row["missing_data_flags"]


def test_composite_source_observations_satisfy_minimum_observation_flag(
    tmp_path: Path,
) -> None:
    config_path = _write_config(tmp_path)
    manifest_path = _write_prepared_data(tmp_path)
    _update_config(
        config_path,
        lambda config: config["dynamic_detector"].update({"min_valid_observations_per_season": 3}),
    )
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
    row = _read_csv_rows(output_dir / "candidate_scores.csv")[0]
    assert "below_min_valid_observations_per_season" not in row["missing_data_flags"]


def _write_config(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
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


def _write_prepared_data(tmp_path: Path, *, extra_blobs: bool = False) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
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
    if extra_blobs:
        ndmi[1:3, 1:3] = 0.05
        ndvi[1:3, 1:3] = 0.10
        bsi[1:3, 1:3] = 1.0
        ndmi[8:10, 7:9] = 0.05
        ndvi[8:10, 7:9] = 0.10
        bsi[8:10, 7:9] = 1.0
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


def _update_config(path: Path, update: Callable[[dict[str, Any]], object]) -> None:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    update(config)
    path.write_text(yaml.safe_dump(config), encoding="utf-8")


def _update_manifest(path: Path, update: Callable[[dict[str, Any]], object]) -> None:
    manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
    update(manifest)
    path.write_text(yaml.safe_dump(manifest), encoding="utf-8")


def _write_empty_geojson(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "crs": {"type": "name", "properties": {"name": CRS}},
                "features": [],
            }
        ),
        encoding="utf-8",
    )


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


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


def _layer(
    layer_id: str,
    feature_name: str,
    filename: str,
    *,
    date: str = "2024-05-15",
) -> dict[str, object]:
    return {
        "id": layer_id,
        "sensor": "sentinel2",
        "feature_name": feature_name,
        "date": date,
        "path": filename,
        "crs": CRS,
        "resolution_m": 10,
        "nodata": -9999,
        "role": "input_feature",
    }
