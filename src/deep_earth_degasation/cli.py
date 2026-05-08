from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Annotated, Any

import typer
from shapely.geometry import MultiPolygon, Polygon, shape
from shapely.geometry.base import BaseGeometry

from deep_earth_degasation.config import MVPConfig, load_config, resolved_config_dict
from deep_earth_degasation.io.candidates import SourceProperties, write_candidate_artifacts
from deep_earth_degasation.io.labeling import write_labeling_table
from deep_earth_degasation.morphology.static_detector import (
    StaticDetectorConfig,
    extract_static_candidates,
)
from deep_earth_degasation.pipeline.manifest import load_prepared_stack_manifest
from deep_earth_degasation.pipeline.run_mvp import GUARDRAIL_MESSAGE, run_dynamic_mvp
from deep_earth_degasation.reports.passport import write_candidate_passport

app = typer.Typer(help="Deep Earth Degasation MVP utilities")


@app.command()
def validate_config(config_path: Path) -> None:
    """Validate a YAML configuration file."""
    config = load_config(config_path)
    typer.echo(json.dumps(resolved_config_dict(config), indent=2))


@app.command()
def status() -> None:
    """Print package status."""
    typer.echo("deep-earth-degasation MVP skeleton is installed.")


@app.command()
def static_candidates(
    input_path: Annotated[
        Path,
        typer.Option(
            "--input",
            "-i",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help=(
                "Input GeoJSON with Polygon or MultiPolygon candidate geometries. "
                "Coordinates must already be projected in metres."
            ),
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            "-o",
            file_okay=False,
            dir_okay=True,
            help="Directory for candidates.geojson and candidate_scores.csv.",
        ),
    ],
    config_path: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="MVP YAML config. Only geometry/static object constraints are used here.",
        ),
    ] = Path("configs/lipetsk_voronezh_mvp.yaml"),
) -> None:
    """Run geometry-only static candidate extraction on metre coordinates."""
    geometries, candidate_ids, source_properties = _load_geojson_geometries(input_path)
    config = load_config(config_path)
    detector_config = _static_detector_config(config)
    candidates = extract_static_candidates(
        geometries, candidate_ids=candidate_ids, config=detector_config
    )
    artifact_paths = write_candidate_artifacts(candidates, output_dir, source_properties)
    typer.echo(
        "Geometry-only static candidate artifacts written. "
        "These are review candidates, not direct H2 detections or proof of active degassing."
    )
    score_rows = _read_score_rows(artifact_paths.candidate_scores_csv)
    passports_dir = output_dir / "passports"
    passport_paths = _passport_paths(score_rows, passports_dir)
    for score_row, passport_path in zip(score_rows, passport_paths, strict=True):
        write_candidate_passport(score_row, passport_path)
    labeling_table_path = output_dir / "labeling_table.csv"
    write_labeling_table(score_rows, labeling_table_path)
    typer.echo(f"candidates_geojson={artifact_paths.candidates_geojson}")
    typer.echo(f"candidate_scores_csv={artifact_paths.candidate_scores_csv}")
    typer.echo(f"passports_dir={passports_dir}")
    typer.echo(f"labeling_table_csv={labeling_table_path}")


@app.command("run-mvp")
def run_mvp(
    config_path: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Dynamic MVP YAML config.",
        ),
    ],
    data_manifest_path: Annotated[
        Path,
        typer.Option(
            "--data-manifest",
            "-m",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Prepared raster/vector stack manifest.",
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            "-o",
            file_okay=False,
            dir_okay=True,
            help="Directory for dynamic MVP artifacts.",
        ),
    ],
) -> None:
    """Run the prepared-data dynamic MVP artifact pipeline."""
    config = load_config(config_path)
    manifest = load_prepared_stack_manifest(data_manifest_path)
    paths = run_dynamic_mvp(config=config, manifest=manifest, output_dir=output_dir)
    typer.echo(GUARDRAIL_MESSAGE)
    typer.echo(f"candidates_geojson={paths.candidates_geojson}")
    typer.echo(f"candidate_scores_csv={paths.candidate_scores_csv}")
    typer.echo(f"labeling_table_csv={paths.labeling_table_csv}")
    typer.echo(f"passports_dir={paths.passports_dir}")
    typer.echo(f"time_series_dir={paths.time_series_dir}")
    typer.echo(f"run_manifest_json={paths.run_manifest_json}")
    typer.echo(f"resolved_config_json={paths.resolved_config_json}")


def _read_score_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _passport_paths(score_rows: list[dict[str, str]], passports_dir: Path) -> list[Path]:
    seen: dict[str, int] = {}
    paths: list[Path] = []
    for score_row in score_rows:
        stem = _safe_filename_stem(score_row.get("candidate_id", ""), score_row.get("rank", ""))
        count = seen.get(stem, 0)
        seen[stem] = count + 1
        if count > 0:
            stem = f"{stem}-{count + 1}"
        paths.append(passports_dir / f"{stem}.md")
    return paths


def _safe_filename_stem(candidate_id: str, rank: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", candidate_id).strip(" ._")
    if stem in {"", ".", ".."}:
        rank_suffix = re.sub(r"[^A-Za-z0-9._-]+", "_", rank).strip(" ._")
        return f"candidate_{rank_suffix or 'unknown'}"
    return stem


def _load_geojson_geometries(path: Path) -> tuple[list[BaseGeometry], list[str], SourceProperties]:
    data = json.loads(path.read_text(encoding="utf-8"))
    _reject_declared_geographic_crs(data)
    features = _geojson_features(data)
    geometries: list[BaseGeometry] = []
    candidate_ids: list[str] = []
    source_properties: SourceProperties = []

    for index, feature in enumerate(features, start=1):
        geometry_data = feature.get("geometry")
        if geometry_data is None:
            raise typer.BadParameter(f"Feature {index} has no geometry.")
        geometry = shape(geometry_data)
        if not isinstance(geometry, Polygon | MultiPolygon):
            raise typer.BadParameter(
                f"Feature {index} has unsupported geometry type {geometry.geom_type}."
            )
        if geometry.is_empty:
            raise typer.BadParameter(f"Feature {index} has empty geometry.")
        candidate_id = _candidate_id(feature, index)
        geometries.append(geometry)
        candidate_ids.append(candidate_id)
        source_properties.append(_feature_properties(feature, index))

    if not geometries:
        raise typer.BadParameter("Input GeoJSON contains no candidate geometries.")
    _reject_lonlat_like_geometries(geometries)
    return geometries, candidate_ids, source_properties


def _reject_declared_geographic_crs(data: dict[str, Any]) -> None:
    crs_text = json.dumps(data.get("crs", {})).lower()
    if any(token in crs_text for token in ("epsg:4326", "crs84", "wgs84", "wgs 84")):
        raise typer.BadParameter(_METRIC_COORDINATE_MESSAGE)


def _reject_lonlat_like_geometries(geometries: list[BaseGeometry]) -> None:
    for geometry in geometries:
        minx, miny, maxx, maxy = geometry.bounds
        if -180.0 <= minx <= maxx <= 180.0 and -90.0 <= miny <= maxy <= 90.0:
            raise typer.BadParameter(_METRIC_COORDINATE_MESSAGE)


_METRIC_COORDINATE_MESSAGE = (
    "static-candidates requires projected metric coordinates. "
    "Reproject lon/lat GeoJSON to a metre-based CRS before running this command; "
    "automatic reprojection is not implemented."
)


def _geojson_features(data: dict[str, Any]) -> list[dict[str, Any]]:
    geojson_type = data.get("type")
    if geojson_type == "FeatureCollection":
        features = data.get("features", [])
        if not isinstance(features, list):
            raise typer.BadParameter("FeatureCollection features must be a list.")
        return [_require_feature(feature, index) for index, feature in enumerate(features, start=1)]
    if geojson_type == "Feature":
        return [_require_feature(data, 1)]
    if geojson_type in {"Polygon", "MultiPolygon"}:
        return [{"type": "Feature", "properties": {}, "geometry": data}]
    raise typer.BadParameter(
        "Input GeoJSON must be a FeatureCollection, Feature, Polygon, or MultiPolygon."
    )


def _require_feature(value: object, index: int) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("type") != "Feature":
        raise typer.BadParameter(f"Item {index} is not a GeoJSON Feature.")
    return value


def _candidate_id(feature: dict[str, Any], index: int) -> str:
    properties = _feature_properties(feature, index)
    candidate_id = properties.get("candidate_id") or feature.get("id")
    if candidate_id is None:
        return f"static-{index:06d}"
    return str(candidate_id)


def _feature_properties(feature: dict[str, Any], index: int) -> dict[str, object]:
    properties = feature.get("properties") or {}
    if not isinstance(properties, dict):
        raise typer.BadParameter(f"Feature {index} properties must be an object.")
    return properties


def _static_detector_config(config: MVPConfig) -> StaticDetectorConfig:
    object_constraints = config.object_constraints
    static_config = config.static_ring_detector
    return StaticDetectorConfig(
        min_diameter_m=object_constraints.min_diameter_m,
        max_diameter_m=object_constraints.max_diameter_m,
        min_area=object_constraints.min_area_ha * 10_000.0,
        max_area=object_constraints.max_area_ha * 10_000.0,
        ellipse_min_circularity=static_config.circularity_min,
    )


if __name__ == "__main__":
    app()
