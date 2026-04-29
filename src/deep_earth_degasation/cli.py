from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer
from shapely.geometry import MultiPolygon, Polygon, shape
from shapely.geometry.base import BaseGeometry

from deep_earth_degasation.config import load_config
from deep_earth_degasation.io.candidates import write_candidate_artifacts
from deep_earth_degasation.morphology.static_detector import (
    StaticDetectorConfig,
    extract_static_candidates,
)

app = typer.Typer(help="Deep Earth Degasation MVP utilities")


@app.command()
def validate_config(config_path: Path) -> None:
    """Validate a YAML configuration file."""
    config = load_config(config_path)
    typer.echo(config.model_dump_json(indent=2))


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
    geometries, candidate_ids = _load_geojson_geometries(input_path)
    config = load_config(config_path)
    detector_config = _static_detector_config(config.object_constraints)
    candidates = extract_static_candidates(
        geometries, candidate_ids=candidate_ids, config=detector_config
    )
    artifact_paths = write_candidate_artifacts(candidates, output_dir)
    typer.echo(
        "Geometry-only static candidate artifacts written. "
        "These are review candidates, not direct H2 detections or proof of active degassing."
    )
    typer.echo(f"candidates_geojson={artifact_paths.candidates_geojson}")
    typer.echo(f"candidate_scores_csv={artifact_paths.candidate_scores_csv}")


def _load_geojson_geometries(path: Path) -> tuple[list[BaseGeometry], list[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    _reject_declared_geographic_crs(data)
    features = _geojson_features(data)
    geometries: list[BaseGeometry] = []
    candidate_ids: list[str] = []

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
        geometries.append(geometry)
        candidate_ids.append(_candidate_id(feature, index))

    if not geometries:
        raise typer.BadParameter("Input GeoJSON contains no candidate geometries.")
    _reject_lonlat_like_geometries(geometries)
    return geometries, candidate_ids


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
    properties = feature.get("properties") or {}
    if not isinstance(properties, dict):
        raise typer.BadParameter(f"Feature {index} properties must be an object.")
    candidate_id = properties.get("candidate_id") or feature.get("id")
    if candidate_id is None:
        return f"static-{index:06d}"
    return str(candidate_id)


def _static_detector_config(object_constraints: Any) -> StaticDetectorConfig:
    return StaticDetectorConfig(
        min_diameter_m=object_constraints.min_diameter_m,
        max_diameter_m=object_constraints.max_diameter_m,
        min_area=object_constraints.min_area_ha * 10_000.0,
        max_area=object_constraints.max_area_ha * 10_000.0,
    )


if __name__ == "__main__":
    app()
