from __future__ import annotations

import csv
import json
from pathlib import Path

from typer.testing import CliRunner

from deep_earth_degasation.cli import app

runner = CliRunner()


def test_static_candidates_command_writes_artifacts(tmp_path: Path) -> None:
    input_path = tmp_path / "input.geojson"
    output_dir = tmp_path / "artifacts"
    input_path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"candidate_id": "low-score-first"},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [
                                [
                                    [500000.0, 5600000.0],
                                    [500120.0, 5600000.0],
                                    [500095.0, 5600035.0],
                                    [500140.0, 5600090.0],
                                    [500045.0, 5600065.0],
                                    [500000.0, 5600120.0],
                                    [500000.0, 5600000.0],
                                ]
                            ],
                        },
                    },
                    {
                        "type": "Feature",
                        "properties": {"candidate_id": "high-score-second"},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [
                                [
                                    [499950.0, 5599950.0],
                                    [500050.0, 5599950.0],
                                    [500050.0, 5600050.0],
                                    [499950.0, 5600050.0],
                                    [499950.0, 5599950.0],
                                ],
                                [
                                    [499985.0, 5599985.0],
                                    [500015.0, 5599985.0],
                                    [500015.0, 5600015.0],
                                    [499985.0, 5600015.0],
                                    [499985.0, 5599985.0],
                                ],
                            ],
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "static-candidates",
            "--input",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--config",
            "configs/lipetsk_voronezh_mvp.yaml",
        ],
    )

    assert result.exit_code == 0
    assert "not direct H2 detections or proof" in result.output
    candidates_geojson = output_dir / "candidates.geojson"
    candidate_scores_csv = output_dir / "candidate_scores.csv"
    assert candidates_geojson.exists()
    assert candidate_scores_csv.exists()

    candidate_data = json.loads(candidates_geojson.read_text(encoding="utf-8"))
    assert [feature["properties"]["candidate_id"] for feature in candidate_data["features"]] == [
        "low-score-first",
        "high-score-second",
    ]

    with candidate_scores_csv.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert rows[0]["rank"] == "1"
    assert rows[0]["candidate_id"] == "high-score-second"
    assert rows[1]["rank"] == "2"
    assert rows[1]["candidate_id"] == "low-score-first"


def test_static_candidates_help_states_geometry_only_limitation() -> None:
    result = runner.invoke(app, ["static-candidates", "--help"])

    assert result.exit_code == 0
    assert "geometry-only" in result.output
    assert "metre" in result.output


def test_static_candidates_rejects_empty_feature_collection(tmp_path: Path) -> None:
    input_path = tmp_path / "empty.geojson"
    input_path.write_text(
        json.dumps({"type": "FeatureCollection", "features": []}), encoding="utf-8"
    )

    result = runner.invoke(
        app,
        [
            "static-candidates",
            "--input",
            str(input_path),
            "--output-dir",
            str(tmp_path / "artifacts"),
            "--config",
            "configs/lipetsk_voronezh_mvp.yaml",
        ],
    )

    assert result.exit_code != 0
    assert "contains no candidate geometries" in result.output


def test_static_candidates_rejects_lonlat_like_coordinates(tmp_path: Path) -> None:
    input_path = tmp_path / "lonlat.geojson"
    input_path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"candidate_id": "lonlat-like"},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [
                                [
                                    [39.0000, 51.0000],
                                    [39.0010, 51.0000],
                                    [39.0010, 51.0010],
                                    [39.0000, 51.0010],
                                    [39.0000, 51.0000],
                                ]
                            ],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "static-candidates",
            "--input",
            str(input_path),
            "--output-dir",
            str(tmp_path / "artifacts"),
            "--config",
            "configs/lipetsk_voronezh_mvp.yaml",
        ],
    )

    assert result.exit_code != 0
    assert "requires projected metric coordinates" in result.output
    assert "automatic reprojection is not implemented" in result.output


def test_static_candidates_rejects_declared_geographic_crs(tmp_path: Path) -> None:
    input_path = tmp_path / "crs84.geojson"
    input_path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"candidate_id": "declared-crs84"},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [
                                [
                                    [500000.0, 5600000.0],
                                    [500100.0, 5600000.0],
                                    [500100.0, 5600100.0],
                                    [500000.0, 5600100.0],
                                    [500000.0, 5600000.0],
                                ]
                            ],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "static-candidates",
            "--input",
            str(input_path),
            "--output-dir",
            str(tmp_path / "artifacts"),
            "--config",
            "configs/lipetsk_voronezh_mvp.yaml",
        ],
    )

    assert result.exit_code != 0
    assert "requires projected metric coordinates" in result.output
