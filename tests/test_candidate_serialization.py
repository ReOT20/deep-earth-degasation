from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from shapely import affinity
from shapely.geometry import Point, Polygon

from deep_earth_degasation.io.candidates import (
    static_candidate_to_score_row,
    write_candidate_artifacts,
    write_candidate_scores_csv,
    write_candidates_geojson,
)
from deep_earth_degasation.morphology.static_detector import extract_static_candidates


def test_writes_valid_candidate_geojson(tmp_path: Path) -> None:
    circle = Point(0.0, 0.0).buffer(50.0, quad_segs=32)
    candidate = extract_static_candidates([circle], candidate_ids=["stable-id"])[0]
    output_path = tmp_path / "candidates.geojson"

    write_candidates_geojson([candidate], output_path)

    data = json.loads(output_path.read_text(encoding="utf-8"))
    feature = data["features"][0]
    assert data["type"] == "FeatureCollection"
    assert feature["type"] == "Feature"
    assert feature["properties"]["candidate_id"] == "stable-id"
    assert feature["properties"]["morphology_type"] == "circle"
    assert feature["properties"]["static_score"] > 0.9
    assert feature["properties"]["area_m2"] == pytest.approx(candidate.shape_metrics.area)
    assert feature["properties"]["diameter_m"] == pytest.approx(
        candidate.shape_metrics.equivalent_diameter
    )
    assert feature["geometry"]["type"] == "Polygon"


def test_writes_candidate_scores_csv_that_csv_tools_can_read(tmp_path: Path) -> None:
    ring = Polygon(
        [(-50.0, -50.0), (50.0, -50.0), (50.0, 50.0), (-50.0, 50.0)],
        holes=[[(-15.0, -15.0), (15.0, -15.0), (15.0, 15.0), (-15.0, 15.0)]],
    )
    candidate = extract_static_candidates([ring], candidate_ids=["ring-1"])[0]
    output_path = tmp_path / "candidate_scores.csv"

    write_candidate_scores_csv([candidate], output_path)

    with output_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert rows[0]["rank"] == "1"
    assert rows[0]["candidate_id"] == "ring-1"
    assert rows[0]["morphology_type"] == "ring"
    assert rows[0]["priority_class"] == "A"
    assert rows[0]["dynamic_score"] == ""
    assert rows[0]["dominant_evidence"] == "ring morphology"


def test_candidate_scores_csv_ranks_candidates_by_static_score(tmp_path: Path) -> None:
    irregular = Polygon(
        [
            (0.0, 0.0),
            (120.0, 0.0),
            (95.0, 35.0),
            (140.0, 90.0),
            (45.0, 65.0),
            (0.0, 120.0),
        ]
    )
    ring = Polygon(
        [(-50.0, -50.0), (50.0, -50.0), (50.0, 50.0), (-50.0, 50.0)],
        holes=[[(-15.0, -15.0), (15.0, -15.0), (15.0, 15.0), (-15.0, 15.0)]],
    )
    candidates = extract_static_candidates(
        [irregular, ring], candidate_ids=["low-score-first", "high-score-second"]
    )
    output_path = tmp_path / "candidate_scores.csv"

    write_candidate_scores_csv(candidates, output_path)

    with output_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert rows[0]["rank"] == "1"
    assert rows[0]["candidate_id"] == "high-score-second"
    assert rows[1]["rank"] == "2"
    assert rows[1]["candidate_id"] == "low-score-first"


def test_circle_ring_and_ellipse_candidates_serialize_expected_morphology() -> None:
    circle = Point(0.0, 0.0).buffer(50.0, quad_segs=32)
    ellipse = affinity.scale(circle, xfact=2.0, yfact=1.0, origin=(0.0, 0.0))
    ring = Polygon(
        [(-50.0, -50.0), (50.0, -50.0), (50.0, 50.0), (-50.0, 50.0)],
        holes=[[(-15.0, -15.0), (15.0, -15.0), (15.0, 15.0), (-15.0, 15.0)]],
    )
    candidates = extract_static_candidates(
        [circle, ring, ellipse], candidate_ids=["circle-1", "ring-1", "ellipse-1"]
    )

    rows = [
        static_candidate_to_score_row(candidate, rank)
        for rank, candidate in enumerate(candidates, start=1)
    ]

    assert [row["candidate_id"] for row in rows] == ["circle-1", "ring-1", "ellipse-1"]
    assert [row["morphology_type"] for row in rows] == ["circle", "ring", "ellipse"]
    assert [row["rank"] for row in rows] == [1, 2, 3]


def test_write_candidate_artifacts_creates_expected_files(tmp_path: Path) -> None:
    candidate = extract_static_candidates([Point(0.0, 0.0).buffer(50.0)])[0]

    paths = write_candidate_artifacts([candidate], tmp_path / "artifacts")

    assert paths.candidates_geojson.exists()
    assert paths.candidate_scores_csv.exists()
    assert paths.candidates_geojson.name == "candidates.geojson"
    assert paths.candidate_scores_csv.name == "candidate_scores.csv"


def test_evidence_and_flags_are_deterministic_json_strings() -> None:
    geometry = Point(0.0, 0.0).buffer(50.0)
    candidate = extract_static_candidates(
        [geometry], candidate_ids=["built-up-1"], landcover_context="built_up"
    )[0]

    row = static_candidate_to_score_row(candidate, rank=1)

    assert json.loads(str(row["evidence"])) == {
        "area": candidate.evidence["area"],
        "circularity": candidate.evidence["circularity"],
        "elongation": candidate.evidence["elongation"],
        "equivalent_diameter": candidate.evidence["equivalent_diameter"],
    }
    assert row["flags"] == '["built_up_risk"]'
    assert row["false_positive_flags"] == '["built_up_risk"]'
