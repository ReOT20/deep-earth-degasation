from __future__ import annotations

import csv
from pathlib import Path

import pytest
from typer.testing import CliRunner

from deep_earth_degasation.cli import app
from deep_earth_degasation.learning.dataset import (
    LEARNING_FIELDNAMES,
    LearningDatasetError,
    build_learning_dataset_rows,
    write_learning_dataset,
)

runner = CliRunner()


def test_build_learning_rows_preserves_canonical_labels_and_roles() -> None:
    rows = build_learning_dataset_rows(
        [_score_row("candidate-a"), _score_row("candidate-b"), _score_row("candidate-c")],
        [
            {
                "candidate_id": "candidate-a",
                "expert_label": "weak_positive",
                "expert_confidence": "2",
                "reviewer_notes": "plausible anomaly",
            },
            {
                "candidate_id": "candidate-b",
                "expert_label": "hard_negative",
                "false_positive_reason": "field_edge",
            },
            {"candidate_id": "candidate-c", "expert_label": "uncertain"},
        ],
        run_id="run-1",
        feature_snapshot_id="manifest-1",
        aoi_name="synthetic",
        geometry_ref="candidates.geojson",
    )

    assert [row["model_label"] for row in rows] == [
        "weak_positive",
        "hard_negative",
        "uncertain",
    ]
    assert [row["pu_role"] for row in rows] == [
        "weak_positive_seed",
        "hard_negative_control",
        "exclude",
    ]
    assert rows[0]["label_source"] == "expert"
    assert rows[0]["label_confidence"] == "2"
    assert rows[0]["run_id"] == "run-1"
    assert rows[0]["feature_snapshot_id"] == "manifest-1"
    assert rows[0]["aoi_name"] == "synthetic"
    assert rows[0]["geometry_ref"] == "candidates.geojson"
    assert rows[0]["split_group"] == "field:field-1"
    assert rows[2]["use_for_training"] == "no"


def test_blank_labels_become_unlabeled_pool() -> None:
    rows = build_learning_dataset_rows([_score_row("candidate-a")])

    assert rows[0]["model_label"] == "unlabeled"
    assert rows[0]["legacy_label"] == ""
    assert rows[0]["pu_role"] == "unlabeled_pool"
    assert rows[0]["label_source"] == "detector_export"
    assert rows[0]["use_for_training"] == "yes"


def test_non_canonical_labels_are_rejected() -> None:
    with pytest.raises(LearningDatasetError, match="unsupported expert_label"):
        build_learning_dataset_rows(
            [_score_row("candidate-a")],
            [{"candidate_id": "candidate-a", "expert_label": "A_field"}],
        )

    with pytest.raises(LearningDatasetError, match="unsupported expert_label"):
        build_learning_dataset_rows(
            [_score_row("candidate-b")],
            [{"candidate_id": "candidate-b", "expert_label": "B"}],
        )


def test_hard_negative_requires_false_positive_reason() -> None:
    with pytest.raises(LearningDatasetError, match="requires false_positive_reason"):
        build_learning_dataset_rows(
            [_score_row("candidate-a")],
            [{"candidate_id": "candidate-a", "expert_label": "hard_negative"}],
        )


def test_write_learning_dataset_uses_deterministic_fields(tmp_path: Path) -> None:
    output_path = tmp_path / "learning_dataset.csv"

    write_learning_dataset([_score_row("candidate-a")], output_path)

    with output_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    assert reader.fieldnames == LEARNING_FIELDNAMES
    assert rows[0]["candidate_id"] == "candidate-a"
    assert rows[0]["model_label"] == "unlabeled"


def test_export_learning_dataset_cli_merges_reviewed_labels(tmp_path: Path) -> None:
    scores_path = tmp_path / "candidate_scores.csv"
    labels_path = tmp_path / "reviewed_labels.csv"
    output_path = tmp_path / "learning_dataset.csv"
    _write_csv(scores_path, [_score_row("candidate-a")])
    _write_csv(
        labels_path,
        [
            {
                "candidate_id": "candidate-a",
                "expert_label": "weak_positive",
                "expert_confidence": "3",
                "reviewer_notes": "clear index anomaly",
            }
        ],
    )

    result = runner.invoke(
        app,
        [
            "export-learning-dataset",
            "--scores",
            str(scores_path),
            "--labels",
            str(labels_path),
            "--output",
            str(output_path),
            "--run-id",
            "run-1",
            "--feature-snapshot-id",
            "snapshot-1",
            "--aoi-name",
            "synthetic",
        ],
    )

    assert result.exit_code == 0, result.output
    assert f"learning_dataset_csv={output_path}" in result.output
    with output_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert rows[0]["model_label"] == "weak_positive"
    assert rows[0]["pu_role"] == "weak_positive_seed"
    assert rows[0]["label_confidence"] == "3"
    assert rows[0]["run_id"] == "run-1"
    assert rows[0]["feature_snapshot_id"] == "snapshot-1"
    assert rows[0]["aoi_name"] == "synthetic"
    assert rows[0]["geometry_ref"] == str(scores_path.with_name("candidates.geojson"))


def test_export_learning_dataset_cli_accepts_explicit_geometry_ref(tmp_path: Path) -> None:
    scores_path = tmp_path / "candidate_scores.csv"
    output_path = tmp_path / "learning_dataset.csv"
    _write_csv(scores_path, [_score_row("candidate-a")])

    result = runner.invoke(
        app,
        [
            "export-learning-dataset",
            "--scores",
            str(scores_path),
            "--output",
            str(output_path),
            "--geometry-ref",
            "review/candidates.geojson",
        ],
    )

    assert result.exit_code == 0, result.output
    with output_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert rows[0]["geometry_ref"] == "review/candidates.geojson"


def _score_row(candidate_id: str) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "rank": 1,
        "priority_class": "C",
        "field_id": "field-1",
        "landcover_branch": "cropland",
        "dominant_landcover_branch": "cropland",
        "object_score": 0.42,
        "static_score": 0.0,
        "dynamic_score": 0.7,
        "moisture_anomaly": 3.0,
        "vegetation_stress": 2.0,
        "soil_brightness_bsi": 1.0,
        "support_pixel_count": 12,
        "dominant_evidence": "cropland dynamic anomaly",
        "false_positive_flags": '["field_edge_risk"]',
        "missing_data_flags": '["missing_sentinel1_VV"]',
        "flags": '["dynamic_only"]',
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
