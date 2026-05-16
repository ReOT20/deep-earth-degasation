from __future__ import annotations

import csv
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import box
from typer.testing import CliRunner

from deep_earth_degasation.cli import app
from deep_earth_degasation.io.label_migration import (
    LabelMigrationError,
    migrate_review_labels,
    write_label_migration_outputs,
)

CRS = "EPSG:32637"
runner = CliRunner()


def test_label_migration_exact_id_match() -> None:
    result = migrate_review_labels(
        old_label_rows=[_reviewed_row("candidate-a", label="weak_positive")],
        old_candidates=_candidates(["candidate-a"], [box(0, 0, 10, 10)]),
        new_label_rows=[_blank_row("candidate-a")],
        new_candidates=_candidates(["candidate-a"], [box(0, 0, 10, 10)]),
    )

    assert result.updated_label_rows[0]["expert_label"] == "weak_positive"
    assert result.updated_label_rows[0]["reviewer_notes"] == "reviewed candidate-a"
    assert result.migration_report_rows[0]["migration_status"] == "migrated"
    assert result.migration_report_rows[0]["review_required"] == "no"
    assert result.retired_label_rows == []


def test_label_migration_geometry_match_without_same_id() -> None:
    result = migrate_review_labels(
        old_label_rows=[_reviewed_row("old-candidate", label="weak_positive")],
        old_candidates=_candidates(["old-candidate"], [box(0, 0, 10, 10)]),
        new_label_rows=[_blank_row("new-candidate")],
        new_candidates=_candidates(["new-candidate"], [box(1, 0, 11, 10)]),
    )

    assert result.updated_label_rows[0]["candidate_id"] == "new-candidate"
    assert result.updated_label_rows[0]["expert_label"] == "weak_positive"
    assert result.migration_report_rows[0]["match_method"] == "field_iou"


def test_label_migration_appends_matched_candidate_missing_from_new_labels() -> None:
    result = migrate_review_labels(
        old_label_rows=[_reviewed_row("old-candidate", label="weak_positive")],
        old_candidates=_candidates(["old-candidate"], [box(0, 0, 10, 10)]),
        new_label_rows=[_blank_row("other-top-n-candidate")],
        new_candidates=_candidates(
            ["other-top-n-candidate", "new-candidate"],
            [box(100, 100, 110, 110), box(1, 0, 11, 10)],
        ),
    )

    appended = result.updated_label_rows[-1]
    assert appended["candidate_id"] == "new-candidate"
    assert appended["expert_label"] == "weak_positive"
    assert appended["migration_status"] == "migrated_missing_target_row"
    assert result.migration_report_rows[0]["migration_status"] == "migrated_missing_target_row"
    assert result.migration_report_rows[0]["review_required"] == "yes"
    assert result.retired_label_rows == []


def test_label_migration_does_not_attach_label_to_changed_exact_id() -> None:
    result = migrate_review_labels(
        old_label_rows=[_reviewed_row("candidate-a", label="weak_positive")],
        old_candidates=_candidates(["candidate-a"], [box(0, 0, 10, 10)]),
        new_label_rows=[_blank_row("candidate-a")],
        new_candidates=_candidates(["candidate-a"], [box(100, 100, 110, 110)]),
    )

    assert result.updated_label_rows[0]["expert_label"] == ""
    assert result.migration_report_rows[0]["migration_status"] == "geometry_changed"
    assert result.migration_report_rows[0]["review_required"] == "yes"
    assert result.retired_label_rows[0]["candidate_id"] == "candidate-a"


def test_label_migration_retires_unmatched_old_label() -> None:
    result = migrate_review_labels(
        old_label_rows=[_reviewed_row("retired-candidate", label="hard_negative")],
        old_candidates=_candidates(["retired-candidate"], [box(0, 0, 10, 10)]),
        new_label_rows=[_blank_row("new-candidate")],
        new_candidates=_candidates(["new-candidate"], [box(100, 100, 110, 110)]),
    )

    assert result.updated_label_rows[0]["expert_label"] == ""
    assert result.migration_report_rows[0]["migration_status"] == "retired"
    assert result.retired_label_rows[0]["expert_label"] == "hard_negative"
    assert result.new_candidate_rows[0]["candidate_id"] == "new-candidate"


def test_label_migration_marks_split_or_merge_review_required() -> None:
    result = migrate_review_labels(
        old_label_rows=[_reviewed_row("old-candidate", label="weak_positive")],
        old_candidates=_candidates(["old-candidate"], [box(0, 0, 10, 10)]),
        new_label_rows=[_blank_row("new-a"), _blank_row("new-b")],
        new_candidates=_candidates(
            ["new-a", "new-b"],
            [box(0, 0, 10, 10), box(0, 0, 10, 10)],
        ),
    )

    assert all(row["expert_label"] == "" for row in result.updated_label_rows)
    assert result.migration_report_rows[0]["migration_status"] == "ambiguous"
    assert result.migration_report_rows[0]["review_required"] == "yes"
    assert result.retired_label_rows[0]["migration_status"] == "ambiguous"


def test_label_migration_rejects_lonlat_candidate_crs() -> None:
    with pytest.raises(LabelMigrationError, match="projected metre CRS"):
        migrate_review_labels(
            old_label_rows=[_reviewed_row("candidate-a", label="weak_positive")],
            old_candidates=_candidates(["candidate-a"], [box(0, 0, 1, 1)], crs="EPSG:4326"),
            new_label_rows=[_blank_row("candidate-a")],
            new_candidates=_candidates(["candidate-a"], [box(0, 0, 1, 1)], crs="EPSG:4326"),
        )


def test_label_migration_rejects_mismatched_projected_crs() -> None:
    with pytest.raises(LabelMigrationError, match="same CRS"):
        migrate_review_labels(
            old_label_rows=[_reviewed_row("candidate-a", label="weak_positive")],
            old_candidates=_candidates(["candidate-a"], [box(0, 0, 10, 10)], crs="EPSG:32637"),
            new_label_rows=[_blank_row("candidate-a")],
            new_candidates=_candidates(["candidate-a"], [box(0, 0, 10, 10)], crs="EPSG:3857"),
        )


def test_write_label_migration_outputs(tmp_path: Path) -> None:
    result = migrate_review_labels(
        old_label_rows=[_reviewed_row("candidate-a", label="weak_positive")],
        old_candidates=_candidates(["candidate-a"], [box(0, 0, 10, 10)]),
        new_label_rows=[_blank_row("candidate-a")],
        new_candidates=_candidates(["candidate-a"], [box(0, 0, 10, 10)]),
    )

    write_label_migration_outputs(result, tmp_path)

    assert (tmp_path / "updated_labeling_table.csv").exists()
    assert (tmp_path / "label_migration_report.csv").exists()
    assert (tmp_path / "retired_labels.csv").exists()
    assert (tmp_path / "new_candidates.csv").exists()
    with (tmp_path / "label_migration_report.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert rows[0]["migration_status"] == "migrated"


def test_migrate_labels_cli_writes_review_refresh_outputs(tmp_path: Path) -> None:
    old_labels = tmp_path / "old_labels.csv"
    new_labels = tmp_path / "new_labels.csv"
    old_candidates = tmp_path / "old_candidates.geojson"
    new_candidates = tmp_path / "new_candidates.geojson"
    output_dir = tmp_path / "migration"
    _write_csv(old_labels, [_reviewed_row("candidate-a", label="weak_positive")])
    _write_csv(new_labels, [_blank_row("candidate-a")])
    _candidates(["candidate-a"], [box(0, 0, 10, 10)]).to_file(old_candidates, driver="GeoJSON")
    _candidates(["candidate-a"], [box(0, 0, 10, 10)]).to_file(new_candidates, driver="GeoJSON")

    result = runner.invoke(
        app,
        [
            "migrate-labels",
            "--old-labels",
            str(old_labels),
            "--old-candidates",
            str(old_candidates),
            "--new-labels",
            str(new_labels),
            "--new-candidates",
            str(new_candidates),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert (
        f"updated_labeling_table_csv={output_dir / 'updated_labeling_table.csv'}" in result.output
    )
    rows = _read_csv(output_dir / "updated_labeling_table.csv")
    assert rows[0]["expert_label"] == "weak_positive"


def _candidates(
    candidate_ids: list[str],
    geometries: list[object],
    *,
    field_id: str = "field-1",
    crs: str = CRS,
) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "candidate_id": candidate_ids,
            "field_id": [field_id] * len(candidate_ids),
            "geometry": geometries,
        },
        geometry="geometry",
        crs=crs,
    )


def _reviewed_row(candidate_id: str, *, label: str) -> dict[str, str]:
    return {
        **_blank_row(candidate_id),
        "expert_label": label,
        "expert_confidence": "3",
        "false_positive_reason": "field_edge" if label == "hard_negative" else "",
        "reviewer_notes": f"reviewed {candidate_id}",
    }


def _blank_row(candidate_id: str) -> dict[str, str]:
    return {
        "candidate_id": candidate_id,
        "rank": "1",
        "field_id": "field-1",
        "expert_label": "",
        "expert_confidence": "",
        "false_positive_reason": "",
        "reviewer_notes": "",
    }


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))
