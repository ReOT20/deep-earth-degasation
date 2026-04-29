from __future__ import annotations

import csv
from pathlib import Path

from deep_earth_degasation.io.labeling import (
    LABELING_FIELDNAMES,
    score_row_to_labeling_row,
    write_labeling_table,
)


def _score_row() -> dict[str, object]:
    return {
        "candidate_id": "candidate-001",
        "rank": 1,
        "priority_class": "A",
        "morphology_type": "ring",
        "static_score": 0.95,
        "dynamic_score": "",
        "dominant_evidence": "ring morphology",
        "false_positive_flags": '["built_up_risk"]',
    }


def test_score_row_to_labeling_row_exports_required_fields() -> None:
    row = score_row_to_labeling_row(_score_row())

    assert list(row) == LABELING_FIELDNAMES
    assert row["candidate_id"] == "candidate-001"
    assert row["rank"] == "1"
    assert row["priority_class"] == "A"
    assert row["morphology_type"] == "ring"
    assert row["static_score"] == "0.95"
    assert row["dynamic_score"] == ""
    assert row["dominant_evidence"] == "ring morphology"
    assert row["false_positive_flags"] == '["built_up_risk"]'


def test_review_fields_are_empty_by_default() -> None:
    row = score_row_to_labeling_row(_score_row())

    assert row["expert_label"] == ""
    assert row["expert_confidence"] == ""
    assert row["false_positive_reason"] == ""
    assert row["reviewer_notes"] == ""


def test_no_negative_or_unlabeled_label_is_prefilled() -> None:
    row = score_row_to_labeling_row(
        {
            **_score_row(),
            "candidate_id": "candidate-unknown",
            "priority_class": "D",
            "false_positive_flags": "[]",
        }
    )

    assert row["expert_label"] == ""
    assert "negative" not in row.values()
    assert "hard_negative" not in row.values()
    assert "unlabeled" not in row.values()


def test_write_labeling_table_writes_csv_readable_by_dict_reader(tmp_path: Path) -> None:
    output_path = tmp_path / "labeling_table.csv"

    write_labeling_table([_score_row()], output_path)

    with output_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert rows[0]["candidate_id"] == "candidate-001"
    assert rows[0]["expert_label"] == ""
    assert rows[0]["reviewer_notes"] == ""
    assert rows[0]["false_positive_flags"] == '["built_up_risk"]'
