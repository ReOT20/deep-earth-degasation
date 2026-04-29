from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping
from pathlib import Path

LABELING_FIELDNAMES = [
    "candidate_id",
    "rank",
    "priority_class",
    "morphology_type",
    "static_score",
    "dynamic_score",
    "dominant_evidence",
    "false_positive_flags",
    "expert_label",
    "expert_confidence",
    "false_positive_reason",
    "reviewer_notes",
]


def score_row_to_labeling_row(score_row: Mapping[str, object]) -> dict[str, str]:
    row = {field: _text(score_row, field) for field in LABELING_FIELDNAMES}
    row["expert_label"] = ""
    row["expert_confidence"] = ""
    row["false_positive_reason"] = ""
    row["reviewer_notes"] = ""
    return row


def write_labeling_table(score_rows: Iterable[Mapping[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=LABELING_FIELDNAMES)
        writer.writeheader()
        for score_row in score_rows:
            writer.writerow(score_row_to_labeling_row(score_row))


def _text(row: Mapping[str, object], key: str) -> str:
    value = row.get(key, "")
    if value is None:
        return ""
    return str(value)
