from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

GUARDRAIL_TEXT = (
    "This is a ranked surface anomaly candidate compatible with possible deep "
    "degassing indicators. It is not direct H2 detection and does not prove active "
    "degassing without expert and field validation."
)


def render_candidate_passport(score_row: Mapping[str, object]) -> str:
    candidate_id = _text(score_row, "candidate_id")
    rank = _text(score_row, "rank")
    priority_class = _text(score_row, "priority_class")
    static_score = _text(score_row, "static_score")
    dynamic_score = _text(score_row, "dynamic_score")
    missing_dynamic = dynamic_score == ""
    missing_dynamic_text = (
        "missing_dynamic_evidence: dynamic score is unavailable for this static-only "
        "geometry artifact."
        if missing_dynamic
        else f"dynamic_score: {dynamic_score}"
    )
    false_positive_flags = _json_list_text(score_row, "false_positive_flags")
    flags = _json_list_text(score_row, "flags")

    return "\n".join(
        [
            f"# Candidate Passport: {candidate_id}",
            "",
            "## Candidate Identity",
            "",
            f"- `candidate_id`: {candidate_id}",
            f"- `rank`: {rank}",
            f"- `priority_class`: {priority_class}",
            "",
            "## Interpretation Guardrail",
            "",
            GUARDRAIL_TEXT,
            "",
            "Use this passport as a review artifact. Satellite ranking alone does not "
            "validate degassing.",
            "",
            "## Score Summary",
            "",
            f"- `static_score`: {static_score}",
            f"- `dynamic_score`: {dynamic_score or 'unavailable'}",
            f"- `dominant_evidence`: {_text(score_row, 'dominant_evidence')}",
            "",
            "The score summarizes review-priority evidence for this candidate. It is not "
            "a scientific proof score.",
            "",
            "## Static Evidence",
            "",
            f"- `morphology_type`: {_text(score_row, 'morphology_type')}",
            f"- `area_m2`: {_text(score_row, 'area_m2')}",
            f"- `diameter_m`: {_text(score_row, 'diameter_m')}",
            f"- `circularity`: {_text(score_row, 'circularity')}",
            f"- `elongation`: {_text(score_row, 'elongation')}",
            f"- `evidence`: {_text(score_row, 'evidence')}",
            f"- `flags`: {flags}",
            "",
            "## Dynamic Evidence",
            "",
            f"- `{missing_dynamic_text}`",
            "",
            "Dynamic anomaly evidence is not produced by the geometry-only static "
            "pipeline. Soil drying or post-rain behavior must not be treated as "
            "standalone proof of degassing.",
            "",
            "## False-Positive Review",
            "",
            f"- `false_positive_flags`: {false_positive_flags}",
            "- `false_positive_reason`: ",
            "- `reviewer_notes`: ",
            "",
            "Review plausible alternatives such as roads, field edges, quarries, "
            "clearcuts, wetlands, irrigation patterns, harvest patterns, clouds or "
            "shadows, gullies, water bodies and built-up objects.",
            "",
            "## Recommended Next Action",
            "",
            "- `recommended_follow_up`: expert image review and comparison with "
            "additional non-private layers before any field validation decision.",
            "",
        ]
    )


def write_candidate_passport(score_row: Mapping[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_candidate_passport(score_row), encoding="utf-8")


def _text(row: Mapping[str, object], key: str) -> str:
    value = row.get(key, "")
    if value is None:
        return ""
    return str(value)


def _json_list_text(row: Mapping[str, object], key: str) -> str:
    value = row.get(key, "")
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    if not isinstance(value, str) or value == "":
        return str(value or "")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return value
    if isinstance(parsed, list):
        return ", ".join(str(item) for item in parsed)
    return str(parsed)
