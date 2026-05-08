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
    object_score = _text(score_row, "object_score")
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
    missing_data_flags = _json_list_text(score_row, "missing_data_flags")
    anomaly_dates = _json_list_text(score_row, "anomalous_dates")
    source_features = _json_list_text(score_row, "source_feature_names")
    source_layers = _json_list_text(score_row, "source_layer_ids")
    flags = _json_list_text(score_row, "flags")
    source_context = _source_context_lines(score_row)

    return "\n".join(
        [
            f"# Candidate Passport: {candidate_id}",
            "",
            "## Candidate Identity",
            "",
            f"- `candidate_id`: {candidate_id}",
            f"- `rank`: {rank}",
            f"- `priority_class`: {priority_class}",
            f"- `evidence_class`: {_text(score_row, 'evidence_class')}",
            f"- `passport_path`: {_text(score_row, 'passport_path')}",
            "",
            *source_context,
            "## Interpretation Guardrail",
            "",
            GUARDRAIL_TEXT,
            "",
            "Use this passport as a review artifact. Satellite ranking alone does not "
            "validate degassing.",
            "",
            "## Score Summary",
            "",
            f"- `object_score`: {object_score}",
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
            f"- `landcover_branch`: {_text(score_row, 'landcover_branch')}",
            f"- `dominant_landcover_branch`: {_text(score_row, 'dominant_landcover_branch')}",
            f"- `field_id`: {_text(score_row, 'field_id')}",
            f"- `anomalous_dates`: {anomaly_dates}",
            f"- `source_feature_names`: {source_features}",
            f"- `source_layer_ids`: {source_layers}",
            f"- `moisture_anomaly`: {_text(score_row, 'moisture_anomaly')}",
            f"- `vegetation_stress`: {_text(score_row, 'vegetation_stress')}",
            f"- `soil_brightness_bsi`: {_text(score_row, 'soil_brightness_bsi')}",
            f"- `thermal_anomaly`: {_text(score_row, 'thermal_anomaly')}",
            f"- `sar_anomaly`: {_text(score_row, 'sar_anomaly')}",
            f"- `persistence`: {_text(score_row, 'persistence')}",
            f"- `post_rain_drying`: {_text(score_row, 'post_rain_drying')}",
            f"- `geology_context`: {_text(score_row, 'geology_context')}",
            "",
            "Dynamic anomaly evidence is not produced by the geometry-only static "
            "pipeline. Soil drying or post-rain behavior must not be treated as "
            "standalone proof of degassing.",
            "",
            "## False-Positive Review",
            "",
            f"- `false_positive_flags`: {false_positive_flags}",
            f"- `false_positive_penalty`: {_text(score_row, 'false_positive_penalty')}",
            f"- `missing_data_flags`: {missing_data_flags}",
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


def _source_context_lines(score_row: Mapping[str, object]) -> list[str]:
    source_fields = [
        ("source_landcover_context", _text(score_row, "source_landcover_context")),
        ("source_morphology_type", _text(score_row, "source_morphology_type")),
        ("source_false_positive_risk", _text(score_row, "source_false_positive_risk")),
        ("source_notes", _text(score_row, "source_notes")),
    ]
    if not any(value for _, value in source_fields):
        return []
    return [
        "## Source Context",
        "",
        *[f"- `{field}`: {value}" for field, value in source_fields],
        "",
    ]
