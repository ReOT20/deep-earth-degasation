from __future__ import annotations

from pathlib import Path

from deep_earth_degasation.reports.passport import (
    render_candidate_passport,
    write_candidate_passport,
)


def _score_row() -> dict[str, object]:
    return {
        "rank": 1,
        "priority_class": "A",
        "candidate_id": "candidate-001",
        "morphology_type": "ring",
        "static_score": 0.95,
        "dynamic_score": "",
        "area_m2": 10_000.0,
        "diameter_m": 112.8,
        "circularity": 0.88,
        "elongation": 1.05,
        "evidence": '{"area":10000.0,"circularity":0.88}',
        "flags": '["built_up_risk"]',
        "dominant_evidence": "ring morphology",
        "false_positive_flags": '["built_up_risk"]',
    }


def test_render_candidate_passport_includes_static_candidate_context() -> None:
    markdown = render_candidate_passport(_score_row())

    assert "# Candidate Passport: candidate-001" in markdown
    assert "`rank`: 1" in markdown
    assert "`priority_class`: A" in markdown
    assert "`morphology_type`: ring" in markdown
    assert "`static_score`: 0.95" in markdown
    assert "ring morphology" in markdown
    assert "False-Positive Review" in markdown
    assert "built_up_risk" in markdown


def test_render_candidate_passport_includes_guardrail_language() -> None:
    markdown = render_candidate_passport(_score_row())

    assert "not direct H2 detection" in markdown
    assert "does not prove active degassing" in markdown
    assert "Satellite ranking alone does not validate degassing" in markdown


def test_blank_dynamic_score_renders_missing_dynamic_evidence() -> None:
    markdown = render_candidate_passport(_score_row())

    assert "`dynamic_score`: unavailable" in markdown
    assert "missing_dynamic_evidence" in markdown
    assert "geometry-only static pipeline" in markdown


def test_write_candidate_passport_writes_markdown_file(tmp_path: Path) -> None:
    output_path = tmp_path / "candidate-001.md"

    write_candidate_passport(_score_row(), output_path)

    assert output_path.exists()
    assert output_path.read_text(encoding="utf-8").startswith("# Candidate Passport: candidate-001")


def test_render_candidate_passport_avoids_confirmed_claims() -> None:
    markdown = render_candidate_passport(_score_row()).lower()

    assert "confirmed degassing" not in markdown
    assert "proven active degassing" not in markdown
    assert "direct h2 detection confirmed" not in markdown
