from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from deep_earth_degasation.cli import _static_detector_config
from deep_earth_degasation.config import load_config, resolved_config_dict


def test_lipetsk_voronezh_config_loads_with_typed_sections() -> None:
    config = load_config(Path("configs/lipetsk_voronezh_mvp.yaml"))

    assert config.project.name == "deep_earth_degasation_mvp"
    assert config.land_cover.classes["cropland"] == ["crops"]
    assert config.object_constraints.min_diameter_m == 40
    assert config.object_constraints.max_diameter_m == 1500
    assert config.static_ring_detector.circularity_min == 0.45
    assert config.dynamic_detector is not None
    assert config.outputs.candidates_top_n == 30


def test_dynamic_template_config_loads_with_planned_sections() -> None:
    config = load_config(Path("configs/lipetsk_voronezh_dynamic_mvp.template.yaml"))

    assert config.project.mode == "prepared_data_dynamic_mvp"
    assert config.prepared_data is not None
    assert config.prepared_data.reject_lonlat_for_metric_operations is True
    assert config.vectors["fields"].required is True
    assert config.features is not None
    assert config.anomaly_components["geology_context"].weight == 0.04
    assert config.validation is not None
    assert config.false_positive_filters is not None
    assert config.false_positive_filters.flag_excluded_zones is False
    assert config.false_positive_filters.flag_woody_patches is True
    assert config.false_positive_filters.woody_patch_buffer_m == 20
    assert config.scoring.penalties is not None
    assert config.scoring.penalties.woody_patch == 0.25
    assert config.outputs.export_resolved_config is True


def test_unknown_top_level_keys_fail(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, {"unexpected_section": True})

    with pytest.raises(ValidationError, match="unexpected_section"):
        load_config(config_path)


def test_unknown_nested_keys_fail(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, {"object_constraints": {"unexpected_key": 1}})

    with pytest.raises(ValidationError, match="unexpected_key"):
        load_config(config_path)


def test_project_metadata_and_provenance_are_free_form(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        {
            "project": {
                "metadata": {"operator": "local-review", "nested": {"allowed": True}},
                "provenance": {"bundle": "manual"},
            }
        },
    )

    config = load_config(config_path)

    assert config.project.metadata["nested"] == {"allowed": True}
    assert config.project.provenance["bundle"] == "manual"


def test_resolved_config_includes_defaults() -> None:
    config = load_config(Path("configs/lipetsk_voronezh_mvp.yaml"))
    resolved = resolved_config_dict(config)

    assert resolved["outputs"]["export_resolved_config"] is True
    assert resolved["project"]["metadata"] == {}
    assert resolved["static_ring_detector"]["ringness_min"] == 0.10


def test_static_detector_config_uses_static_ring_thresholds(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        {"static_ring_detector": {"circularity_min": 0.72}},
    )

    config = load_config(config_path)
    detector_config = _static_detector_config(config)

    assert detector_config.ellipse_min_circularity == 0.72
    assert detector_config.max_diameter_m == 1500


def test_object_size_threshold_decision_is_documented() -> None:
    contract = Path("docs/prepared_data_contract.md").read_text(encoding="utf-8")

    assert "1500" in contract
    assert "1000" in contract
    assert "phase-specific" in contract


def _write_config(tmp_path: Path, overrides: dict[str, Any]) -> Path:
    data = yaml.safe_load(Path("configs/lipetsk_voronezh_mvp.yaml").read_text(encoding="utf-8"))
    _deep_update(data, overrides)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return config_path


def _deep_update(target: dict[str, Any], overrides: dict[str, Any]) -> None:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
