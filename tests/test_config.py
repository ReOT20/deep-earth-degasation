from __future__ import annotations

from pathlib import Path

from deep_earth_degasation.config import load_config


def test_lipetsk_voronezh_config_loads() -> None:
    config = load_config(Path("configs/lipetsk_voronezh_mvp.yaml"))

    assert config.project["name"] == "deep_earth_degasation_mvp"
    assert config.object_constraints.min_diameter_m == 40
    assert config.object_constraints.max_diameter_m == 1500
    assert config.outputs.candidates_top_n == 30
