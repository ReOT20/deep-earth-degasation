from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class ObjectConstraints(BaseModel):
    min_diameter_m: float = 40.0
    max_diameter_m: float = 1500.0
    min_area_ha: float = 0.1
    max_area_ha: float = 200.0
    min_distance_to_field_edge_m: float = 20.0


class OutputConfig(BaseModel):
    candidates_top_n: int = 30
    export_geojson: bool = True
    export_csv: bool = True
    generate_passports: bool = True


class MVPConfig(BaseModel):
    project: dict[str, Any] = Field(default_factory=dict)
    aoi: dict[str, Any] = Field(default_factory=dict)
    time: dict[str, Any] = Field(default_factory=dict)
    object_constraints: ObjectConstraints = Field(default_factory=ObjectConstraints)
    scoring: dict[str, Any] = Field(default_factory=dict)
    outputs: OutputConfig = Field(default_factory=OutputConfig)


def load_config(path: str | Path) -> MVPConfig:
    """Load a YAML project configuration file."""
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return MVPConfig.model_validate(data)
