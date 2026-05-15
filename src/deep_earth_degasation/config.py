from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


Metadata = dict[str, Any]


class ProjectConfig(StrictModel):
    name: str
    version: str
    mode: str | None = None
    claim_guardrail: str | None = None
    metadata: Metadata = Field(default_factory=dict)
    provenance: Metadata = Field(default_factory=dict)


class AOIConfig(StrictModel):
    name: str
    input_geojson: str
    metric_crs: str | None = None
    metadata: Metadata = Field(default_factory=dict)
    provenance: Metadata = Field(default_factory=dict)


class TimeConfig(StrictModel):
    start_year: int
    end_year: int
    vegetation_months: list[int]
    bare_soil_months: list[int]
    minimum_repeated_seasons: PositiveInt | None = None


class LandCoverConfig(StrictModel):
    primary_branch: str | None = None
    classes: dict[str, list[str]]
    mixed_candidate_min_classes: PositiveInt


class ObjectConstraints(StrictModel):
    min_diameter_m: float = 40.0
    max_diameter_m: float = 1500.0
    min_area_ha: float = 0.1
    max_area_ha: float = 200.0
    min_distance_to_field_edge_m: float = 20.0

    @model_validator(mode="after")
    def validate_ranges(self) -> ObjectConstraints:
        if self.min_diameter_m <= 0 or self.max_diameter_m <= 0:
            raise ValueError("Object diameter constraints must be positive.")
        if self.min_diameter_m > self.max_diameter_m:
            raise ValueError("min_diameter_m must be <= max_diameter_m.")
        if self.min_area_ha <= 0 or self.max_area_ha <= 0:
            raise ValueError("Object area constraints must be positive.")
        if self.min_area_ha > self.max_area_ha:
            raise ValueError("min_area_ha must be <= max_area_ha.")
        return self


class PreparedDataConfig(StrictModel):
    raster_stack_manifest: str
    reject_lonlat_for_metric_operations: bool = True
    allow_reprojection: bool = False
    allow_resampling: bool = False


class VectorLayerConfig(StrictModel):
    path: str
    required: bool = False
    id_field: str | None = None
    role: str | None = None
    crs: str | None = None


class Sentinel2Config(StrictModel):
    cloud_threshold: float = 30.0
    indices: list[str]
    optional_indices: list[str] = Field(default_factory=list)


class Sentinel1Config(StrictModel):
    features: list[str]


class LandsatConfig(StrictModel):
    features: list[str]


class WeatherConfig(StrictModel):
    features: list[str] = Field(default_factory=list)


class FeatureConfig(StrictModel):
    sentinel2: Sentinel2Config
    sentinel1: Sentinel1Config
    landsat: LandsatConfig
    weather: WeatherConfig | None = None


class NormalizationConfig(StrictModel):
    cropland_method: str
    forest_method: str | None = None
    center: Literal["median"]
    scale: Literal["MAD"]
    min_valid_pixels_per_field_date: PositiveInt | None = None
    mad_epsilon: PositiveFloat | None = None
    missing_field_policy: str | None = None


class StaticRingDetectorConfig(StrictModel):
    enabled: bool = True
    circularity_min: float = Field(default=0.45, ge=0.0, le=1.0)
    ringness_min: float = Field(default=0.10, ge=0.0, le=1.0)
    multiscale_radii_m: list[PositiveFloat] = Field(default_factory=list)


class DynamicDetectorConfig(StrictModel):
    anomaly_percentile: float = Field(default=95.0, ge=0.0, le=100.0)
    support_percentile: float | None = Field(default=None, ge=0.0, le=100.0)
    min_support_pixels: PositiveInt = 1
    min_valid_observations_per_season: PositiveInt = 4
    min_repeated_seasons: PositiveInt = 2
    connected_components: bool = True
    morphology_filter: bool = True
    merge_distance_m: float | None = None
    merge_across_dates: bool | None = None


class FalsePositiveFiltersConfig(StrictModel):
    use_penalties: bool = True
    flag_roads: bool = True
    flag_water: bool = True
    flag_builtup: bool | None = None
    flag_built_up: bool | None = None
    flag_field_edges: bool = True
    flag_linear_objects: bool = True
    flag_cloud_shadows: bool = True
    flag_harvest_patterns: bool = True
    flag_forest_clearcuts: bool | None = None
    flag_forest_roads: bool | None = None
    flag_wetlands: bool | None = None
    flag_irrigation: bool | None = None
    flag_quarries: bool = True
    road_buffer_m: float | None = None
    water_buffer_m: float | None = None
    builtup_buffer_m: float | None = None
    field_edge_buffer_m: float | None = None
    max_elongation_without_penalty: float | None = None
    missing_context_policy: str | None = None


class ScoreWeights(StrictModel):
    moisture_weight: float | None = None
    vegetation_weight: float | None = None
    brightness_weight: float | None = None
    thermal_weight: float | None = None
    sar_weight: float | None = None
    morphology_weight: float | None = None
    persistence_weight: float | None = None
    post_rain_weight: float | None = None
    geology_weight: float | None = None
    canopy_structure_weight: float | None = None
    canopy_moisture_weight: float | None = None
    dem_weight: float | None = None
    cluster_adjacency_weight: float | None = None
    object_integrity_weight: float | None = None
    cropland_signal_weight: float | None = None
    forest_signal_weight: float | None = None


class PriorityThresholds(StrictModel):
    A: float = Field(ge=0.0, le=1.0)
    B: float = Field(ge=0.0, le=1.0)
    C: float = Field(ge=0.0, le=1.0)
    D: float = Field(ge=0.0, le=1.0)


class PenaltyConfig(StrictModel):
    field_edge: float | None = None
    road: float | None = None
    water: float | None = None
    built_up: float | None = None
    quarry: float | None = None
    linear_object: float | None = None
    cloud_shadow: float | None = None
    harvest_pattern: float | None = None


class ScoringConfig(StrictModel):
    cropland: ScoreWeights
    forest: ScoreWeights | None = None
    mixed: ScoreWeights | None = None
    priority_thresholds: PriorityThresholds | None = None
    penalties: PenaltyConfig | None = None


class AnomalyComponentConfig(StrictModel):
    inputs: list[str]
    high_score_means: str
    weight: float


class ValidationConfig(StrictModel):
    top_n: PositiveInt = 20
    known_site_recall_top_n: list[PositiveInt] = Field(default_factory=list)
    expert_precision_top_n: PositiveInt | None = None
    export_false_positive_counts: bool = True
    do_not_treat_unlabeled_as_negative: bool = True


class OutputConfig(StrictModel):
    output_dir: str | None = None
    candidates_top_n: int = 30
    export_geojson: bool = True
    export_csv: bool = True
    generate_passports: bool = True
    generate_quicklooks: bool | None = None
    export_time_series: bool | None = None
    export_validation_summary: bool | None = None
    export_resolved_config: bool = True


class MVPConfig(StrictModel):
    project: ProjectConfig
    aoi: AOIConfig
    time: TimeConfig
    land_cover: LandCoverConfig
    object_constraints: ObjectConstraints = Field(default_factory=ObjectConstraints)
    prepared_data: PreparedDataConfig | None = None
    vectors: dict[str, VectorLayerConfig] = Field(default_factory=dict)
    sentinel2: Sentinel2Config | None = None
    sentinel1: Sentinel1Config | None = None
    landsat: LandsatConfig | None = None
    features: FeatureConfig | None = None
    normalization: NormalizationConfig | None = None
    static_ring_detector: StaticRingDetectorConfig = Field(default_factory=StaticRingDetectorConfig)
    dynamic_detector: DynamicDetectorConfig | None = None
    false_positive_filters: FalsePositiveFiltersConfig | None = None
    anomaly_components: dict[str, AnomalyComponentConfig] = Field(default_factory=dict)
    scoring: ScoringConfig
    validation: ValidationConfig | None = None
    outputs: OutputConfig = Field(default_factory=OutputConfig)


def load_config(path: str | Path) -> MVPConfig:
    """Load a YAML project configuration file."""
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return MVPConfig.model_validate(data)


def resolved_config_dict(config: MVPConfig) -> dict[str, Any]:
    """Return a JSON-serializable config with defaults made explicit."""
    return config.model_dump(mode="json")
