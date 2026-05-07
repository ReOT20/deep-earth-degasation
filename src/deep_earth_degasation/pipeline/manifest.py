from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ManifestError(ValueError):
    """Raised when a prepared-data manifest is invalid."""


@dataclass(frozen=True)
class RunManifestSpec:
    id: str
    description: str | None
    created_by: str | None
    notes: str | None


@dataclass(frozen=True)
class AOISpec:
    path: Path
    crs: str


@dataclass(frozen=True)
class VectorLayerSpec:
    path: Path
    crs: str
    role: str
    required: bool
    id_field: str | None = None


@dataclass(frozen=True)
class RasterLayerSpec:
    id: str
    sensor: str
    feature_name: str
    date: str | None
    path: Path
    crs: str
    resolution_m: float
    nodata: float | None
    role: str
    quality_mask_path: Path | None = None


@dataclass(frozen=True)
class QualitySpec:
    reject_mismatched_crs: bool = True
    reject_mismatched_grid: bool = True
    allow_resampling: bool = False
    min_valid_observations_per_season: int | None = None


@dataclass(frozen=True)
class PreparedStackManifest:
    path: Path
    run: RunManifestSpec
    crs: str
    resolution_m: float
    aoi: AOISpec
    vectors: dict[str, VectorLayerSpec]
    raster_layers: tuple[RasterLayerSpec, ...]
    quality: QualitySpec
    notes: tuple[str, ...]


def load_prepared_stack_manifest(path: str | Path) -> PreparedStackManifest:
    """Load and strictly validate a prepared raster stack manifest."""
    manifest_path = Path(path)
    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ManifestError("Prepared stack manifest must be a mapping.")

    _require_keys(
        data,
        required={"run", "crs", "resolution_m", "aoi", "vectors", "raster_layers", "quality"},
        optional={"notes"},
        section="manifest",
    )
    base_dir = manifest_path.parent
    layers = tuple(_parse_raster_layers(data["raster_layers"], base_dir=base_dir))
    _reject_duplicate_layer_ids(layers)

    return PreparedStackManifest(
        path=manifest_path,
        run=_parse_run(data["run"]),
        crs=str(data["crs"]),
        resolution_m=float(data["resolution_m"]),
        aoi=_parse_aoi(data["aoi"], base_dir=base_dir),
        vectors=_parse_vectors(data["vectors"], base_dir=base_dir),
        raster_layers=layers,
        quality=_parse_quality(data["quality"]),
        notes=tuple(str(note) for note in data.get("notes", [])),
    )


def _parse_run(value: object) -> RunManifestSpec:
    data = _require_mapping(value, "run")
    _require_keys(
        data, required={"id"}, optional={"description", "created_by", "notes"}, section="run"
    )
    return RunManifestSpec(
        id=str(data["id"]),
        description=_optional_text(data.get("description")),
        created_by=_optional_text(data.get("created_by")),
        notes=_optional_text(data.get("notes")),
    )


def _parse_aoi(value: object, *, base_dir: Path) -> AOISpec:
    data = _require_mapping(value, "aoi")
    _require_keys(data, required={"path", "crs"}, optional=set(), section="aoi")
    return AOISpec(path=_resolve_path(base_dir, data["path"]), crs=str(data["crs"]))


def _parse_vectors(value: object, *, base_dir: Path) -> dict[str, VectorLayerSpec]:
    data = _require_mapping(value, "vectors")
    vectors: dict[str, VectorLayerSpec] = {}
    for name, raw_layer in data.items():
        layer = _require_mapping(raw_layer, f"vectors.{name}")
        _require_keys(
            layer,
            required={"path", "crs", "role", "required"},
            optional={"id_field"},
            section=f"vectors.{name}",
        )
        vectors[str(name)] = VectorLayerSpec(
            path=_resolve_path(base_dir, layer["path"]),
            crs=str(layer["crs"]),
            role=str(layer["role"]),
            required=_parse_bool(layer["required"], f"vectors.{name}.required"),
            id_field=_optional_text(layer.get("id_field")),
        )
    return vectors


def _parse_raster_layers(value: object, *, base_dir: Path) -> list[RasterLayerSpec]:
    if not isinstance(value, list):
        raise ManifestError("raster_layers must be a list.")

    layers: list[RasterLayerSpec] = []
    for index, raw_layer in enumerate(value, start=1):
        layer = _require_mapping(raw_layer, f"raster_layers[{index}]")
        _require_keys(
            layer,
            required={
                "id",
                "sensor",
                "feature_name",
                "path",
                "crs",
                "resolution_m",
                "nodata",
                "role",
            },
            optional={"date", "quality_mask_path"},
            section=f"raster_layers[{index}]",
        )
        layers.append(
            RasterLayerSpec(
                id=str(layer["id"]),
                sensor=str(layer["sensor"]),
                feature_name=str(layer["feature_name"]),
                date=_optional_text(layer.get("date")),
                path=_resolve_path(base_dir, layer["path"]),
                crs=str(layer["crs"]),
                resolution_m=float(layer["resolution_m"]),
                nodata=None if layer["nodata"] is None else float(layer["nodata"]),
                role=str(layer["role"]),
                quality_mask_path=(
                    None
                    if layer.get("quality_mask_path") is None
                    else _resolve_path(base_dir, layer["quality_mask_path"])
                ),
            )
        )
    return layers


def _parse_quality(value: object) -> QualitySpec:
    data = _require_mapping(value, "quality")
    _require_keys(
        data,
        required=set(),
        optional={
            "reject_mismatched_crs",
            "reject_mismatched_grid",
            "allow_resampling",
            "min_valid_observations_per_season",
        },
        section="quality",
    )
    return QualitySpec(
        reject_mismatched_crs=_parse_bool(
            data.get("reject_mismatched_crs", True), "quality.reject_mismatched_crs"
        ),
        reject_mismatched_grid=_parse_bool(
            data.get("reject_mismatched_grid", True), "quality.reject_mismatched_grid"
        ),
        allow_resampling=_parse_bool(
            data.get("allow_resampling", False), "quality.allow_resampling"
        ),
        min_valid_observations_per_season=(
            None
            if data.get("min_valid_observations_per_season") is None
            else int(data["min_valid_observations_per_season"])
        ),
    )


def _reject_duplicate_layer_ids(layers: tuple[RasterLayerSpec, ...]) -> None:
    seen: set[str] = set()
    for layer in layers:
        if layer.id in seen:
            raise ManifestError(f"Duplicate raster layer id {layer.id!r}.")
        seen.add(layer.id)


def _require_mapping(value: object, section: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ManifestError(f"{section} must be a mapping.")
    return value


def _require_keys(
    data: dict[str, Any],
    *,
    required: set[str],
    optional: set[str],
    section: str,
) -> None:
    keys = set(data)
    missing = required - keys
    if missing:
        raise ManifestError(f"{section} is missing required keys: {sorted(missing)}.")
    unknown = keys - required - optional
    if unknown:
        raise ManifestError(f"{section} contains unknown keys: {sorted(unknown)}.")


def _resolve_path(base_dir: Path, value: object) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path

    manifest_relative = (base_dir / path).resolve()
    if manifest_relative.exists():
        return manifest_relative

    repo_root = _find_repo_root(base_dir)
    if repo_root is not None and path.parts and path.parts[0] == "data":
        return (repo_root / path).resolve()

    return manifest_relative


def _find_repo_root(start: Path) -> Path | None:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    return None


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _parse_bool(value: object, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ManifestError(f"{field_name} must be a boolean, not {type(value).__name__}.")
