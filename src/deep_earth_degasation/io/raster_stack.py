from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import geopandas as gpd
import numpy as np
import numpy.typing as npt
import rasterio
from affine import Affine
from rasterio.coords import BoundingBox
from rasterio.crs import CRS

from deep_earth_degasation.pipeline.manifest import PreparedStackManifest, RasterLayerSpec


class RasterStackError(ValueError):
    """Raised when prepared raster data violates the manifest contract."""


FloatArray: TypeAlias = npt.NDArray[np.float64]


@dataclass(frozen=True)
class RasterLayer:
    spec: RasterLayerSpec
    data: FloatArray
    crs: str
    transform: Affine
    bounds: BoundingBox
    resolution_m: tuple[float, float]
    shape: tuple[int, int]


@dataclass(frozen=True)
class RasterStack:
    manifest: PreparedStackManifest
    layers: tuple[RasterLayer, ...]


def load_raster_stack(manifest: PreparedStackManifest) -> RasterStack:
    """Load a prepared raster stack from a validated manifest."""
    if not manifest.raster_layers:
        raise RasterStackError("Manifest contains no raster layers.")

    aoi_bounds = _load_aoi_bounds(manifest)
    layers = tuple(
        _load_layer(layer_spec, manifest=manifest) for layer_spec in manifest.raster_layers
    )
    _validate_common_grid(layers, manifest=manifest)
    for layer in layers:
        _validate_aoi_coverage(layer, aoi_bounds)
    return RasterStack(manifest=manifest, layers=layers)


def write_run_manifest(stack: RasterStack, path: str | Path) -> None:
    """Write deterministic provenance for a loaded prepared raster stack."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": stack.manifest.run.id,
        "crs": stack.manifest.crs,
        "resolution_m": stack.manifest.resolution_m,
        "layers": [
            {
                "id": layer.spec.id,
                "sensor": layer.spec.sensor,
                "feature_name": layer.spec.feature_name,
                "date": layer.spec.date,
                "path": str(layer.spec.path),
                "quality_mask_path": (
                    None
                    if layer.spec.quality_mask_path is None
                    else str(layer.spec.quality_mask_path)
                ),
                "nodata": layer.spec.nodata,
                "role": layer.spec.role,
                "shape": list(layer.shape),
                "resolution_m": list(layer.resolution_m),
                "transform": list(layer.transform),
                "bounds": {
                    "left": layer.bounds.left,
                    "bottom": layer.bounds.bottom,
                    "right": layer.bounds.right,
                    "top": layer.bounds.top,
                },
            }
            for layer in stack.layers
        ],
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_layer(layer_spec: RasterLayerSpec, *, manifest: PreparedStackManifest) -> RasterLayer:
    with rasterio.open(layer_spec.path) as src:
        if src.count != 1:
            raise RasterStackError(f"{layer_spec.id} must be a single-band raster.")
        raster_crs = _require_raster_crs(src.crs, layer_spec.id)
        _validate_crs(raster_crs, expected=layer_spec.crs, layer_id=layer_spec.id)
        _validate_crs(raster_crs, expected=manifest.crs, layer_id=layer_spec.id)
        _validate_resolution(src.res, expected=layer_spec.resolution_m, layer_id=layer_spec.id)

        data = src.read(1).astype(np.float64)
        if layer_spec.nodata is not None:
            data[data == layer_spec.nodata] = np.nan
        if src.nodata is not None:
            data[data == float(src.nodata)] = np.nan

        if layer_spec.quality_mask_path is not None:
            data = _apply_quality_mask(
                data, layer_spec.quality_mask_path, reference=src, layer_id=layer_spec.id
            )

        return RasterLayer(
            spec=layer_spec,
            data=data,
            crs=raster_crs.to_string(),
            transform=src.transform,
            bounds=src.bounds,
            resolution_m=(float(src.res[0]), float(src.res[1])),
            shape=(src.height, src.width),
        )


def _apply_quality_mask(
    data: FloatArray,
    mask_path: Path,
    *,
    reference: Any,
    layer_id: str,
) -> FloatArray:
    with rasterio.open(mask_path) as mask:
        _validate_crs(
            _require_raster_crs(mask.crs, f"{layer_id} quality mask"),
            expected=str(reference.crs),
            layer_id=f"{layer_id} quality mask",
        )
        if (
            mask.transform != reference.transform
            or mask.width != reference.width
            or mask.height != reference.height
        ):
            raise RasterStackError(f"{layer_id} quality mask grid does not match raster grid.")
        mask_data = mask.read(1)

    output = data.copy()
    output[mask_data == 0] = np.nan
    return output


def _validate_common_grid(
    layers: tuple[RasterLayer, ...], *, manifest: PreparedStackManifest
) -> None:
    if not manifest.quality.reject_mismatched_grid:
        return

    reference_by_resolution: dict[float, RasterLayer] = {}
    for layer in layers:
        resolution = layer.spec.resolution_m
        reference = reference_by_resolution.setdefault(resolution, layer)
        if layer.transform != reference.transform or layer.shape != reference.shape:
            raise RasterStackError(
                f"{layer.spec.id} grid does not match {reference.spec.id}; "
                f"prepared rasters with {resolution:g} m resolution must share a grid."
            )


def _load_aoi_bounds(manifest: PreparedStackManifest) -> BoundingBox:
    aoi = gpd.read_file(manifest.aoi.path)
    if aoi.empty:
        raise RasterStackError("AOI vector contains no features.")
    if aoi.crs is None:
        raise RasterStackError("AOI vector has no CRS metadata.")
    if not CRS.from_user_input(aoi.crs).equals(CRS.from_user_input(manifest.aoi.crs)):
        raise RasterStackError("AOI CRS does not match manifest AOI CRS.")
    if not CRS.from_user_input(aoi.crs).equals(CRS.from_user_input(manifest.crs)):
        raise RasterStackError("AOI CRS does not match manifest stack CRS.")

    minx, miny, maxx, maxy = aoi.total_bounds
    return BoundingBox(left=float(minx), bottom=float(miny), right=float(maxx), top=float(maxy))


def _validate_aoi_coverage(layer: RasterLayer, aoi_bounds: BoundingBox) -> None:
    bounds = layer.bounds
    covers = (
        bounds.left <= aoi_bounds.left
        and bounds.right >= aoi_bounds.right
        and bounds.bottom <= aoi_bounds.bottom
        and bounds.top >= aoi_bounds.top
    )
    if not covers:
        raise RasterStackError(f"{layer.spec.id} does not cover the AOI bounds.")


def _require_raster_crs(crs: CRS | None, layer_id: str) -> CRS:
    if crs is None:
        raise RasterStackError(f"{layer_id} has no CRS metadata.")
    return crs


def _validate_crs(crs: CRS, *, expected: str, layer_id: str) -> None:
    if not crs.equals(CRS.from_user_input(expected)):
        raise RasterStackError(f"{layer_id} CRS does not match expected CRS {expected}.")


def _validate_resolution(
    resolution: tuple[float, float],
    *,
    expected: float,
    layer_id: str,
) -> None:
    if not np.isclose(float(resolution[0]), expected) or not np.isclose(
        float(resolution[1]), expected
    ):
        raise RasterStackError(f"{layer_id} resolution does not match manifest metadata.")
