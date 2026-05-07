from __future__ import annotations

from deep_earth_degasation.features.indices import brightness, bsi, msi, ndmi, ndvi, ndwi
from deep_earth_degasation.features.types import DynamicFeatureLayer, DynamicFeatureResult
from deep_earth_degasation.io.raster_stack import FloatArray, RasterLayer, RasterStack

SPECTRAL_EVIDENCE_DIRECTIONS = {
    "NDVI": "lower_values_indicate_vegetation_stress",
    "NDMI": "lower_values_indicate_moisture_stress",
    "NDWI": "lower_values_indicate_water_or_moisture_stress",
    "MSI": "higher_values_indicate_moisture_stress",
    "BSI": "higher_values_indicate_bare_soil_or_brightness",
    "brightness": "higher_values_indicate_brightness",
}


def sentinel2_features_from_stack(stack: RasterStack) -> DynamicFeatureResult:
    """Compute available Sentinel-2 spectral feature layers from a prepared stack."""
    flags: list[str] = []
    source = _layers_by_feature(stack, sensor="sentinel2")
    features: list[DynamicFeatureLayer] = []

    _append_prepared_layer(features, source, "NDVI", "sentinel2", flags)
    _append_prepared_layer(features, source, "NDMI", "sentinel2", flags)
    _append_prepared_layer(features, source, "NDWI", "sentinel2", flags)
    _append_prepared_layer(features, source, "MSI", "sentinel2", flags)
    _append_prepared_layer(features, source, "BSI", "sentinel2", flags)
    _append_prepared_layer(features, source, "brightness", "sentinel2", flags)

    if "red_edge" not in source:
        flags.append("missing_sentinel2_red_edge")
    if "EVI" not in source:
        flags.append("missing_sentinel2_EVI")

    return DynamicFeatureResult(features=tuple(features), missing_data_flags=tuple(flags))


def compute_ndvi(nir: FloatArray, red: FloatArray) -> DynamicFeatureLayer:
    return _computed_spectral_layer("NDVI", ndvi(nir, red), ("NIR", "red"))


def compute_ndmi(nir: FloatArray, swir1: FloatArray) -> DynamicFeatureLayer:
    return _computed_spectral_layer("NDMI", ndmi(nir, swir1), ("NIR", "SWIR1"))


def compute_ndwi(nir: FloatArray, green: FloatArray) -> DynamicFeatureLayer:
    return _computed_spectral_layer("NDWI", ndwi(nir, green), ("NIR", "green"))


def compute_msi(nir: FloatArray, swir1: FloatArray) -> DynamicFeatureLayer:
    return _computed_spectral_layer("MSI", msi(nir, swir1), ("NIR", "SWIR1"))


def compute_bsi(
    red: FloatArray, nir: FloatArray, swir1: FloatArray, blue: FloatArray
) -> DynamicFeatureLayer:
    return _computed_spectral_layer(
        "BSI", bsi(red, nir, swir1, blue), ("red", "NIR", "SWIR1", "blue")
    )


def compute_brightness(*bands: FloatArray) -> DynamicFeatureLayer:
    return _computed_spectral_layer("brightness", brightness(*bands), ("multiband",))


def _computed_spectral_layer(
    name: str, data: FloatArray, source_layer_ids: tuple[str, ...]
) -> DynamicFeatureLayer:
    return DynamicFeatureLayer(
        name=name,
        data=data,
        sensor="sentinel2",
        date=None,
        source_layer_ids=source_layer_ids,
        evidence_direction=SPECTRAL_EVIDENCE_DIRECTIONS[name],
    )


def _append_prepared_layer(
    features: list[DynamicFeatureLayer],
    source: dict[str, list[RasterLayer]],
    feature_name: str,
    sensor: str,
    flags: list[str],
) -> None:
    layers = source.get(feature_name, [])
    if not layers:
        flags.append(f"missing_{sensor}_{feature_name}")
        return

    for layer in sorted(layers, key=lambda item: (item.spec.date or "", item.spec.id)):
        features.append(
            DynamicFeatureLayer(
                name=feature_name,
                data=layer.data,
                sensor=sensor,
                date=layer.spec.date,
                source_layer_ids=(layer.spec.id,),
                evidence_direction=SPECTRAL_EVIDENCE_DIRECTIONS[feature_name],
            )
        )


def _layers_by_feature(stack: RasterStack, *, sensor: str) -> dict[str, list[RasterLayer]]:
    by_feature: dict[str, list[RasterLayer]] = {}
    for layer in stack.layers:
        if layer.spec.sensor.lower() == sensor.lower():
            by_feature.setdefault(layer.spec.feature_name, []).append(layer)
    return by_feature


def shape_mismatch_flag(name: str, *arrays: FloatArray) -> tuple[str, ...]:
    shapes = {array.shape for array in arrays}
    if len(shapes) <= 1:
        return ()
    return (f"{name}_shape_mismatch",)
