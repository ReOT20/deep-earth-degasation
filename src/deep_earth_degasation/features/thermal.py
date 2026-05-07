from __future__ import annotations

from deep_earth_degasation.features.types import DynamicFeatureLayer, DynamicFeatureResult
from deep_earth_degasation.io.raster_stack import RasterStack

THERMAL_EVIDENCE_DIRECTIONS = {
    "LST": "higher_values_indicate_heat_or_thermal_support",
}


def landsat_thermal_features_from_stack(stack: RasterStack) -> DynamicFeatureResult:
    """Return prepared Landsat LST thermal feature layers with provenance."""
    layers = [
        layer
        for layer in stack.layers
        if layer.spec.sensor.lower() == "landsat" and layer.spec.feature_name == "LST"
    ]
    if not layers:
        return DynamicFeatureResult(features=(), missing_data_flags=("missing_landsat_LST",))

    features = tuple(
        DynamicFeatureLayer(
            name="LST",
            data=layer.data,
            sensor="landsat",
            date=layer.spec.date,
            source_layer_ids=(layer.spec.id,),
            evidence_direction=THERMAL_EVIDENCE_DIRECTIONS["LST"],
        )
        for layer in sorted(layers, key=lambda layer: layer.spec.date or "")
    )
    return DynamicFeatureResult(features=features)
