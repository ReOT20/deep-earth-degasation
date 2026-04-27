from __future__ import annotations

import numpy as np

_EPS = 1e-9


def _ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return numerator / (denominator + _EPS)


def ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """Normalized Difference Vegetation Index."""
    return _ratio(nir - red, nir + red)


def ndmi(nir: np.ndarray, swir1: np.ndarray) -> np.ndarray:
    """Normalized Difference Moisture Index."""
    return _ratio(nir - swir1, nir + swir1)


def ndwi(nir: np.ndarray, green: np.ndarray) -> np.ndarray:
    """McFeeters-style NDWI variant using green and NIR."""
    return _ratio(green - nir, green + nir)


def msi(nir: np.ndarray, swir1: np.ndarray) -> np.ndarray:
    """Moisture Stress Index. Higher values often indicate stronger water stress."""
    return _ratio(swir1, nir)


def bsi(red: np.ndarray, nir: np.ndarray, swir1: np.ndarray, blue: np.ndarray) -> np.ndarray:
    """Bare Soil Index."""
    return _ratio((swir1 + red) - (nir + blue), (swir1 + red) + (nir + blue))


def brightness(*bands: np.ndarray) -> np.ndarray:
    """Simple mean brightness proxy across supplied bands."""
    if not bands:
        raise ValueError("At least one band is required")
    return np.nanmean(np.stack(bands, axis=0), axis=0)


def robust_zscore(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    """Median/MAD robust z-score."""
    median = np.nanmedian(values, axis=axis, keepdims=True)
    mad = np.nanmedian(np.abs(values - median), axis=axis, keepdims=True)
    return (values - median) / (1.4826 * mad + _EPS)
