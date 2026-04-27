from __future__ import annotations

import numpy as np
import pytest

from deep_earth_degasation.features.indices import brightness, bsi, ndmi, ndvi, robust_zscore


def test_ndvi_expected_values() -> None:
    nir = np.array([0.8, 0.2], dtype=float)
    red = np.array([0.2, 0.1], dtype=float)

    result = ndvi(nir, red)

    assert np.allclose(result, np.array([0.6, 1.0 / 3.0]), rtol=1e-6)


def test_ndmi_expected_values() -> None:
    nir = np.array([0.6, 0.4], dtype=float)
    swir1 = np.array([0.2, 0.4], dtype=float)

    result = ndmi(nir, swir1)

    assert np.allclose(result, np.array([0.5, 0.0]), rtol=1e-6)


def test_bsi_returns_finite_values() -> None:
    red = np.array([0.2, 0.3], dtype=float)
    nir = np.array([0.4, 0.5], dtype=float)
    swir1 = np.array([0.6, 0.7], dtype=float)
    blue = np.array([0.1, 0.2], dtype=float)

    result = bsi(red, nir, swir1, blue)

    assert np.isfinite(result).all()


def test_brightness_requires_at_least_one_band() -> None:
    with pytest.raises(ValueError, match="At least one band is required"):
        brightness()


def test_robust_zscore_handles_constant_arrays() -> None:
    values = np.ones((3, 3), dtype=float)

    result = robust_zscore(values)

    assert np.isfinite(result).all()
    assert np.allclose(result, np.zeros_like(values))
