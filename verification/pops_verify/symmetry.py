"""Symmetry diagnostics from an already-sampled field.

Plan §7.8:

    E_xy = ||U(x, y) - U(y, x)||_2 / ||U||_2
    E_radial = (max_θ R(θ) - min_θ R(θ)) / ⟨R⟩
"""
from __future__ import annotations

import numpy as np


def _as_float64(value) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("non-numeric values") from exc
    if array.size == 0:
        raise ValueError("empty input")
    return array


def _as_field(value) -> np.ndarray:
    array = _as_float64(value)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("shape mismatch")
    if not np.all(np.isfinite(array)):
        raise ValueError("non-finite values")
    return array


def _as_radii(value) -> np.ndarray:
    array = _as_float64(value)
    if array.ndim != 1:
        raise ValueError("shape mismatch")
    if not np.all(np.isfinite(array)):
        raise ValueError("non-finite values")
    return array


def xy_symmetry_error(field) -> float:
    """Return the §7.8 swap-symmetry error E_xy of a square scalar field."""
    samples = _as_field(field)
    scale = float(np.linalg.norm(samples))
    if scale == 0.0:
        raise ValueError("vanishing field norm")
    error = float(np.linalg.norm(samples - samples.T) / scale)
    if not np.isfinite(error):
        raise ValueError("non-finite values")
    return error


def radial_anisotropy(radii) -> float:
    """Return the §7.8 radial anisotropy of already-extracted R(θ) samples."""
    samples = _as_radii(radii)
    mean_radius = float(np.mean(samples))
    if mean_radius == 0.0:
        raise ValueError("zero mean radius")
    error = float((np.max(samples) - np.min(samples)) / mean_radius)
    if not np.isfinite(error):
        raise ValueError("non-finite values")
    return error
