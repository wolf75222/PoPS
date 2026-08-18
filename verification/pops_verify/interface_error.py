"""Coarse-fine interface-band errors from an already-sampled field.

Plan §7.9:

    E_cf = max_{d(x, Γ_cf) < m h_f} |U_h - U_exact|

Report separately: interface-band error, far/bulk error, E_cf / E_bulk,
and the index of the maximum error.
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


def _as_scalar(value) -> float:
    array = _as_float64(value)
    if array.ndim != 0:
        raise ValueError("shape mismatch")
    if not np.isfinite(array):
        raise ValueError("non-finite values")
    return float(array)


def _as_field(value) -> np.ndarray:
    array = _as_float64(value)
    if not np.all(np.isfinite(array)):
        raise ValueError("non-finite values")
    return array


def _require_boolean_mask(mask, shape) -> np.ndarray:
    array = np.asarray(mask)
    if array.size == 0:
        raise ValueError("empty input")
    if array.dtype != bool:
        raise ValueError("non-boolean mask")
    if array.shape != shape:
        raise ValueError("shape mismatch")
    if not np.any(array):
        raise ValueError("empty band")
    return array


def _aligned_error(u, u_exact, mask) -> tuple[np.ndarray, np.ndarray]:
    field = _as_field(u)
    oracle = _as_field(u_exact)
    if field.shape != oracle.shape:
        raise ValueError("shape mismatch")
    selected = _require_boolean_mask(mask, field.shape)
    return np.abs(field - oracle), selected


def interface_band_mask(distance, *, h_fine, band_cells=4) -> np.ndarray:
    """Return the §7.9 boolean band d(x, Γ_cf) < m h_f."""
    samples = _as_field(distance)
    h_fine_value = _as_scalar(h_fine)
    band_cells_value = _as_scalar(band_cells)
    if h_fine_value <= 0.0 or band_cells_value <= 0.0:
        raise ValueError("non-positive band inputs")
    return samples < band_cells_value * h_fine_value


def band_max_abs_error(u, u_exact, mask) -> float:
    """Return max |U_h - U_exact| on the True cells of mask."""
    abs_error, selected = _aligned_error(u, u_exact, mask)
    error = float(np.max(abs_error[selected]))
    if not np.isfinite(error):
        raise ValueError("non-finite values")
    return error


def interface_bulk_ratio(e_cf, e_bulk) -> float:
    """Return the §7.9 ratio E_cf / E_bulk."""
    e_cf_value = _as_scalar(e_cf)
    e_bulk_value = _as_scalar(e_bulk)
    if e_bulk_value == 0.0:
        raise ValueError("zero bulk error")
    ratio = e_cf_value / e_bulk_value
    if not np.isfinite(ratio):
        raise ValueError("non-finite values")
    return float(ratio)


def max_error_location(u, u_exact, mask) -> tuple[int, ...]:
    """Return the original-array index of the first band maximum."""
    abs_error, selected = _aligned_error(u, u_exact, mask)
    ranked = np.where(selected, abs_error, -np.inf)
    flat = int(np.argmax(ranked))
    return tuple(int(index) for index in np.unravel_index(flat, ranked.shape))
