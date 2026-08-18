"""Observed order from a resolution series of already-computed errors.

Plan §7.4 (ratio-two specialization):

    p_obs = log(E_h / E_{h/2}) / log 2

General Richardson form used here (reduces to §7.4 when h_{i+1} = h_i / 2):

    p_i = log(E_i / E_{i+1}) / log(h_i / h_{i+1})
"""
from __future__ import annotations

import numpy as np


def observed_order(errors, resolutions) -> np.ndarray:
    """Return pairwise observed orders from error scalars and spacings ``h``."""
    try:
        error_series = np.asarray(errors, dtype=np.float64)
        spacing_series = np.asarray(resolutions, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("non-numeric values") from exc
    if error_series.ndim != 1 or spacing_series.ndim != 1:
        raise ValueError("shape mismatch")
    if error_series.size == 0 or spacing_series.size == 0:
        raise ValueError("empty input")
    if error_series.size < 2 or spacing_series.size < 2:
        raise ValueError("length < 2")
    if error_series.shape != spacing_series.shape:
        raise ValueError("shape mismatch")
    if not (np.all(np.isfinite(error_series)) and np.all(np.isfinite(spacing_series))):
        raise ValueError("non-finite values")
    if np.any(spacing_series <= 0.0):
        raise ValueError("non-positive spacing")
    if np.any(error_series <= 0.0):
        raise ValueError("non-positive errors")
    if np.any(spacing_series[:-1] == spacing_series[1:]):
        raise ValueError("zero spacing ratio")
    orders = np.log(error_series[:-1] / error_series[1:]) / np.log(
        spacing_series[:-1] / spacing_series[1:]
    )
    if not np.all(np.isfinite(orders)):
        raise ValueError("non-finite values")
    return orders
