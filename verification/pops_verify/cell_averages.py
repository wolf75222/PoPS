"""Analytic cell averages of an external oracle on Cartesian cells.

Plan §7.3:

    Ubar_i^exact = (1 / V_i) ∫_{V_i} U^exact(x, t) dV

Cell-center samples are not this integral. Quadrature is the 4-point
Gauss–Legendre tensor product (degree ≤ 7 per axis).
"""
from __future__ import annotations

import numpy as np

_GL4_NODES = np.array(
    [
        -0.861136311594052575223946488892809505,
        -0.339981043584856264802665759103244687,
        0.339981043584856264802665759103244687,
        0.861136311594052575223946488892809505,
    ],
    dtype=np.float64,
)
_GL4_WEIGHTS = np.array(
    [
        0.347854845137453857373063949221999408,
        0.652145154862546142626936050778000593,
        0.652145154862546142626936050778000593,
        0.347854845137453857373063949221999408,
    ],
    dtype=np.float64,
)


def _as_float64_bounds(cell_lo, cell_hi):
    try:
        lo = np.asarray(cell_lo, dtype=np.float64)
        hi = np.asarray(cell_hi, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("non-numeric cell bounds") from exc
    return lo, hi


def _normalize_bounds(lo, hi):
    if lo.size == 0 or hi.size == 0:
        raise ValueError("empty input")
    if lo.ndim == 0:
        lo = lo.reshape(1)
    if hi.ndim == 0:
        hi = hi.reshape(1)
    if lo.ndim == 1 and hi.ndim == 1:
        lo = lo[:, None]
        hi = hi[:, None]
    if lo.shape != hi.shape:
        raise ValueError("shape mismatch")
    dim = lo.shape[-1]
    if dim not in (1, 2, 3):
        raise ValueError("spatial dimension must be 1, 2, or 3")
    if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
        raise ValueError("non-finite values")
    extents = hi - lo
    if np.any(extents <= 0.0):
        raise ValueError("non-positive cell extent")
    return lo, hi, extents, dim


def _scalar_time(t):
    if t is None:
        return None
    time = np.asarray(t, dtype=np.float64)
    if time.shape != ():
        raise ValueError("t must be a scalar")
    if not np.isfinite(time):
        raise ValueError("non-finite values")
    return float(time)


def _quadrature_nodes(dim):
    grids = np.meshgrid(*(_GL4_NODES for _ in range(dim)), indexing="ij")
    weight_grids = np.meshgrid(*(_GL4_WEIGHTS for _ in range(dim)), indexing="ij")
    nodes = np.stack([grid.reshape(-1) for grid in grids], axis=-1)
    weights = np.prod(
        np.stack([grid.reshape(-1) for grid in weight_grids], axis=-1),
        axis=-1,
    )
    return nodes, weights


def analytic_cell_averages(u_exact, cell_lo, cell_hi, t=None) -> np.ndarray:
    """Return the §7.3 analytic average of ``u_exact`` on each Cartesian cell."""
    if not callable(u_exact):
        raise ValueError("u_exact must be callable")
    lo, hi = _as_float64_bounds(cell_lo, cell_hi)
    lo, hi, extents, dim = _normalize_bounds(lo, hi)
    time = _scalar_time(t)
    nodes, weights = _quadrature_nodes(dim)
    center = 0.5 * (lo + hi)
    half = 0.5 * extents
    points = center[..., None, :] + half[..., None, :] * nodes
    coords = [points[..., axis] for axis in range(dim)]
    try:
        values = u_exact(*coords) if time is None else u_exact(*coords, time)
    except TypeError as exc:
        raise ValueError("u_exact signature mismatch") from exc
    try:
        values = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("non-finite values") from exc
    if values.size == 0:
        raise ValueError("empty input")
    try:
        values = np.broadcast_to(values, coords[0].shape)
    except ValueError as exc:
        raise ValueError("shape mismatch") from exc
    if not np.all(np.isfinite(values)):
        raise ValueError("non-finite values")
    averages = np.sum(values * weights, axis=-1) / (2.0**dim)
    if not np.all(np.isfinite(averages)):
        raise ValueError("non-finite values")
    return np.array(averages, dtype=np.float64, copy=True)
