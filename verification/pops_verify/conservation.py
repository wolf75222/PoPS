"""Discrete conservation residual from an already-reduced balance statement.

Plan §7.5:

    δQ(t) = Q(t) - Q(0) - ∫ S_Q dt + ∫∫ F_Q · n dA dt

Discrete public convention (Balance / BalanceTerms), with
storage_change = Q(t) - Q(0):

    residual = storage_change + outward_boundary_flux - sources - reflux - projection

Plan §8.4:

    tol(Q) = max(abs_tol, rel_tol * |Q_scale|, C * ε_mach * N_updates)
"""
from __future__ import annotations

import numpy as np

_MACHINE_EPS = float(np.finfo(np.float64).eps)


def _as_float64(value) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("non-numeric values") from exc
    if array.size == 0:
        raise ValueError("empty input")
    return array


def conservation_residual(
    storage_change,
    outward_boundary_flux,
    sources,
    reflux=0.0,
    projection=0.0,
) -> np.ndarray:
    """Return the discrete conservation residual of already-reduced terms."""
    terms = [
        _as_float64(storage_change),
        _as_float64(outward_boundary_flux),
        _as_float64(sources),
        _as_float64(reflux),
        _as_float64(projection),
    ]
    if any(term.ndim > 1 for term in terms):
        raise ValueError("shape mismatch")
    try:
        storage, flux, src, ref, proj = np.broadcast_arrays(*terms)
    except ValueError as exc:
        raise ValueError("shape mismatch") from exc
    if storage.ndim > 1:
        raise ValueError("shape mismatch")
    if not all(
        np.all(np.isfinite(term)) for term in (storage, flux, src, ref, proj)
    ):
        raise ValueError("non-finite values")
    residual = storage + flux - src - ref - proj
    if not np.all(np.isfinite(residual)):
        raise ValueError("non-finite values")
    return residual


def conservation_tolerance(
    q_scale, *, abs_tol, rel_tol, n_updates, c=1.0
) -> float:
    """Return the §8.4 conservation tolerance for an already-reduced scale."""
    values = {
        "q_scale": _as_float64(q_scale),
        "abs_tol": _as_float64(abs_tol),
        "rel_tol": _as_float64(rel_tol),
        "n_updates": _as_float64(n_updates),
        "c": _as_float64(c),
    }
    for array in values.values():
        if array.ndim != 0:
            raise ValueError("shape mismatch")
        if not np.isfinite(array):
            raise ValueError("non-finite values")
    abs_tol_value = float(values["abs_tol"])
    rel_tol_value = float(values["rel_tol"])
    n_updates_value = float(values["n_updates"])
    c_value = float(values["c"])
    if min(abs_tol_value, rel_tol_value, n_updates_value, c_value) < 0.0:
        raise ValueError("negative tolerance inputs")
    tolerance = max(
        abs_tol_value,
        rel_tol_value * abs(float(values["q_scale"])),
        c_value * _MACHINE_EPS * n_updates_value,
    )
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("non-finite values")
    return float(tolerance)
