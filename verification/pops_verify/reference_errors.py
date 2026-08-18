"""Volume-weighted errors of a numerical field against an external oracle.

Plan §7.1:

    L1    = sum_i V_i |U_i - U_i^exact| / sum_i V_i
    L2    = (sum_i V_i |U_i - U_i^exact|^2 / sum_i V_i)^{1/2}
    Linf  = max_i |U_i - U_i^exact|
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class ReferenceErrors:
    """L1, L2, and Linf norms of ``U - U_exact``, with L1/L2 volume-weighted."""

    l1: float
    l2: float
    linf: float


def reference_errors(u, u_exact, volumes) -> ReferenceErrors:
    """Return volume-weighted L1/L2 and max-norm Linf against ``u_exact``."""
    field = np.asarray(u)
    oracle = np.asarray(u_exact)
    cell_volumes = np.asarray(volumes)
    if field.size == 0 or oracle.size == 0 or cell_volumes.size == 0:
        raise ValueError("empty input")
    try:
        field, oracle, cell_volumes = np.broadcast_arrays(field, oracle, cell_volumes)
    except ValueError as exc:
        raise ValueError("shape mismatch") from exc
    if not (
        np.all(np.isfinite(field))
        and np.all(np.isfinite(oracle))
        and np.all(np.isfinite(cell_volumes))
    ):
        raise ValueError("non-finite values")
    total_volume = float(np.sum(cell_volumes, dtype=np.float64))
    if not np.isfinite(total_volume):
        raise ValueError("non-finite values")
    if total_volume <= 0.0:
        raise ValueError("non-positive total volume")
    abs_error = np.abs(np.subtract(field, oracle, dtype=np.float64))
    l1_numerator = float(np.sum(cell_volumes * abs_error))
    l2_numerator = float(np.sum(cell_volumes * np.square(abs_error)))
    linf = float(np.max(abs_error))
    l1 = l1_numerator / total_volume
    l2 = float(np.sqrt(l2_numerator / total_volume))
    if not (
        np.isfinite(l1_numerator)
        and np.isfinite(l2_numerator)
        and np.isfinite(l1)
        and np.isfinite(l2)
        and np.isfinite(linf)
    ):
        raise ValueError("non-finite values")
    return ReferenceErrors(l1=l1, l2=l2, linf=linf)
