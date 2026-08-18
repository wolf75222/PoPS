"""IF-09 TR-01 sine stored as float32 and float64.

The manufactured field is TR-01's periodic sine. This module only changes
the floating dtype of that field. Does not import pops or read a PoPS output.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from verification.pops_verify.case_authoring import load_sibling_module

_TR01_EXACT = (
    Path(__file__).resolve().parents[2] / "transport" / "advection_sine" / "exact.py"
)
_tr01 = load_sibling_module(_TR01_EXACT)

exact_sine = _tr01.exact_sine_1d

DEFAULT_N_CELLS = 32
T = 0.25
PERIOD = 1.0
DTYPES = (np.float32, np.float64)
LINF_BOUND = 1.0e-6


def cell_centers(n_cells: int = DEFAULT_N_CELLS, period: float = PERIOD) -> np.ndarray:
    """Return uniform cell centers on the periodic interval."""
    width = float(period) / int(n_cells)
    return (np.arange(int(n_cells), dtype=np.float64) + 0.5) * width


def cell_volumes(n_cells: int = DEFAULT_N_CELLS, period: float = PERIOD) -> np.ndarray:
    """Return uniform cell volumes on the periodic interval."""
    width = float(period) / int(n_cells)
    return np.full(int(n_cells), width, dtype=np.float64)


def exact_sine_as(x, t, dtype, **kwargs) -> np.ndarray:
    """Return the TR-01 sine stored in ``dtype`` (float32 or float64)."""
    requested = np.dtype(dtype)
    if requested not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError(f"unsupported dtype {dtype!r}")
    field = np.asarray(exact_sine(x, t, **kwargs), dtype=np.float64)
    return np.asarray(field, dtype=requested)


def fields_are_finite(*fields) -> bool:
    """Return True when every supplied field is nonempty and finite."""
    if not fields:
        return False
    return all(
        np.asarray(field).size > 0 and bool(np.all(np.isfinite(field)))
        for field in fields
    )
