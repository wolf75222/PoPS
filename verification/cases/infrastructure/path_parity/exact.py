"""IF-07 native / DSL / hybrid path labels. Exact field is TR-01's sine.

The same manufactured sine is sampled independently under the three path
labels native, dsl, and hybrid.

Does not import pops or read a PoPS output.
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

PATHS = ("native", "dsl", "hybrid")
DEFAULT_N_CELLS = 32
PERIOD = 1.0


def cell_centers(n_cells: int, period: float = PERIOD) -> np.ndarray:
    """Return uniform cell centers on the periodic interval."""
    width = float(period) / int(n_cells)
    return (np.arange(int(n_cells), dtype=np.float64) + 0.5) * width


def cell_volumes(n_cells: int, period: float = PERIOD) -> np.ndarray:
    """Return uniform cell volumes on the periodic interval."""
    width = float(period) / int(n_cells)
    return np.full(int(n_cells), width, dtype=np.float64)


def exact_on_path(n_cells: int, name, t, **kwargs):
    """Sample the TR-01 sine under one native / DSL / hybrid path label."""
    if name not in PATHS:
        raise ValueError(f"unknown path {name!r}")
    centers = cell_centers(n_cells)
    return np.asarray(exact_sine(centers, t, **kwargs), dtype=np.float64)
