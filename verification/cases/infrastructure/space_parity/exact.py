"""IF-03 Kokkos Serial vs OpenMP labels. Exact field is TR-01's sine.

The same manufactured sine is sampled once under each execution-space
label KokkosSerial and KokkosOpenMP. In-memory only; no live Kokkos.

Native labels map to OMP_NUM_THREADS=1 (Serial-like) and 8 (OpenMP).
GPU spaces are named here only so the runner can refuse them.

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

exact_sine = _tr01.exact_sine

EXECUTION_SPACES = ("KokkosSerial", "KokkosOpenMP")
SPACE_THREADS = {"KokkosSerial": 1, "KokkosOpenMP": 8}
GPU_SPACES = ("KokkosCuda", "KokkosHIP")
CUDA_UNAVAILABLE = "no public CUDA space"
REQUIRED_NATIVE_DIM = 1
DEFAULT_N_CELLS = 32
DEFAULT_T_END = 0.25
PERIOD = 1.0


def cell_centers(n_cells: int, period: float = PERIOD) -> np.ndarray:
    """Return uniform cell centers on the periodic interval."""
    width = float(period) / int(n_cells)
    return (np.arange(int(n_cells), dtype=np.float64) + 0.5) * width


def cell_volumes(n_cells: int, period: float = PERIOD) -> np.ndarray:
    """Return uniform cell volumes on the periodic interval."""
    width = float(period) / int(n_cells)
    return np.full(int(n_cells), width, dtype=np.float64)


def exact_on_space(n_cells: int, name, t, **kwargs):
    """Sample the TR-01 sine labelled by a Kokkos execution space."""
    if name not in EXECUTION_SPACES:
        raise ValueError(f"unknown execution space {name!r}")
    centers = cell_centers(n_cells)
    return np.asarray(exact_sine(centers, t, **kwargs), dtype=np.float64)
