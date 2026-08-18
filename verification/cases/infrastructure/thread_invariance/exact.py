"""IF-02 OpenMP thread-count labels. Exact field is TR-01's sine.

The same manufactured sine is sampled independently on OpenMP-style static
slices as if run with 1, 2, 4, or 8 threads. The arrays are identical.

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

THREAD_COUNTS = (1, 2, 4, 8)
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


def thread_slices(n_cells: int, n_threads: int):
    """Return (start, stop) cell slices for OpenMP-style static scheduling."""
    count = int(n_cells)
    threads = int(n_threads)
    if threads not in THREAD_COUNTS:
        raise ValueError(f"unknown thread count {n_threads!r}")
    if count % threads != 0:
        raise ValueError(f"n={count} is not divisible by {threads} threads")
    width = count // threads
    return [(index * width, (index + 1) * width) for index in range(threads)]


def exact_on_thread_count(n_cells: int, n_threads: int, t, **kwargs):
    """Sample the TR-01 sine independently on each thread's static slice."""
    centers = cell_centers(n_cells)
    pieces = [
        np.asarray(exact_sine(centers[start:stop], t, **kwargs), dtype=np.float64)
        for start, stop in thread_slices(n_cells, n_threads)
    ]
    return np.concatenate(pieces)
