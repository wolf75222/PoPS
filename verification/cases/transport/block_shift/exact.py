"""TR-05 translated same-level block faces. Exact field is TR-02's Gaussian.

The physical problem is unchanged; only the two-block join moves.
Does not import pops or read a PoPS output.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

INTERFACE_X = (0.25, 0.2578125, 0.375)
DEFAULT_N_CELLS = 128
PERIOD = 1.0


def _tr02_exact():
    path = Path(__file__).resolve().parents[1] / "gaussian_pulse" / "exact.py"
    spec = importlib.util.spec_from_file_location(
        "pops_verify_case_exact_gaussian_pulse", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cell_centers(n_cells: int, period: float = PERIOD) -> np.ndarray:
    """Return uniform cell centers on the periodic interval."""
    width = float(period) / int(n_cells)
    return (np.arange(int(n_cells), dtype=np.float64) + 0.5) * width


def cell_volumes(n_cells: int, period: float = PERIOD) -> np.ndarray:
    """Return uniform cell volumes on the periodic interval."""
    width = float(period) / int(n_cells)
    return np.full(int(n_cells), width, dtype=np.float64)


def n_left_cells(n_cells: int, interface_x: float, period: float = PERIOD) -> int:
    """Return the left-block cell count when the join is a cell face."""
    width = float(period) / int(n_cells)
    ratio = float(interface_x) / width
    n_left = int(round(ratio))
    if abs(n_left * width - float(interface_x)) > 1.0e-15:
        raise ValueError(
            f"interface {interface_x} is not a cell face on n={n_cells}"
        )
    if n_left <= 0 or n_left >= int(n_cells):
        raise ValueError("interface must split the domain into two nonempty blocks")
    return n_left


def exact_on_decomposition(n_cells: int, interface_x: float, t, **kwargs):
    """Sample the TR-02 Gaussian independently on each side of the join."""
    centers = cell_centers(n_cells)
    n_left = n_left_cells(n_cells, interface_x)
    pulse = _tr02_exact()
    left = np.asarray(
        pulse.exact_gaussian(centers[:n_left], t, **kwargs), dtype=np.float64
    )
    right = np.asarray(
        pulse.exact_gaussian(centers[n_left:], t, **kwargs), dtype=np.float64
    )
    return np.concatenate([left, right])
