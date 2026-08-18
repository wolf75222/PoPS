"""IF-01 MPI decomposition placements. Exact field is TR-01's sine.

The same manufactured sine is sampled independently on one block [0, 1],
two blocks [0, 0.5] ∪ [0.5, 1], and the 1×4 / 4×1 1-d splits at
0.25 / 0.5 / 0.75.

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

PLACEMENTS = ("one_block", "two_block", "1x4", "4x1")
ONE_BLOCK_EDGES = (0.0, 1.0)
TWO_BLOCK_EDGES = (0.0, 0.5, 1.0)
FOUR_BLOCK_SPLITS = (0.25, 0.5, 0.75)
FOUR_BLOCK_EDGES = (0.0, 0.25, 0.5, 0.75, 1.0)
DEFAULT_N_CELLS = 32
PERIOD = 1.0

_PLACEMENT_EDGES = {
    "one_block": ONE_BLOCK_EDGES,
    "two_block": TWO_BLOCK_EDGES,
    "1x4": FOUR_BLOCK_EDGES,
    "4x1": FOUR_BLOCK_EDGES,
}


def cell_centers(n_cells: int, period: float = PERIOD) -> np.ndarray:
    """Return uniform cell centers on the periodic interval."""
    width = float(period) / int(n_cells)
    return (np.arange(int(n_cells), dtype=np.float64) + 0.5) * width


def cell_volumes(n_cells: int, period: float = PERIOD) -> np.ndarray:
    """Return uniform cell volumes on the periodic interval."""
    width = float(period) / int(n_cells)
    return np.full(int(n_cells), width, dtype=np.float64)


def placement_edges(name):
    """Return the 1-d block edges of a named MPI-style placement."""
    if name not in _PLACEMENT_EDGES:
        raise ValueError(f"unknown placement {name!r}")
    return _PLACEMENT_EDGES[name]


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


def block_slices(n_cells: int, edges, period: float = PERIOD):
    """Return (start, stop) cell slices for each block of a 1-d split."""
    cuts = [0]
    for interface in edges[1:-1]:
        cuts.append(n_left_cells(n_cells, interface, period))
    cuts.append(int(n_cells))
    if cuts != sorted(cuts) or len(set(cuts)) != len(cuts):
        raise ValueError("block edges must be strictly increasing cell faces")
    return list(zip(cuts[:-1], cuts[1:], strict=True))


def exact_on_placement(n_cells: int, name, t, **kwargs):
    """Sample the TR-01 sine independently on each block of the placement."""
    centers = cell_centers(n_cells)
    pieces = [
        np.asarray(exact_sine(centers[start:stop], t, **kwargs), dtype=np.float64)
        for start, stop in block_slices(n_cells, placement_edges(name))
    ]
    return np.concatenate(pieces)
