"""TR-04 manufactured oracle: reuse the TR-02 periodic Gaussian.

Three 1-d placements (face / edge / corner) sit on the same two-block join
at x=0.5. Exact translation by whole periods maps them onto each other.

Does not import pops or read a PoPS output.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from verification.pops_verify.case_authoring import load_sibling_module

_TR02_EXACT = (
    Path(__file__).resolve().parents[1] / "gaussian_pulse" / "exact.py"
)
_tr02 = load_sibling_module(_TR02_EXACT)

exact_gaussian = _tr02.exact_gaussian
minimum_image = _tr02.minimum_image
PERIOD = _tr02.PERIOD
Q0 = _tr02.Q0
AMP = _tr02.AMP
X0 = _tr02.X0
SIGMA = _tr02.SIGMA
A = _tr02.A

PLACEMENTS = ("face", "edge", "corner")
FACE = 0.5
N_BLOCKS = 2
BLOCK_EDGES = (0.0, FACE, 1.0)
N_CELLS = 32


def placement_time(name) -> float:
    """Time at which the pulse centre sits on the face, plus whole periods."""
    if name not in PLACEMENTS:
        raise ValueError(f"unknown placement {name!r}")
    index = PLACEMENTS.index(name)
    return (float(FACE) - float(X0)) / float(A) + float(index) * float(PERIOD)


def two_block_cell_centers(n_cells: int = N_CELLS):
    """Return cell centres and volumes of the two-block join at x=0.5."""
    count = int(n_cells)
    if count < 2 or count % 2 != 0:
        raise ValueError(f"two-block join needs an even n_cells >= 2, got {n_cells!r}")
    width = 1.0 / float(count)
    centers = (np.arange(count, dtype=np.float64) + 0.5) * width
    volumes = np.full(count, width, dtype=np.float64)
    return centers, volumes


def exact_on_placement(name, x, **kwargs):
    """Evaluate the TR-02 Gaussian at the named 1-d crossing time."""
    return exact_gaussian(x, placement_time(name), **kwargs)
