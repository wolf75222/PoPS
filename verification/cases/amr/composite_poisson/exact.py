"""AM-10 manufactured oracle: reuse the PO-01 trigonometric Poisson.

Does not import pops or read a PoPS output.
"""
from __future__ import annotations

from pathlib import Path

from verification.pops_verify.case_authoring import load_sibling_module

_PO01_EXACT = (
    Path(__file__).resolve().parents[2] / "poisson" / "periodic_trig" / "exact.py"
)
_po01 = load_sibling_module(_PO01_EXACT)

phi_exact = _po01.phi_exact
rhs_exact = _po01.rhs_exact
e_exact = _po01.e_exact
uniform_cell_grid = _po01.uniform_cell_grid
TWO_PI = _po01.TWO_PI

N_LEVELS = 2
N_COARSE = 16
INTERFACE = 0.5
REFINEMENT_RATIO = 2
COVERED_PARENT_RESIDUAL = 1.0e6
