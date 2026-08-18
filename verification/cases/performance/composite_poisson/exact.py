"""PF-05 manufactured oracle: reuse AM-10 / PO-01 trigonometric Poisson.

Does not import pops or read a PoPS output.
"""
from __future__ import annotations

from pathlib import Path

from verification.pops_verify.case_authoring import load_sibling_module

_AM10_EXACT = (
    Path(__file__).resolve().parents[2] / "amr" / "composite_poisson" / "exact.py"
)
_am10 = load_sibling_module(_AM10_EXACT)

phi_exact = _am10.phi_exact
rhs_exact = _am10.rhs_exact
e_exact = _am10.e_exact
uniform_cell_grid = _am10.uniform_cell_grid
TWO_PI = _am10.TWO_PI

N_LEVELS = _am10.N_LEVELS
N_COARSE = _am10.N_COARSE
INTERFACE = _am10.INTERFACE
REFINEMENT_RATIO = _am10.REFINEMENT_RATIO
COVERED_PARENT_RESIDUAL = _am10.COVERED_PARENT_RESIDUAL
