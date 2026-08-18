"""TM-01 manufactured oracle: reuse the TR-01 periodic sine.

Does not import pops or read a PoPS output.
"""
from __future__ import annotations

from pathlib import Path

from verification.pops_verify.case_authoring import load_sibling_module

_TR01_EXACT = (
    Path(__file__).resolve().parents[2] / "transport" / "advection_sine" / "exact.py"
)
_tr01 = load_sibling_module(_TR01_EXACT)

exact_sine = _tr01.exact_sine_1d
uniform_cell_centers = _tr01.uniform_cell_centers

N_CELLS = 64
DT = 1.0 / 128.0
DT_SERIES = tuple(DT / factor for factor in (1, 2, 4, 8))
