"""IF-07 in-memory path labels. Public Case is the DSL path.

In-memory native / dsl / hybrid labels sample the same exact sine.
The public TR-01 Case is the DSL authoring path. Hybrid / native C++
authoring is capability-gated; this module does not invent a second
authoring stack. ``run_native`` refuses that missing stack.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.reference_errors import reference_errors

_exact = load_sibling_module(Path(__file__).with_name("exact.py"))
_v15 = load_sibling_module(Path(__file__).resolve().parents[1] / "_v15.py")

HYBRID_NATIVE_CPP_REFUSAL = "public hybrid/native C++ authoring not active"


class NativeUnavailable(RuntimeError):
    """Raised when hybrid or native C++ authoring is requested."""


def exact_fields_for_paths(n_cells: int = _exact.DEFAULT_N_CELLS, t=0.0):
    """Return exact fields keyed by the three path labels."""
    return {
        name: _exact.exact_on_path(n_cells, name, t)
        for name in _exact.PATHS
    }


def max_path_difference(n_cells: int = _exact.DEFAULT_N_CELLS, t=0.0) -> float:
    """Return the max pairwise L∞ between the three exact path labels."""
    fields = exact_fields_for_paths(n_cells, t)
    volumes = _exact.cell_volumes(n_cells)
    linf = 0.0
    for left, right in combinations(fields.values(), 2):
        errors = reference_errors(left, right, volumes)
        linf = max(linf, errors.linf)
    return float(linf)


def public_dsl_case(n_cells: int = _exact.DEFAULT_N_CELLS):
    """Return the shared 1-d TR-01 Case. That Case is the DSL path."""
    from verification.pops_verify.tr01_runtime import build_case

    return build_case(n_cells)


def refuse_hybrid_native_cpp() -> str:
    """Return why hybrid / native C++ authoring is not a second stack."""
    return HYBRID_NATIVE_CPP_REFUSAL


def run_native(n_cells: int = _exact.DEFAULT_N_CELLS, t_end=0.25, request=None):
    """Require a native / DSL / hybrid comparison. DSL-only is not success."""
    _v15.bind_campaign(request, NativeUnavailable)
    del n_cells, t_end
    raise NativeUnavailable(refuse_hybrid_native_cpp())
