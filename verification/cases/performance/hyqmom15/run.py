"""PF-12 in-memory 15-component saxpy.

Synthetic HyQMOM15-width state versus 5-component Euler bytes/cell.
Optional ``run_native`` refuses unless a public 15-component Case exists.
"""
from __future__ import annotations

import time
from pathlib import Path

from verification.pops_verify.case_authoring import load_sibling_module

_exact = load_sibling_module(Path(__file__).with_name("exact.py"))

HYQMOM15_CASE_REFUSAL = "public 15-component HyQMOM15 Case is not available"


class NativeUnavailable(RuntimeError):
    """Raised when a public 15-component Case cannot be timed."""


def wide_state(n_cells=_exact.N_CELLS, n_components=_exact.N_COMPONENTS) -> dict:
    """Return the synthetic (n, 15) HyQMOM15-width state."""
    state = _exact.interior_state(n_cells, n_components)
    return {
        "state": state,
        "n_components": int(n_components),
        "bytes_per_cell": _exact.bytes_per_cell(n_components),
    }


def bytes_per_cell(n_components=_exact.N_COMPONENTS) -> int:
    """Return float64 bytes/cell for the HyQMOM15-width state."""
    return _exact.bytes_per_cell(n_components)


def width_comparison() -> dict:
    """Compare 15-component HyQMOM15 width to 5-component Euler width."""
    hyqmom15 = _exact.bytes_per_cell(_exact.N_COMPONENTS)
    euler = _exact.euler_bytes_per_cell()
    return {
        "hyqmom15_components": int(_exact.N_COMPONENTS),
        "hyqmom15_bytes_per_cell": hyqmom15,
        "euler_components": int(_exact.EULER_COMPONENTS),
        "euler_bytes_per_cell": euler,
        "ratio": float(hyqmom15) / float(euler),
    }


def saxpy_fields(n_cells=_exact.N_CELLS, alpha=_exact.SAXPY_ALPHA) -> dict:
    """Return a = 2 * b on a (n, 15) state (SAXPY with a zero destination)."""
    b = _exact.interior_state(n_cells)
    a = _exact.saxpy(b, alpha)
    return {
        "a": a,
        "b": b,
        "volumes": _exact.cell_volumes(n_cells),
        "alpha": float(alpha),
    }


def public_hyqmom15_case():
    """Return a public 15-component Case factory, or ``None``.

    ``pops.lib.models.moments.HyQMOM15`` is a Model factory
    (``vlasov_lorentz``), not a Case. No verification case authors a
    15-component Case with ``run_native``.
    """
    try:
        from pops.lib.models.moments import HyQMOM15
    except ImportError:
        return None
    for name in ("build_case", "case", "run_native"):
        candidate = getattr(HyQMOM15, name, None)
        if callable(candidate):
            return candidate
    return None


def refuse_public_hyqmom15_case() -> str:
    """Return why a native 15-component timer cannot run."""
    if public_hyqmom15_case() is None:
        return HYQMOM15_CASE_REFUSAL
    return "public 15-component HyQMOM15 Case cannot be driven without inventing authoring"



def run_native(*args, **kwargs):
    """PF timed work belongs to benchmarks/manifest.toml, not a sibling wrap."""
    from verification.pops_verify.official_benchmark import refuse_unofficial_pf

    raise NativeUnavailable(refuse_unofficial_pf('PF-12'))

