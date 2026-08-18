"""Public 1-d Poisson authoring for PO-04.

Uniform FFT is a reduced substitute. Campaign ``run_native(request=)`` refuses
the normative Huang–Greengard AMR ID.
"""
from __future__ import annotations

from pathlib import Path

from pops.problem import Case
from verification.pops_verify.case_authoring import load_sibling_module

_CASE_DIR = Path(__file__).resolve().parent


class AuthoringPending(RuntimeError):
    """Raised only if public elliptic validate/resolve cannot be completed."""


class NativeUnavailable(RuntimeError):
    """Raised when a native PO-04 run cannot start honestly."""


def _exact_module():
    return load_sibling_module(_CASE_DIR / "exact.py")


def build_rhs_and_oracle(n_cells: int):
    """Return in-memory cell-center RHS and exact φ, E on a uniform 1-d grid."""
    exact = _exact_module()
    centers, volumes = exact.uniform_cell_grid(n_cells)
    return {
        "x": centers,
        "volumes": volumes,
        "phi": exact.phi_exact(centers),
        "rhs": exact.rhs_exact(centers),
        "e": exact.e_exact(centers),
    }


def _poisson_kwargs(n_cells: int) -> dict:
    from pops.solvers.elliptic import FFT

    return {
        "case_name": "po04-huang-greengard",
        "model_name": "po04_huang_greengard",
        "domain_name": "po04-domain",
        "solver": FFT(),
        "n_cells": n_cells,
    }


def build_case(n_cells: int) -> Case:
    """Author the same Case used by resolve_plan. Uniform FFT is not PO-04 AMR."""
    from verification.pops_verify.elliptic_stationary import author_periodic_poisson

    return author_periodic_poisson(**_poisson_kwargs(n_cells))[0]


def resolve_plan(n_cells: int):
    """Validate and resolve the unified Case. Does not compile or call pops.run."""
    from verification.pops_verify.case_authoring import (
        resolve_case,
        uniform_periodic_layout,
    )
    from verification.pops_verify.elliptic_stationary import author_periodic_poisson

    case, _instance, frame, count = author_periodic_poisson(**_poisson_kwargs(n_cells))
    return resolve_case(case, layout=uniform_periodic_layout(frame, (count,)))


def run_native(n_cells: int = 16, t_end: float = 1.0, *, request=None):
    """Refuse the normative PO-04 AMR claim. Uniform FFT is not Huang–Greengard AMR."""
    from verification.pops_verify.native_evidence import REDUCED_NOT_SUPPORTED

    del n_cells, t_end
    if request is not None and int(request.pops_native_dim) != 1:
        raise NativeUnavailable(
            f"PO-04 requires pops_native_dim=1 (got {request.pops_native_dim}); "
            "no fallback"
        )
    raise NativeUnavailable(REDUCED_NOT_SUPPORTED["PO-04"])
