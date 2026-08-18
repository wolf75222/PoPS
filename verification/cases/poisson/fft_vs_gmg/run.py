"""Public 1-d Poisson authoring for PO-05.

Uniform FFT and GeometricMG (CartesianCG on uniform Systems) are reduced
substitutes. Campaign ``run_native(request=)`` refuses the FFT-vs-GMG ID.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from pops.problem import Case
from pops.solvers.elliptic import GeometricMG
from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.reference_errors import reference_errors

_CASE_DIR = Path(__file__).resolve().parent
N_CELLS = 32


class AuthoringPending(RuntimeError):
    """Raised only if public elliptic validate/resolve cannot be completed."""


class NativeUnavailable(RuntimeError):
    """Raised when a native PO-05 run cannot start honestly."""


def _exact_module():
    return load_sibling_module(_CASE_DIR / "exact.py")


def build_rhs_and_oracle(n_cells: int = N_CELLS):
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


def spectral_cross_oracle(n_cells: int = N_CELLS):
    """Compare spectral φ to analytic φ after mean-mode removal."""
    exact = _exact_module()
    sample = build_rhs_and_oracle(n_cells)
    phi_spectral = exact.spectral_solve(sample["rhs"])
    phi_s = exact.mean_free(phi_spectral, sample["volumes"])
    phi_a = exact.mean_free(sample["phi"], sample["volumes"])
    potential = reference_errors(phi_s, phi_a, sample["volumes"])
    residual = exact.gmg_stub_residual(sample["phi"], sample["rhs"])
    residual_err = reference_errors(
        residual, np.zeros_like(residual), sample["volumes"]
    )
    sample["phi_spectral"] = phi_spectral
    sample["phi_spectral_mean_free"] = phi_s
    sample["phi_analytic_mean_free"] = phi_a
    sample["potential_error"] = potential
    sample["gmg_residual"] = residual
    sample["residual_error"] = residual_err
    return sample


def _poisson_kwargs(n_cells: int, *, solver=None, case_name: str = "po05-fft-vs-gmg") -> dict:
    from pops.solvers.elliptic import FFT

    return {
        "case_name": case_name,
        "model_name": "po05_fft_vs_gmg",
        "domain_name": "po05-domain",
        "solver": FFT() if solver is None else solver,
        "n_cells": n_cells,
    }


def build_case(n_cells: int = N_CELLS) -> Case:
    """Author the same FFT Case used by resolve_plan."""
    from verification.pops_verify.elliptic_stationary import author_periodic_poisson

    return author_periodic_poisson(**_poisson_kwargs(n_cells))[0]


def build_gmg_case(n_cells: int = N_CELLS) -> Case:
    """Author the GeometricMG counterpart (lowers to CartesianCG on uniform)."""
    from verification.pops_verify.elliptic_stationary import author_periodic_poisson

    return author_periodic_poisson(
        **_poisson_kwargs(n_cells, solver=GeometricMG(), case_name="po05-fft-vs-gmg-gmg")
    )[0]


def resolve_plan(n_cells: int = N_CELLS):
    """Validate and resolve the unified FFT Case. Does not compile or call pops.run."""
    from verification.pops_verify.case_authoring import (
        resolve_case,
        uniform_periodic_layout,
    )
    from verification.pops_verify.elliptic_stationary import author_periodic_poisson

    case, _instance, frame, count = author_periodic_poisson(**_poisson_kwargs(n_cells))
    return resolve_case(case, layout=uniform_periodic_layout(frame, (count,)))


def run_native(n_cells: int = N_CELLS, t_end: float = 1.0, *, request=None):
    """Refuse FFT-vs-GMG. Uniform GeometricMG is CartesianCG, not the PO-05 ID."""
    from verification.pops_verify.native_evidence import REDUCED_NOT_SUPPORTED

    del n_cells, t_end
    if request is not None and int(request.pops_native_dim) != 1:
        raise NativeUnavailable(
            f"PO-05 requires pops_native_dim=1 (got {request.pops_native_dim}); "
            "no fallback"
        )
    raise NativeUnavailable(REDUCED_NOT_SUPPORTED["PO-05"])
