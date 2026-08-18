"""Public 1-d periodic Poisson authoring for PO-05.

Does not compile, bind, or launch a solver. Native elliptic execution is not
implemented in this worktree (no private solver). The FFT vs GMG comparison
is the in-memory spectral solve versus the discrete -Δ residual stub.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import pops
from pops.domain import CartesianDomain
from pops.fields import (
    CellCenteredSecondOrder,
    ConstantNullspace,
    FieldDiscretization,
    FieldOutput,
    GradientOutput,
    MeanValueGauge,
)
from pops.fields.bcs import AllPhysicalBoundaries, BoundaryCondition, Periodic
from pops.frames import Cartesian1D
from pops.math import laplacian
from pops.physics import Model
from pops.problem import Case
from pops.solvers.elliptic import FFT, GeometricMG
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


def build_case(n_cells: int = N_CELLS) -> Case:
    """Author a 1-d periodic Poisson Case: -laplacian(phi) == rhs, FFT."""
    del n_cells
    return _build_case("po05-fft-vs-gmg", "po05_fft_vs_gmg", FFT())


def build_gmg_case(n_cells: int = N_CELLS) -> Case:
    """Author the same Case with GeometricMG (public API counterpart)."""
    del n_cells
    return _build_case("po05-fft-vs-gmg-gmg", "po05_fft_vs_gmg", GeometricMG())


def _build_case(case_name: str, model_name: str, solver) -> Case:
    frame = CartesianDomain("po05-domain", (0.0,), (1.0,)).frame(Cartesian1D())
    model = Model(model_name, frame=frame)
    state = model.state("U", components=["rhs"])
    (rhs,) = state
    potential = model.field("potential")
    operator = model.field_operator(
        "poisson",
        unknown=potential,
        equation=(-laplacian(potential) == rhs),
        outputs=(
            FieldOutput("potential", potential),
            GradientOutput("electric_field", potential, sign=-1),
        ),
    )
    case = Case(case_name)
    case.block("electrostatic", model)
    case.field(
        operator,
        FieldDiscretization(
            method=CellCenteredSecondOrder(),
            boundaries=(BoundaryCondition(AllPhysicalBoundaries(), Periodic()),),
            solver=solver,
            nullspace=ConstantNullspace(),
            gauge=MeanValueGauge(0.0),
        ),
    )
    return case


def resolve_plan(n_cells: int = N_CELLS):
    """Validate the public Cases. Full resolve is pending a whole-system Program.

    ``pops.validate`` succeeds for both the FFT and GeometricMG Cases.
    ``pops.resolve`` requires a whole-system Program for Uniform layouts.
    This case does not invent a hyperbolic stepper or a private elliptic
    solver, so resolve stays documented as AuthoringPending.
    """
    from pops.solvers.elliptic import FFT
    from verification.pops_verify.case_authoring import (
        resolve_case,
        uniform_periodic_layout,
    )
    from verification.pops_verify.elliptic_stationary import author_periodic_poisson

    case, _instance, frame, count = author_periodic_poisson(
        case_name="po05-fft-vs-gmg",
        model_name="po05_fft_vs_gmg",
        domain_name="po05-domain",
        solver=FFT(),
        n_cells=n_cells,
    )
    return resolve_case(case, layout=uniform_periodic_layout(frame, (count,)))


def run_native(n_cells: int = N_CELLS, t_end: float = 1.0, *, request=None):
    """Run uniform FFT and GeometricMG. GMG on a uniform System is CartesianCG."""
    from pops.solvers.elliptic import FFT, GeometricMG
    from verification.pops_verify.elliptic_stationary import run_periodic_poisson_native
    from verification.pops_verify.native_evidence import (
        maybe_campaign_payload,
        resolution_from_request,
    )

    if request is not None and int(request.pops_native_dim) != 1:
        raise NativeUnavailable(
            f"PO-05 requires pops_native_dim=1 (got {request.pops_native_dim}); "
            "no fallback"
        )
    n_cells = resolution_from_request(request, n_cells)
    sample = build_rhs_and_oracle(n_cells)
    try:
        phi_fft = run_periodic_poisson_native(
            case_name="po05-fft-vs-gmg",
            model_name="po05_fft_vs_gmg",
            domain_name="po05-domain",
            solver=FFT(),
            n_cells=n_cells,
            rhs=sample["rhs"],
            t_end=t_end,
        )
        phi_gmg = run_periodic_poisson_native(
            case_name="po05-fft-vs-gmg-gmg",
            model_name="po05_fft_vs_gmg",
            domain_name="po05-domain",
            solver=GeometricMG(),
            n_cells=n_cells,
            rhs=sample["rhs"],
            t_end=t_end,
        )
    except RuntimeError as exc:
        raise NativeUnavailable(str(exc)) from exc
    pair = {"fft": phi_fft, "gmg": phi_gmg}
    return maybe_campaign_payload(
        request,
        pair,
        n_cells=n_cells,
        t_end=t_end,
        time_program="ForwardEuler",
        cfl=1.0,
        dimension=1,
    )
