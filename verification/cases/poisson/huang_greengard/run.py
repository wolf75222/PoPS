"""Public 1-d periodic Poisson authoring for PO-04.

Does not compile, bind, or launch a solver. Native elliptic execution is not
implemented in this worktree (no private solver).
"""
from __future__ import annotations

from pathlib import Path

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
from pops.solvers.elliptic import FFT
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


def build_case(n_cells: int) -> Case:
    """Author a 1-d periodic Poisson Case: -laplacian(phi) == rhs, FFT, periodic."""
    del n_cells
    frame = CartesianDomain("po04-domain", (0.0,), (1.0,)).frame(Cartesian1D())
    model = Model("po04_huang_greengard", frame=frame)
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
    case = Case("po04-huang-greengard")
    case.block("electrostatic", model)
    case.field(
        operator,
        FieldDiscretization(
            method=CellCenteredSecondOrder(),
            boundaries=(BoundaryCondition(AllPhysicalBoundaries(), Periodic()),),
            solver=FFT(),
            nullspace=ConstantNullspace(),
            gauge=MeanValueGauge(0.0),
        ),
    )
    return case


def resolve_plan(n_cells: int):
    """Validate the public Case. Full resolve is pending a whole-system Program.

    ``pops.validate`` succeeds. ``pops.resolve`` requires a whole-system Program
    for Uniform layouts. This case does not invent a hyperbolic stepper or a
    private elliptic solver, so resolve stays documented as AuthoringPending.
    """
    from pops.solvers.elliptic import FFT
    from verification.pops_verify.case_authoring import (
        resolve_case,
        uniform_periodic_layout,
    )
    from verification.pops_verify.elliptic_stationary import author_periodic_poisson

    case, _instance, frame, count = author_periodic_poisson(
        case_name="po04-huang-greengard",
        model_name="po04_huang_greengard",
        domain_name="po04-domain",
        solver=FFT(),
        n_cells=n_cells,
    )
    return resolve_case(case, layout=uniform_periodic_layout(frame, (count,)))


def run_native(n_cells: int = 16, t_end: float = 1.0, *, request=None):
    """Compile and run a uniform FFT of the HG blobs. AMR composite is not claimed."""
    from pops.solvers.elliptic import FFT
    from verification.pops_verify.elliptic_stationary import run_periodic_poisson_native
    from verification.pops_verify.native_evidence import (
        maybe_campaign_payload,
        resolution_from_request,
    )

    if request is not None and int(request.pops_native_dim) != 1:
        raise NativeUnavailable(
            f"PO-04 requires pops_native_dim=1 (got {request.pops_native_dim}); "
            "no fallback"
        )
    n_cells = resolution_from_request(request, n_cells)
    sample = build_rhs_and_oracle(n_cells)
    try:
        phi = run_periodic_poisson_native(
            case_name="po04-huang-greengard",
            model_name="po04_huang_greengard",
            domain_name="po04-domain",
            solver=FFT(),
            n_cells=n_cells,
            rhs=sample["rhs"],
            t_end=t_end,
        )
    except RuntimeError as exc:
        raise NativeUnavailable(str(exc)) from exc
    return maybe_campaign_payload(
        request,
        phi,
        n_cells=n_cells,
        t_end=t_end,
        time_program="ForwardEuler",
        cfl=1.0,
        dimension=1,
    )
