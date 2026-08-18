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
    pops.validate(build_case(n_cells))
    raise AuthoringPending(
        "PO-04 Case validates; resolve needs a whole-system Program "
        "(no invented time stepper or private elliptic solver)"
    )
