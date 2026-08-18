"""Public 2-d Dirichlet Poisson authoring for PO-02.

Does not compile, bind, or launch a solver. Native elliptic execution is not
implemented in this worktree (no private solver).
"""
from __future__ import annotations

from pathlib import Path

import pops
from pops.domain import Rectangle
from pops.fields import (
    CellCenteredSecondOrder,
    FieldDiscretization,
    FieldOutput,
    GradientOutput,
)
from pops.fields.bcs import AllPhysicalBoundaries, BoundaryCondition, Dirichlet
from pops.frames import Cartesian2D
from pops.math import laplacian
from pops.physics import Model
from pops.problem import Case
from pops.solvers.elliptic import GeometricMG

from verification.pops_verify.case_authoring import load_sibling_module

_EXACT = load_sibling_module(Path(__file__).with_name("exact.py"))


class AuthoringPending(RuntimeError):
    """Raised only if public elliptic validate/resolve cannot be completed."""


def build_rhs_and_oracle(n_cells: int):
    """Return in-memory cell-center RHS and exact φ, E on a uniform 2-d grid."""
    x, y, volumes = _EXACT.uniform_cell_grid(n_cells)
    ex, ey = _EXACT.e_exact(x, y)
    return {
        "x": x,
        "y": y,
        "volumes": volumes,
        "phi": _EXACT.phi_exact(x, y),
        "rhs": _EXACT.rhs_exact(x, y),
        "e": (ex, ey),
    }


def build_case(n_cells: int) -> Case:
    """Author a 2-d Dirichlet Poisson Case: -laplacian(phi) == rhs, GeometricMG."""
    del n_cells
    frame = Rectangle("po02-domain", lower=(0.0, 0.0), upper=(1.0, 1.0)).frame(
        Cartesian2D()
    )
    model = Model("po02_dirichlet_mms", frame=frame)
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
    case = Case("po02-dirichlet-mms")
    case.block("electrostatic", model)
    case.field(
        operator,
        FieldDiscretization(
            method=CellCenteredSecondOrder(),
            boundaries=(BoundaryCondition(AllPhysicalBoundaries(), Dirichlet(0.0)),),
            solver=GeometricMG(),
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
        "PO-02 Case validates; resolve needs a whole-system Program "
        "(no invented time stepper or private elliptic solver)"
    )
