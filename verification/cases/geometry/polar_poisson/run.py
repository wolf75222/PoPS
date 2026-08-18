"""GE-01 Cartesian Poisson authoring and public polar-runtime refusal.

The manufactured harmonic is evaluated in-memory. ``build_case`` authors a
public 2-d Cartesian Poisson Case for the Cartesian equivalent. The public
polar System is not active; ``refuse_public_polar_runtime`` returns the
documented reason. ``run_native`` raises ``NativeUnavailable`` with that
same string. There is no public PolarMesh runtime.
"""
from __future__ import annotations

from pathlib import Path

from verification.pops_verify.case_authoring import load_sibling_module

_exact = load_sibling_module(Path(__file__).with_name("exact.py"))

N_CELLS = int(_exact.N_R)
POLAR_RUNTIME_REFUSAL = "public polar System not active"


class NativeUnavailable(RuntimeError):
    """Native compile/run is not available in this environment."""


class AuthoringPending(RuntimeError):
    """Raised only if public elliptic validate/resolve cannot be completed."""


def refuse_public_polar_runtime() -> str:
    """Return the documented reason the public polar System is not active."""
    return POLAR_RUNTIME_REFUSAL


def _box_frame():
    from pops.domain import CartesianDomain
    from pops.frames import Cartesian2D

    return CartesianDomain(
        "ge01-box",
        (-float(_exact.R_MAX), -float(_exact.R_MAX)),
        (float(_exact.R_MAX), float(_exact.R_MAX)),
    ).frame(Cartesian2D())


def build_case(n_cells: int = N_CELLS):
    """Author a 2-d Cartesian Poisson Case for the Cartesian equivalent.

    Does not compile, bind, or launch a solver. Polar System authoring is
    refused by ``refuse_public_polar_runtime``.
    """
    from pops.fields import (
        CellCenteredSecondOrder,
        FieldDiscretization,
        FieldOutput,
        GradientOutput,
    )
    from pops.fields.bcs import AllPhysicalBoundaries, BoundaryCondition, Dirichlet
    from pops.math import laplacian
    from pops.physics import Model
    from pops.problem import Case
    from pops.solvers.elliptic import GeometricMG

    del n_cells
    frame = _box_frame()
    model = Model("ge01_polar_poisson", frame=frame)
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
    case = Case("ge01-polar-poisson")
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


def resolve_plan(n_cells: int = N_CELLS):
    """Validate the public Cartesian Case. Polar resolve is refused.

    ``pops.validate`` succeeds. ``pops.resolve`` requires a whole-system
    Program for Uniform layouts. This case does not invent a hyperbolic
    stepper or a private elliptic solver.
    """
    import pops

    pops.validate(build_case(n_cells))
    raise AuthoringPending(
        "GE-01 Cartesian Case validates; resolve needs a whole-system Program "
        "(no invented time stepper or private elliptic solver). "
        + refuse_public_polar_runtime()
    )


def run_native(n_cells: int = N_CELLS):
    """Refuse native polar compile. Cartesian authoring stays in ``build_case``."""
    del n_cells
    raise NativeUnavailable(refuse_public_polar_runtime())
